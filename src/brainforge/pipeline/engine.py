import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainforge.dataset.writer import DatasetRecord
    from brainforge.domains.base import GateResult
    from brainforge.rag.retrieval import Retriever

from brainforge.case import Case
from brainforge.config.models import Config, PipelineDef
from brainforge.domains.base import DomainPack
from brainforge.errors import ProviderError
from brainforge.pipeline.judge import compute_agreement, is_unresolved
from brainforge.providers.base import ChatMessage, ChatRequest, Provider
from brainforge.providers.observability import UsageLogger, estimate_cost
from brainforge.roles.registry import RoleRegistry
from brainforge.types import RoleKind


@dataclass
class PipelineContext:
    case: Case
    rag_context: list = field(default_factory=list)
    previous_results: dict = field(default_factory=dict)
    kd_positions: list = field(default_factory=list)
    raw_assistant: str | None = None


@dataclass
class CaseResult:
    case_id: str
    accepted: bool
    record: "DatasetRecord | None" = None
    gate_reasons: list[str] = field(default_factory=list)
    teacher_results: dict = field(default_factory=dict)
    judge_result: dict | None = None
    metadata: dict = field(default_factory=dict)
    kd_positions: list | None = None
    kd_model: str | None = None
    provider_error: str | None = None


class PipelineEngine:
    def __init__(
        self,
        config: Config,
        pipeline_name: str,
        roles: RoleRegistry,
        pack: DomainPack,
        retriever: "Retriever | None" = None,
        usage_logger: UsageLogger | None = None,
        provider_override: Provider | None = None,
        pipeline: "PipelineDef | None" = None,
        kd: bool = False,
        kd_topk: int = 8,
    ):
        if pipeline is not None:
            self.pipeline = pipeline
        elif pipeline_name in config.pipelines:
            self.pipeline = config.pipelines[pipeline_name]
        else:
            raise KeyError(f"pipeline '{pipeline_name}' not found in configuration")
        self.config = config
        self.pipeline_name = pipeline_name
        self.roles = roles
        self.model_registry = roles.model_registry
        self.pack = pack
        self.retriever = retriever
        self.usage_logger = usage_logger
        self.steps = self.pipeline.steps or pack.steps_for_mode(self.pipeline.mode)
        kinds = [roles.resolve(step).kind for step in self.steps]
        if RoleKind.JUDGE in kinds and kinds[-1] != RoleKind.JUDGE:
            raise ValueError(
                f"pipeline '{pipeline_name}' must end with its judge role "
                f"(found '{self.steps[-1]}')"
            )
        self.retry_critic = self._resolve_retry_critic(roles, pipeline_name)
        self.provider_override = provider_override
        self.kd = kd
        self.kd_topk = kd_topk

    def _resolve_retry_critic(self, roles: RoleRegistry, pipeline_name: str) -> str | None:
        """Validate the optional escalation role: it must exist and be a critic."""
        name = self.pipeline.retry_with_critic
        if name is None:
            return None
        if name not in self.config.roles:
            raise ValueError(
                f"pipeline '{pipeline_name}': retry_with_critic role '{name}' "
                "is not defined in the configuration"
            )
        if roles.resolve(name).kind is not RoleKind.CRITIC:
            raise ValueError(
                f"pipeline '{pipeline_name}': retry_with_critic role '{name}' "
                "must resolve to a critic role"
            )
        return name

    def run_case(self, case: Case) -> CaseResult:
        context = PipelineContext(case=case)
        context.rag_context = self._retrieve(case)
        for step in self.steps:
            try:
                context.previous_results[step] = self._run_step(step, case, context)
            except ProviderError as exc:
                # Infrastructure failure, not case unreliability: skip the case
                # instead of aborting the whole batch, and never fabricate a
                # fallback record from it.
                return CaseResult(
                    case_id=case.id,
                    accepted=False,
                    gate_reasons=[f"provider_error: {step}: {exc}"],
                    teacher_results=self._teacher_results(context),
                    provider_error=str(exc),
                )
        final_role = self.steps[-1]
        final_binding = self.roles.resolve(final_role)
        canonical = context.previous_results[final_role]
        gate = self.pack.quality_gate(
            final_binding.kind, canonical, self.pipeline.min_confidence, case=case
        )
        accepted, reasons = self._gate_decision(case, canonical, gate)
        if not accepted and (critic := self.retry_critic) is not None:
            # Escalation: the adversarial critique gets a look, then the judge
            # re-arbitrates with it. The critic never sees the first verdict
            # (_teacher_results filters judge outputs), so it is not anchored.
            first_pass_reasons = list(reasons)
            retry_error = self._run_escalation(case, context, critic, final_role)
            if retry_error is None:
                canonical = context.previous_results[final_role]
                gate = self.pack.quality_gate(
                    final_binding.kind, canonical, self.pipeline.min_confidence, case=case
                )
                accepted, reasons = self._gate_decision(case, canonical, gate)
                metadata = self._metadata(case, context, gate)
                metadata["critic_retry"] = True
                metadata["first_pass_reasons"] = first_pass_reasons
            else:
                return CaseResult(
                    case_id=case.id,
                    accepted=False,
                    gate_reasons=[f"provider_error: {retry_error}"],
                    teacher_results=self._teacher_results(context),
                    provider_error=retry_error,
                )
        else:
            metadata = self._metadata(case, context, gate)
        if accepted:
            record = self._record(case, canonical, metadata, raw_assistant=context.raw_assistant)
            return CaseResult(
                case_id=case.id,
                accepted=True,
                record=record,
                teacher_results=self._teacher_results(context),
                judge_result=canonical,
                metadata=metadata,
                kd_positions=context.kd_positions or None,
                kd_model=(metadata["judge"]["model"] if metadata.get("judge") else None),
            )
        if self.pipeline.fallback_examples:
            record = self._fallback_record(case, metadata, reasons)
            return CaseResult(
                case_id=case.id,
                accepted=False,
                record=record,
                gate_reasons=reasons,
                teacher_results=self._teacher_results(context),
                judge_result=canonical,
                metadata=metadata,
            )
        return CaseResult(
            case_id=case.id,
            accepted=False,
            gate_reasons=reasons,
            teacher_results=self._teacher_results(context),
            judge_result=canonical,
            metadata=metadata,
        )

    def _gate_decision(
        self, case: Case, canonical: dict, gate: "GateResult"
    ) -> tuple[bool, list[str]]:
        """Gate plus rejection policy: (accepted, reasons)."""
        accepted = gate.passed
        reasons = list(gate.reasons)
        if self.pipeline.reject_unresolved and is_unresolved(canonical):
            accepted = False
            reasons.append("unresolved_disagreement")
        recitation_risk, _reason = self._recitation_risk(case)
        if recitation_risk and self.pipeline.reject_recitation_risk:
            accepted = False
            reasons.append("recitation_risk")
        return accepted, reasons

    def _run_escalation(
        self, case: Case, context: PipelineContext, critic: str, final_role: str
    ) -> str | None:
        """Run critic then judge again; returns an error string on provider failure."""
        for step in (critic, final_role):
            try:
                context.previous_results[step] = self._run_step(step, case, context)
            except ProviderError as exc:
                return str(exc)
        return None

    def _teacher_results(self, context: PipelineContext) -> dict:
        return {
            step: data
            for step, data in context.previous_results.items()
            if self.roles.resolve(step).kind != RoleKind.JUDGE
        }

    def _run_step(self, step: str, case: Case, context: PipelineContext) -> dict:
        binding = self.roles.resolve(step)
        schema = self.pack.output_schema(binding.kind)
        if binding.kind == RoleKind.JUDGE:
            user_prompt = self.pack.judge_prompt(
                case, context.rag_context, self._teacher_results(context)
            )
        elif binding.kind == RoleKind.CRITIC:
            user_prompt = self.pack.critic_prompt(
                step, case, context.rag_context, self._teacher_results(context)
            )
        else:
            user_prompt = self.pack.teacher_prompt(step, case, context.rag_context)
        system_prompt = binding.system_prompt or self.pack.system_prompt(step, binding.kind)
        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ],
            temperature=binding.temperature,
            top_p=binding.top_p,
            max_tokens=binding.max_tokens,
            cache_extra=self._cache_extra(case, context),
        )
        provider = self.provider_override or binding.provider
        if self.kd and binding.kind == RoleKind.JUDGE:
            response = provider.structured_logprobs(
                request, binding.model_name, schema, top_logprobs=self.kd_topk
            )
            context.kd_positions = [position.model_dump() for position in (response.logprobs or [])]
            context.raw_assistant = response.content
        else:
            response = provider.structured(request, binding.model_name, schema)
        if self.usage_logger is not None:
            definition = self.model_registry.resolve(binding.model_key).definition
            self.usage_logger.log(
                response,
                role=step,
                cost_usd=estimate_cost(definition, response),
            )
        assert response.data is not None
        return response.data

    def _retrieve(self, case: Case) -> list:
        if self.retriever is None:
            return []
        query = case.input.description or case.input.code[:200]
        if not query:
            return []
        return self.retriever.search(query)

    def _cache_extra(self, case: Case, context: PipelineContext) -> str:
        rag_texts = "".join(getattr(chunk, "text", str(chunk)) for chunk in context.rag_context)
        payload = case.model_dump_json() + rag_texts
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _recitation_risk(self, case: Case) -> tuple[bool, str | None]:
        cutoffs = []
        for step in self.steps:
            binding = self.roles.resolve(step)
            if binding.kind == RoleKind.JUDGE:
                continue
            definition = self.model_registry.resolve(binding.model_key)
            cutoffs.append(definition.definition.knowledge_cutoff)
        source_date = case.source.date
        if not source_date:
            return True, "unknown_source_date"
        if not cutoffs or any(cutoff is None for cutoff in cutoffs):
            return True, "unknown_teacher_cutoff"
        cutoff_strings = [str(cutoff) for cutoff in cutoffs]
        source_month = source_date[:7]
        if source_month <= max(cutoff_strings):
            return True, "pre_cutoff_source"
        return False, None

    def _metadata(self, case: Case, context: PipelineContext, gate: "GateResult") -> dict:
        teachers = []
        judge_entry = None
        for step in self.steps:
            binding = self.roles.resolve(step)
            definition = self.model_registry.resolve(binding.model_key).definition
            entry = {
                "role": step,
                "kind": binding.kind.value,
                "provider": definition.provider,
                "model": definition.model,
                "family": definition.family,
            }
            if binding.kind == RoleKind.JUDGE:
                judge_entry = entry
            else:
                teachers.append(entry)
        risk, risk_reason = self._recitation_risk(case)
        canonical = context.previous_results[self.steps[-1]]
        final_is_judge = judge_entry is not None
        return {
            "pipeline": self.pipeline_name,
            "domain": self.pipeline.domain,
            "language": case.language,
            "source": case.source.model_dump(),
            "source_date": case.source.date,
            "recitation_risk": risk,
            "recitation_reason": risk_reason,
            "teachers": teachers,
            "judge": judge_entry,
            "teacher_agreement": (
                compute_agreement(canonical, self._teacher_results(context))
                if final_is_judge
                else {}
            ),
            "rag": {
                "used": bool(context.rag_context),
                "chunks": [
                    {
                        "source": getattr(chunk, "source", None),
                        "doc_id": getattr(chunk, "doc_id", None),
                        "chunk_id": getattr(chunk, "chunk_id", None),
                        "hash": getattr(chunk, "hash", None),
                        "score": getattr(chunk, "score", None),
                    }
                    for chunk in context.rag_context
                ],
            },
            "quality": {"passed": gate.passed, "reasons": gate.reasons},
            "created_at": datetime.now(UTC).isoformat(),
        }

    def _record(
        self, case: Case, canonical: dict, metadata: dict, raw_assistant: str | None = None
    ) -> "DatasetRecord":
        from brainforge.dataset.writer import DatasetRecord

        messages = self.pack.build_messages(self.pack.student_input(case), canonical, raw_assistant)
        return DatasetRecord(
            id=case.id,
            domain=self.pipeline.domain,
            messages=messages,
            metadata=metadata,
        )

    def _fallback_record(self, case: Case, metadata: dict, reasons: list[str]) -> "DatasetRecord":
        from brainforge.dataset.writer import build_fallback_record

        fallback_metadata = dict(metadata)
        fallback_metadata["fallback"] = True
        fallback_metadata["fallback_reason"] = "; ".join(reasons)
        return build_fallback_record(
            case=case,
            domain=self.pipeline.domain,
            student_input=self.pack.student_input(case),
            reasons=reasons,
            metadata=fallback_metadata,
        )


class _PipelineState:
    """Self-tracing structured state file for one ``run_pipeline`` call.

    Long runs used to be invisible: no record of which case was running,
    how many completed, accepted vs rejected (this hid a duplicate-instance
    race once). The file is written atomically (sibling temp file, then
    rename, same discipline as ``write_jsonl``) before and after every case,
    so progress is observable from another shell without touching the run.
    """

    def __init__(self, path: Path, total_cases: int) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cases: list[dict] = []
        self._done = 0
        self._accepted = 0
        self._rejected = 0
        self._errors = 0
        self._payload: dict = {
            "started_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "total_cases": total_cases,
            "done": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "current_case": None,
            "cases": self._cases,
        }
        self._flush()

    def _flush(self) -> None:
        self._payload["updated_at"] = datetime.now(UTC).isoformat()
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, self.path)

    def case_started(self, case_id: str) -> dict:
        with self._lock:
            self._payload["current_case"] = case_id
            entry: dict = {
                "id": case_id,
                "status": "running",
                "duration_ms": None,
                "gate_reasons": None,
            }
            self._cases.append(entry)
            self._flush()
            return entry

    def case_finished(self, entry: dict, status: str, duration_ms: int, gate_reasons: int) -> None:
        with self._lock:
            entry["status"] = status
            entry["duration_ms"] = duration_ms
            entry["gate_reasons"] = gate_reasons
            self._done += 1
            if status == "accepted":
                self._accepted += 1
            elif status == "error":
                self._errors += 1
            else:
                self._rejected += 1
            self._payload["done"] = self._done
            self._payload["accepted"] = self._accepted
            self._payload["rejected"] = self._rejected
            self._payload["errors"] = self._errors
            self._flush()

    def finish(self) -> None:
        with self._lock:
            self._payload["status"] = "done"
            self._payload["current_case"] = None
            self._flush()


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def run_pipeline(
    engine: PipelineEngine,
    cases: list[Case],
    output_path: Path | str,
    rejected_dir: Path | str | None = None,
    kd_sidecar: Path | str | None = None,
    state_path: Path | str | None = None,
) -> dict:
    """Run cases through the engine, optionally in parallel (pipeline.concurrency).

    Results are collected in submission order so dataset output stays
    deterministic. The dataset and the KD sidecar are buffered and written
    once, atomically, after the last case: the files always reflect exactly
    this run, so a crash plus rerun can neither duplicate lines nor leave a
    stale record behind (re-runs are cheap anyway thanks to the sqlite
    response cache). Rejected-case artifacts keep streaming per case.

    Progress is self-traced to a structured state file (``<output>.state.json``
    by default, override with ``state_path``): it is written atomically before
    and after every case, so long runs stay observable (which case is running,
    done/accepted/rejected/error counts) even from another shell.
    """
    from concurrent.futures import ThreadPoolExecutor

    from brainforge.dataset.writer import write_jsonl, write_rejected

    state_file = Path(state_path) if state_path is not None else Path(f"{output_path}.state.json")
    state = _PipelineState(state_file, len(cases))
    workers = max(getattr(engine.pipeline, "concurrency", 1) or 1, 1)

    def run_traced(case: Case) -> CaseResult:
        entry = state.case_started(case.id)
        started = time.perf_counter()
        try:
            result = engine.run_case(case)
        except Exception:
            state.case_finished(entry, "error", _elapsed_ms(started), 0)
            raise
        if result.provider_error is not None:
            status = "error"
        elif result.accepted:
            status = "accepted"
        else:
            status = "rejected"
        state.case_finished(entry, status, _elapsed_ms(started), len(result.gate_reasons))
        return result

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_traced, cases))
    else:
        results = [run_traced(case) for case in cases]
    accepted_count = 0
    rejected_count = 0
    fallback_count = 0
    error_count = 0
    records: list[DatasetRecord] = []
    sidecar_lines: list[str] = []
    sidecar_path = Path(kd_sidecar) if kd_sidecar is not None else None
    for case, result in zip(cases, results, strict=True):
        if result.provider_error is not None:
            error_count += 1
            if rejected_dir is not None:
                write_rejected(
                    rejected_dir,
                    engine.pipeline_name,
                    case,
                    result.gate_reasons,
                    result.judge_result,
                    result.metadata,
                )
            continue
        if result.accepted and result.record is not None:
            records.append(result.record)
            accepted_count += 1
            if sidecar_path is not None and result.kd_positions:
                sidecar_lines.append(
                    json.dumps(
                        {
                            "id": case.id,
                            "model": result.kd_model,
                            "positions": result.kd_positions,
                        },
                        ensure_ascii=False,
                    )
                )
        elif result.record is not None:
            # rejected case that became an abstention training example
            records.append(result.record)
            fallback_count += 1
            if rejected_dir is not None:
                write_rejected(
                    rejected_dir,
                    engine.pipeline_name,
                    case,
                    result.gate_reasons,
                    result.judge_result,
                    result.metadata,
                )
        else:
            rejected_count += 1
            if rejected_dir is not None:
                write_rejected(
                    rejected_dir,
                    engine.pipeline_name,
                    case,
                    result.gate_reasons,
                    result.judge_result,
                    result.metadata,
                )
    write_jsonl(output_path, records)
    if sidecar_path is not None:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text("".join(f"{line}\n" for line in sidecar_lines), encoding="utf-8")
    state.finish()
    return {
        "accepted": accepted_count,
        "rejected": rejected_count,
        "fallback": fallback_count,
        "errors": error_count,
    }

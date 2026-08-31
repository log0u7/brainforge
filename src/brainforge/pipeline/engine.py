import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel

from brainforge.case import Case
from brainforge.config.models import Config
from brainforge.domains.base import DomainPack
from brainforge.pipeline.context import PipelineContext
from brainforge.pipeline.judge import compute_agreement, is_unresolved
from brainforge.providers.base import ChatMessage, ChatRequest, Provider
from brainforge.providers.observability import UsageLogger, estimate_cost
from brainforge.roles.registry import RoleRegistry
from brainforge.types import RoleKind


@dataclass
class CaseResult:
    case_id: str
    accepted: bool
    record: BaseModel | None = None
    gate_reasons: list[str] = field(default_factory=list)
    teacher_results: dict = field(default_factory=dict)
    judge_result: dict | None = None
    metadata: dict = field(default_factory=dict)


class PipelineEngine:
    def __init__(
        self,
        config: Config,
        pipeline_name: str,
        roles: RoleRegistry,
        pack: DomainPack,
        retriever=None,
        usage_logger: UsageLogger | None = None,
        provider_override: Provider | None = None,
        pipeline=None,
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
        self.provider_override = provider_override

    def run_case(self, case: Case) -> CaseResult:
        context = PipelineContext(case=case)
        context.rag_context = self._retrieve(case)
        for step in self.steps:
            context.previous_results[step] = self._run_step(step, case, context)
        final_role = self.steps[-1]
        final_binding = self.roles.resolve(final_role)
        canonical = context.previous_results[final_role]
        gate = self.pack.quality_gate(final_binding.kind, canonical, self.pipeline.min_confidence)
        metadata = self._metadata(case, context, gate)
        accepted = gate.passed
        reasons = list(gate.reasons)
        if self.pipeline.reject_unresolved and is_unresolved(canonical):
            accepted = False
            reasons.append("unresolved_disagreement")
        if metadata["recitation_risk"] and self.pipeline.reject_recitation_risk:
            accepted = False
            reasons.append("recitation_risk")
        if accepted:
            record = self._record(case, canonical, metadata)
            return CaseResult(
                case_id=case.id,
                accepted=True,
                record=record,
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
        source_month = source_date[:7]
        if source_month <= max(cutoffs):
            return True, "pre_cutoff_source"
        return False, None

    def _metadata(self, case: Case, context: PipelineContext, gate) -> dict:
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

    def _record(self, case: Case, canonical: dict, metadata: dict) -> BaseModel:
        from brainforge.dataset.writer import build_record

        return build_record(
            case=case,
            domain=self.pipeline.domain,
            student_input=self.pack.student_input(case),
            canonical=canonical,
            metadata=metadata,
        )


def run_pipeline(
    engine: PipelineEngine,
    cases: list[Case],
    output_path,
    rejected_dir=None,
) -> dict:
    from brainforge.dataset.writer import append_record, write_rejected

    accepted_count = 0
    rejected_count = 0
    for case in cases:
        result = engine.run_case(case)
        if result.accepted and result.record is not None:
            append_record(output_path, result.record)
            accepted_count += 1
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
    return {"accepted": accepted_count, "rejected": rejected_count}

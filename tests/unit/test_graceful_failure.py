"""Graceful failure: fallback examples for rejected cases and per-case
provider-error isolation. See ADR-0016 and arXiv:2610.20401.
"""

import json

import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.config.models import Config, ProviderConfig
from brainforge.dataset.validation import validate_record
from brainforge.dataset.writer import build_fallback_record, read_jsonl
from brainforge.domains import get_pack
from brainforge.errors import ProviderError
from brainforge.models.registry import ModelRegistry
from brainforge.pipeline.engine import PipelineEngine, run_pipeline
from brainforge.providers.base import ChatResponse, Provider
from brainforge.roles.registry import RoleRegistry
from tests.conftest import raw_config


@pytest.fixture
def postcutoff_case():
    return Case(
        id="case-post",
        source=CaseSource(type="git", repository="repo-a", commit="abc", date="2026-08-15"),
        input=CaseInput(code="x = 1", description="recent case"),
        metadata={"language": "python"},
    )


class ScriptedProvider(Provider):
    def __init__(self, responses_by_model: dict[str, dict]):
        super().__init__("scripted", ProviderConfig(type="mock"))
        self.responses = responses_by_model

    def complete(self, request, model):
        raise NotImplementedError

    def structured(self, request, model, schema):
        data = self.responses[model]
        return ChatResponse(content=json.dumps(data), data=data, provider=self.name, model=model)


class FailingProvider(Provider):
    """Raises ProviderError on every call, like an unreachable endpoint."""

    def __init__(self):
        super().__init__("failing", ProviderConfig(type="mock"))

    def complete(self, request, model):
        raise ProviderError("endpoint unreachable")

    def structured(self, request, model, schema):
        raise ProviderError("endpoint unreachable")


def scripted_config():
    config_raw = raw_config()
    config_raw["pipelines"]["security_dataset"]["steps"] = [
        "security_teacher",
        "coding_teacher",
        "judge",
    ]
    return Config.model_validate(config_raw)


def unresolved_responses():
    return {
        "mock-security": {
            "summary": "s",
            "vulnerability_found": True,
            "confidence": 0.9,
            "evidence": [{"description": "e"}],
            "reasoning": "r",
        },
        "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
        "mock-judge": {
            "verdict": "insufficient_information",
            "confidence": 0.4,
            "reasoning": "not enough info",
            "disagreement": True,
        },
    }


def build_engine(config, provider=None):
    roles = RoleRegistry(config, ModelRegistry(config))
    return PipelineEngine(
        config,
        "security_dataset",
        roles,
        get_pack("security"),
        provider_override=provider,
    )


def test_fallback_examples_defaults_off():
    config = scripted_config()
    assert config.pipelines["security_dataset"].fallback_examples is False


def test_build_fallback_record_shape(postcutoff_case):
    record = build_fallback_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze this",
        reasons=["unresolved_disagreement"],
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": False, "reasons": ["unresolved_disagreement"]},
            "recitation_risk": False,
            "fallback": True,
            "fallback_reason": "unresolved_disagreement",
        },
    )
    assistant = json.loads(record.messages[-1].content)
    assert assistant["verdict"] == "insufficient_information"
    assert "unresolved_disagreement" in assistant["reasoning"]
    assert record.metadata["fallback"] is True
    assert validate_record(record.model_dump()) == []


def test_validation_accepts_fallback_without_passing_gate(postcutoff_case):
    record = build_fallback_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze this",
        reasons=["confidence 0.4 below threshold 0.5"],
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": False, "reasons": ["confidence 0.4 below threshold 0.5"]},
            "recitation_risk": False,
            "fallback": True,
            "fallback_reason": "confidence 0.4 below threshold 0.5",
        },
    )
    assert validate_record(record.model_dump()) == []


def test_validation_rejects_fallback_without_reason(postcutoff_case):
    record = build_fallback_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze this",
        reasons=["unresolved_disagreement"],
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": False, "reasons": ["unresolved_disagreement"]},
            "recitation_risk": False,
            "fallback": True,
        },
    )
    errors = validate_record(record.model_dump())
    assert any("fallback_reason" in error for error in errors)


def test_validation_still_requires_passing_gate_without_fallback(postcutoff_case):
    record = build_fallback_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze this",
        reasons=["unresolved_disagreement"],
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": False, "reasons": ["unresolved_disagreement"]},
            "recitation_risk": False,
        },
    )
    errors = validate_record(record.model_dump())
    assert any("quality gate did not pass" in error for error in errors)


def test_engine_emits_fallback_record_when_enabled(postcutoff_case):
    config = scripted_config()
    config.pipelines["security_dataset"].fallback_examples = True
    engine = build_engine(config, ScriptedProvider(unresolved_responses()))
    result = engine.run_case(postcutoff_case)
    assert not result.accepted
    assert result.record is not None
    assistant = json.loads(result.record.messages[-1].content)
    assert assistant["verdict"] == "insufficient_information"
    assert result.record.metadata["fallback"] is True
    assert validate_record(result.record.model_dump()) == []


def test_engine_no_fallback_record_when_disabled(postcutoff_case):
    config = scripted_config()
    engine = build_engine(config, ScriptedProvider(unresolved_responses()))
    result = engine.run_case(postcutoff_case)
    assert not result.accepted
    assert result.record is None


def test_run_pipeline_writes_fallback_records(postcutoff_case, tmp_path):
    config = scripted_config()
    config.pipelines["security_dataset"].fallback_examples = True
    engine = build_engine(config, ScriptedProvider(unresolved_responses()))
    output = tmp_path / "out.jsonl"
    rejected = tmp_path / "rejected"
    stats = run_pipeline(engine, [postcutoff_case], output, rejected)
    assert stats == {"accepted": 0, "rejected": 0, "fallback": 1, "errors": 0}
    records = read_jsonl(output)
    assert len(records) == 1
    assert (
        json.loads(records[0]["messages"][-1]["content"])["verdict"] == "insufficient_information"
    )
    # audit trail preserved
    assert (rejected / "security_dataset").is_dir()


def test_run_pipeline_counts_errors_separately(postcutoff_case, tmp_path):
    config = scripted_config()
    engine = build_engine(config, FailingProvider())
    output = tmp_path / "out.jsonl"
    stats = run_pipeline(engine, [postcutoff_case], output, tmp_path / "rejected")
    assert stats == {"accepted": 0, "rejected": 0, "fallback": 0, "errors": 1}
    assert read_jsonl(output) == []


def test_provider_error_does_not_abort_the_batch(postcutoff_case, tmp_path):
    """One unreachable case must not kill the whole run: the batch continues."""
    config = scripted_config()
    responses = {
        "mock-security": {
            "summary": "s",
            "vulnerability_found": True,
            "confidence": 0.9,
            "evidence": [{"description": "e"}],
            "reasoning": "r",
        },
        "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
        "mock-judge": {
            "verdict": "confirmed",
            "confidence": 0.9,
            "reasoning": "r",
            "cwe": "CWE-78",
            "evidence": [{"description": "e"}],
            "disagreement": False,
        },
    }

    class FlakyProvider(ScriptedProvider):
        def structured(self, request, model, schema):
            if model == "mock-security":
                raise ProviderError("endpoint unreachable")
            return super().structured(request, model, schema)

    engine = build_engine(config, FlakyProvider(responses))
    output = tmp_path / "out.jsonl"
    stats = run_pipeline(engine, [postcutoff_case], output, tmp_path / "rejected")
    assert stats["errors"] == 1
    assert stats["accepted"] == 0
    # the failing case is recorded with its reason, not silently dropped
    rejected_payload = json.loads(
        (tmp_path / "rejected" / "security_dataset" / "case-post.json").read_text(encoding="utf-8")
    )
    assert any("provider_error" in reason for reason in rejected_payload["gate_reasons"])


def test_provider_error_case_is_not_a_fallback_example(postcutoff_case, tmp_path):
    """Infrastructure failure is not unreliability: no abstention record."""
    config = scripted_config()
    config.pipelines["security_dataset"].fallback_examples = True
    engine = build_engine(config, FailingProvider())
    output = tmp_path / "out.jsonl"
    stats = run_pipeline(engine, [postcutoff_case], output, tmp_path / "rejected")
    assert stats == {"accepted": 0, "rejected": 0, "fallback": 0, "errors": 1}
    assert read_jsonl(output) == []


def test_fallback_record_user_message_is_the_student_input(postcutoff_case):
    record = build_fallback_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze this",
        reasons=["unresolved_disagreement"],
        metadata={"fallback": True, "fallback_reason": "unresolved_disagreement"},
    )
    assert record.messages[0].role == "user"
    assert record.messages[0].content == "Analyze this"
    assert [m.role for m in record.messages] == ["user", "assistant"]


def confirmed_responses():
    return {
        "mock-security": {
            "summary": "s",
            "vulnerability_found": True,
            "confidence": 0.9,
            "evidence": [{"description": "e"}],
            "reasoning": "r",
        },
        "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
        "mock-judge": {
            "verdict": "confirmed",
            "confidence": 0.9,
            "reasoning": "r",
            "cwe": "CWE-78",
            "evidence": [{"description": "e"}],
            "disagreement": False,
        },
    }


def test_rerun_does_not_duplicate_records(postcutoff_case, tmp_path):
    """A crash plus rerun used to append a second copy of every record."""
    config = scripted_config()
    engine = build_engine(config, ScriptedProvider(confirmed_responses()))
    output = tmp_path / "out.jsonl"
    first = run_pipeline(engine, [postcutoff_case], output, tmp_path / "rejected")
    second = run_pipeline(engine, [postcutoff_case], output, tmp_path / "rejected")
    assert first["accepted"] == 1
    assert second["accepted"] == 1
    assert len(read_jsonl(output)) == 1


def test_rerun_reflects_changed_responses(postcutoff_case, tmp_path):
    """Re-running after a prompt/config tweak must not keep stale records."""
    config = scripted_config()
    output = tmp_path / "out.jsonl"
    run_pipeline(
        build_engine(config, ScriptedProvider(confirmed_responses())),
        [postcutoff_case],
        output,
        tmp_path / "rejected",
    )
    # same cases, but the ensemble now rejects: the stale record must go
    run_pipeline(
        build_engine(config, ScriptedProvider(unresolved_responses())),
        [postcutoff_case],
        output,
        tmp_path / "rejected",
    )
    assert read_jsonl(output) == []


def test_partial_output_is_replaced_not_appended(postcutoff_case, tmp_path):
    """A half-written dataset from a crashed run must be fully replaced."""
    config = scripted_config()
    output = tmp_path / "out.jsonl"
    output.write_text('{"id": "case-crashed", "domain": "security"}\n', encoding="utf-8")
    run_pipeline(
        build_engine(config, ScriptedProvider(confirmed_responses())),
        [postcutoff_case],
        output,
        tmp_path / "rejected",
    )
    records = read_jsonl(output)
    assert [record["id"] for record in records] == ["case-post"]


class LogprobsScriptedProvider(ScriptedProvider):
    """Scripted provider that also serves per-token top-k logprobs."""

    def complete_logprobs(self, request, model, top_logprobs=8):
        from brainforge.providers.base import PositionLogprobs

        data = self.responses[model]
        content = json.dumps(data)
        positions = [
            PositionLogprobs(token=word, logprob=-0.1, topk=[(word, -0.1)])
            for word in content.split(" ")
        ]
        return ChatResponse(
            content=content,
            data=data,
            provider=self.name,
            model=model,
            logprobs=positions,
        )


def test_rerun_does_not_duplicate_kd_sidecar(postcutoff_case, tmp_path):
    config = scripted_config()
    engine = build_engine(config, LogprobsScriptedProvider(confirmed_responses()))
    engine.kd = True
    output = tmp_path / "out.jsonl"
    sidecar = tmp_path / "out.kd.jsonl"
    for _ in range(2):
        run_pipeline(engine, [postcutoff_case], output, tmp_path / "rejected", kd_sidecar=sidecar)
    assert len(sidecar.read_text(encoding="utf-8").strip().splitlines()) == 1

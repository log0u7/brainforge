"""Critic escalation: on gate rejection, run the critic then re-judge.

The one useful cycle for this pipeline (gate rejected -> adversarial critique
-> judge re-arbitrates), implemented inline in run_case without any graph
framework.
"""

import json

import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.config.models import Config, ProviderConfig
from brainforge.dataset.validation import validate_record
from brainforge.dataset.writer import read_jsonl
from brainforge.domains import get_pack
from brainforge.models.registry import ModelRegistry
from brainforge.pipeline.engine import PipelineEngine, run_pipeline
from brainforge.providers.base import ChatResponse, Provider
from brainforge.roles.registry import RoleRegistry
from tests.conftest import raw_config


@pytest.fixture
def escalation_case():
    return Case(
        id="case-esc",
        source=CaseSource(type="git", repository="repo-a", commit="abc", date="2026-08-15"),
        input=CaseInput(code="x = 1", description="a recent case"),
        metadata={"language": "python"},
    )


TEACHER_RESPONSES = {
    "mock-security": {
        "summary": "s",
        "vulnerability_found": True,
        "confidence": 0.9,
        "evidence": [{"description": "e"}],
        "reasoning": "r",
    },
    "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
    "mock-general": {
        "summary": "the teachers overstate the risk",
        "confidence": 0.6,
        "reasoning": "the evidence does not support the claim",
        "critique": "overstated",
    },
}

UNRESOLVED_JUDGE = {
    "verdict": "insufficient_information",
    "confidence": 0.4,
    "reasoning": "not enough info",
    "disagreement": True,
}

CONFIRMED_JUDGE = {
    "verdict": "confirmed",
    "confidence": 0.9,
    "reasoning": "r",
    "cwe": "CWE-78",
    "evidence": [{"description": "e"}],
    "disagreement": False,
}


class SequencedJudgeProvider(Provider):
    """Serves fixed teacher/critic responses and a queue of judge verdicts."""

    def __init__(self, judge_sequence: list[dict]):
        super().__init__("scripted", ProviderConfig(type="mock"))
        self.responses = TEACHER_RESPONSES
        self.judge_sequence = list(judge_sequence)
        self.calls: list[tuple[str, str]] = []

    def complete(self, request, model):
        raise NotImplementedError

    def structured(self, request, model, schema):
        self.calls.append((model, " ".join(f"{m.role}:{m.content}" for m in request.messages)))
        data = self.judge_sequence.pop(0) if model == "mock-judge" else self.responses[model]
        return ChatResponse(content=json.dumps(data), data=data, provider=self.name, model=model)


def escalation_config(**overrides) -> Config:
    config_raw = raw_config()
    config_raw["pipelines"]["security_dataset"]["steps"] = [
        "security_teacher",
        "coding_teacher",
        "judge",
    ]
    config_raw["roles"]["general_critic"] = {"model": "general"}
    config = Config.model_validate(config_raw)
    for key, value in overrides.items():
        setattr(config.pipelines["security_dataset"], key, value)
    return config


def build_engine(config, provider):
    roles = RoleRegistry(config, ModelRegistry(config))
    return PipelineEngine(
        config,
        "security_dataset",
        roles,
        get_pack("security"),
        provider_override=provider,
    )


def test_construction_rejects_unknown_retry_role(escalation_case):
    config = escalation_config(retry_with_critic="ghost_critic")
    with pytest.raises(ValueError, match="retry_with_critic"):
        build_engine(config, SequencedJudgeProvider([CONFIRMED_JUDGE]))


def test_construction_rejects_non_critic_retry_role(escalation_case):
    config = escalation_config(retry_with_critic="security_teacher")
    with pytest.raises(ValueError, match="critic"):
        build_engine(config, SequencedJudgeProvider([CONFIRMED_JUDGE]))


def test_escalation_accepts_after_critic(escalation_case):
    config = escalation_config(retry_with_critic="general_critic")
    provider = SequencedJudgeProvider([UNRESOLVED_JUDGE, CONFIRMED_JUDGE])
    result = build_engine(config, provider).run_case(escalation_case)
    assert result.accepted, result.gate_reasons
    assert json.loads(result.record.messages[-1].content)["verdict"] == "confirmed"
    assert result.record.metadata["critic_retry"] is True
    assert result.record.metadata["first_pass_reasons"]
    # the critic ran, then the judge ran a second time
    critic_calls = [model for model, _ in provider.calls if model == "mock-general"]
    judge_calls = [model for model, _ in provider.calls if model == "mock-judge"]
    assert critic_calls == ["mock-general"]
    assert judge_calls == ["mock-judge", "mock-judge"]


def test_critic_does_not_see_the_first_judge_verdict(escalation_case):
    config = escalation_config(retry_with_critic="general_critic")
    provider = SequencedJudgeProvider([UNRESOLVED_JUDGE, CONFIRMED_JUDGE])
    build_engine(config, provider).run_case(escalation_case)
    critic_messages = [text for model, text in provider.calls if model == "mock-general"]
    assert critic_messages
    assert "insufficient_information" not in critic_messages[0]
    # the second judge call does see the critique
    second_judge = [text for model, text in provider.calls if model == "mock-judge"][1]
    assert "overstated" in second_judge


def test_escalation_rejected_twice_falls_back(escalation_case, tmp_path):
    config = escalation_config(retry_with_critic="general_critic", fallback_examples=True)
    provider = SequencedJudgeProvider([UNRESOLVED_JUDGE, UNRESOLVED_JUDGE])
    result = build_engine(config, provider).run_case(escalation_case)
    assert not result.accepted
    assert result.record is not None
    assert json.loads(result.record.messages[-1].content)["verdict"] == "insufficient_information"
    assert result.record.metadata["critic_retry"] is True
    assert validate_record(result.record.model_dump()) == []


def test_no_escalation_without_the_knob(escalation_case):
    config = escalation_config()
    provider = SequencedJudgeProvider([UNRESOLVED_JUDGE])
    result = build_engine(config, provider).run_case(escalation_case)
    assert not result.accepted
    assert result.record is None
    assert "mock-general" not in [model for model, _ in provider.calls]
    assert [model for model, _ in provider.calls].count("mock-judge") == 1


def test_run_pipeline_with_escalation_writes_final_record(escalation_case, tmp_path):
    config = escalation_config(retry_with_critic="general_critic")
    provider = SequencedJudgeProvider([UNRESOLVED_JUDGE, CONFIRMED_JUDGE])
    output = tmp_path / "out.jsonl"
    stats = run_pipeline(build_engine(config, provider), [escalation_case], output)
    assert stats == {"accepted": 1, "rejected": 0, "fallback": 0, "errors": 0}
    records = read_jsonl(output)
    assert json.loads(records[0]["messages"][-1]["content"])["verdict"] == "confirmed"

"""Knowledge distillation plumbing tests (no torch: providers, sidecar, wiring).

The real loss math (KD cross-entropy + KL) is covered in the integration
suite (tests/integration/test_kd_chain.py, opt-in via BRAINFORCE_IT=1).
"""

import json

import pytest
from typer.testing import CliRunner

from brainforge.cli import app
from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError
from brainforge.pipeline.engine import PipelineEngine, run_pipeline
from brainforge.providers.base import ChatRequest, ChatResponse, PositionLogprobs, Provider
from brainforge.providers.mock import MockProvider
from brainforge.training.kd import align_teacher_positions

runner = CliRunner()

JUDGE_PAYLOAD = {
    "verdict": "confirmed",
    "cwe": "CWE-78",
    "confidence": 0.9,
    "evidence": [{"description": "e"}],
    "reasoning": "r",
    "disagreement": False,
}


class LogprobsScriptedProvider(Provider):
    """Scripted provider that also serves per-token top-k logprobs."""

    def __init__(self, responses_by_model: dict[str, dict], positions_by_model=None):
        super().__init__("scripted-kd", _mock_provider_config())
        self.responses = responses_by_model
        self.positions_by_model = positions_by_model or {}
        self.logprobs_calls: list[str] = []

    def complete(self, request, model):
        raise NotImplementedError

    def structured(self, request, model, schema):
        data = self.responses[model]
        return ChatResponse(content=json.dumps(data), data=data, provider=self.name, model=model)

    def complete_logprobs(self, request, model, top_logprobs=8):
        self.logprobs_calls.append(model)
        data = self.responses[model]
        positions = self.positions_by_model.get(model) or _fake_positions(json.dumps(data))
        return ChatResponse(
            content=json.dumps(data),
            data=data,
            provider=self.name,
            model=model,
            logprobs=positions,
        )


def _mock_provider_config():
    from brainforge.config.models import ProviderConfig

    return ProviderConfig(type="mock")


def _fake_positions(content: str) -> list[PositionLogprobs]:
    positions = []
    for word in content.split(" "):
        topk = [(word, -0.1), ("}", -3.0), ("{", -3.5)]
        positions.append(PositionLogprobs(token=word, logprob=-0.1, topk=topk))
    return positions


def _kd_config():
    from brainforge.config.models import Config
    from tests.conftest import raw_config

    config_raw = raw_config()
    config_raw["pipelines"]["security_dataset"]["steps"] = [
        "security_teacher",
        "coding_teacher",
        "judge",
    ]
    return Config.model_validate(config_raw)


def _case():
    from brainforge.case import Case, CaseInput, CaseSource

    return Case(
        id="case-kd",
        source=CaseSource(type="git", repository="repo", date="2026-08-15"),
        input=CaseInput(code="x = 1", description="a case"),
    )


def test_structured_logprobs_default_not_supported():
    from brainforge.config.models import ProviderConfig
    from brainforge.providers.base import ChatRequest as Req

    class Bare(Provider):
        def complete(self, request, model):
            raise NotImplementedError

    provider = Bare("bare", ProviderConfig(type="mock"))
    from pydantic import BaseModel

    class Schema(BaseModel):
        a: int

    with pytest.raises(BrainforgeError, match="does not support logprobs"):
        provider.complete_logprobs(Req(messages=[]), "m")


def test_structured_logprobs_parses_and_validates():
    from pydantic import BaseModel, Field

    class Schema(BaseModel):
        verdict: str
        confidence: float = Field(ge=0, le=1)

    data = {"verdict": "ok", "confidence": 0.9}
    provider = LogprobsScriptedProvider({"m": data}, {"m": _fake_positions(json.dumps(data))})
    request = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    response = provider.structured_logprobs(request, "m", Schema, top_logprobs=4)
    assert response.data == data
    assert response.logprobs is not None
    assert len(response.logprobs) > 0
    assert all(len(position.topk) <= 4 for position in response.logprobs)


def test_structured_logprobs_retries_invalid_json():
    from pydantic import BaseModel, Field

    class Schema(BaseModel):
        verdict: str
        confidence: float = Field(ge=0, le=1)

    class FlakyProvider(LogprobsScriptedProvider):
        def __init__(self):
            super().__init__({"m": {"verdict": "ok", "confidence": 0.9}})
            self.calls = 0

        def complete_logprobs(self, request, model, top_logprobs=8):
            self.calls += 1
            if self.calls == 1:
                return ChatResponse(content="not json", provider=self.name, model=model)
            return super().complete_logprobs(request, model, top_logprobs)

    flaky = FlakyProvider()
    response = flaky.structured_logprobs(
        ChatRequest(messages=[{"role": "user", "content": "hi"}]), "m", Schema
    )
    assert response.data["verdict"] == "ok"
    assert flaky.calls == 2


def test_mock_structured_logprobs_positions_rebuild_content():
    from pydantic import BaseModel, Field

    class Schema(BaseModel):
        verdict: str
        confidence: float = Field(ge=0, le=1)

    provider = MockProvider("mock", _mock_provider_config())
    response = provider.structured_logprobs(
        ChatRequest(messages=[{"role": "user", "content": "hi"}]), "m", Schema
    )
    rebuilt = "".join(position.token for position in response.logprobs)
    assert rebuilt == response.content


def test_engine_kd_mode_captures_judge_positions():
    config = _kd_config()
    from brainforge.domains import get_pack
    from brainforge.models.registry import ModelRegistry
    from brainforge.roles.registry import RoleRegistry

    provider = LogprobsScriptedProvider(
        {
            "mock-security": JUDGE_PAYLOAD,
            "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
            "mock-judge": JUDGE_PAYLOAD,
        }
    )
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(
        config,
        "security_dataset",
        roles,
        get_pack("security"),
        provider_override=provider,
        kd=True,
    )
    result = engine.run_case(_case())
    assert result.accepted, result.gate_reasons
    assert result.kd_positions
    assert provider.logprobs_calls == ["mock-judge"]
    assert result.kd_model == "mock-judge"


def test_engine_without_kd_does_not_call_logprobs():
    config = _kd_config()
    from brainforge.domains import get_pack
    from brainforge.models.registry import ModelRegistry
    from brainforge.roles.registry import RoleRegistry

    provider = LogprobsScriptedProvider(
        {
            "mock-security": JUDGE_PAYLOAD,
            "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
            "mock-judge": JUDGE_PAYLOAD,
        }
    )
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(
        config, "security_dataset", roles, get_pack("security"), provider_override=provider
    )
    result = engine.run_case(_case())
    assert result.accepted
    assert not result.kd_positions
    assert provider.logprobs_calls == []


def test_run_pipeline_writes_sidecar_for_accepted_records(tmp_path):
    config = _kd_config()
    from brainforge.domains import get_pack
    from brainforge.models.registry import ModelRegistry
    from brainforge.roles.registry import RoleRegistry

    provider = LogprobsScriptedProvider(
        {
            "mock-security": JUDGE_PAYLOAD,
            "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
            "mock-judge": JUDGE_PAYLOAD,
        }
    )
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(
        config,
        "security_dataset",
        roles,
        get_pack("security"),
        provider_override=provider,
        kd=True,
    )
    dataset = tmp_path / "ds.jsonl"
    sidecar = tmp_path / "ds.kd.jsonl"
    stats = run_pipeline(engine, [_case()], dataset, kd_sidecar=sidecar)
    assert stats["accepted"] == 1
    records = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["id"] == "case-kd"
    assert records[0]["model"] == "mock-judge"
    assert records[0]["positions"]


def test_run_pipeline_skips_sidecar_when_disabled(tmp_path):
    config = _kd_config()
    from brainforge.domains import get_pack
    from brainforge.models.registry import ModelRegistry
    from brainforge.roles.registry import RoleRegistry

    provider = LogprobsScriptedProvider(
        {
            "mock-security": JUDGE_PAYLOAD,
            "mock-coding": {"summary": "s", "behavior": "b", "confidence": 0.9, "reasoning": "r"},
            "mock-judge": JUDGE_PAYLOAD,
        }
    )
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(
        config,
        "security_dataset",
        roles,
        get_pack("security"),
        provider_override=provider,
        kd=True,
    )
    dataset = tmp_path / "ds.jsonl"
    run_pipeline(engine, [_case()], dataset)
    assert not (tmp_path / "ds.kd.jsonl").exists()


class FakeTokenizer:
    """Minimal tokenizer: each character is a token (1:1 char alignment)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return ord(token) if len(token) == 1 else None

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def decode(self, ids):
        return "".join(chr(i) for i in ids if isinstance(i, int))


def test_align_teacher_positions_maps_by_char_midpoint():
    teacher_positions = [
        {"token": "he", "logprob": -0.1, "topk": [["he", -0.1]]},
        {"token": "llo", "logprob": -0.2, "topk": [["llo", -0.2]]},
    ]
    content = "hello"
    full_text = "<user>hi<assistant>hello"
    content_start = full_text.rfind(content)
    aligned = align_teacher_positions(
        teacher_positions, content, content_start, full_text, FakeTokenizer()
    )
    # char tokens: "he" midpoint at content_start+1 -> 'e' at index 20, predictor 19;
    # "llo" midpoint at content_start+3.5 -> 'l' at index 22, predictor 21
    assert aligned == [(19, [["he", -0.1]]), (21, [["llo", -0.2]])]


def test_align_teacher_positions_empty():
    assert align_teacher_positions([], "abc", 0, "abc", FakeTokenizer()) == []


def test_kd_config_fields_defaults():
    config = TrainingConfig()
    assert config.kd_alpha == 0.5
    assert config.kd_temperature == 2.0
    assert config.kd_topk == 8


def test_kd_config_fields_validation():
    with pytest.raises(ValueError):
        TrainingConfig(kd_alpha=1.5)
    with pytest.raises(ValueError):
        TrainingConfig(kd_temperature=0)
    with pytest.raises(ValueError):
        TrainingConfig(kd_topk=0)


def test_cli_pipeline_kd_flag_wires_through(project_env, monkeypatch):
    calls = {}

    def fake_execute(cfg, pipeline_name, input_dir, output, **kwargs):
        calls["kd"] = kwargs.get("kd")
        return {"accepted": 0, "rejected": 0}, 0

    monkeypatch.setattr("brainforge.cli.pipeline_cmd.execute_pipeline", fake_execute)
    result = runner.invoke(app, ["pipeline", "run", "security_dataset", "--kd"])
    assert result.exit_code == 0, result.output
    assert calls["kd"] is True


def test_train_kd_requires_sidecar(tmp_path):
    from brainforge.training.kd import train_kd

    dataset_dir = tmp_path / "prepared"
    dataset_dir.mkdir()
    record = {"id": "x", "messages": [{"role": "user", "content": "hi"}], "metadata": {}}
    (dataset_dir / "train.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(BrainforgeError, match=r"kd\.jsonl"):
        train_kd(TrainingConfig(), dataset_dir, tmp_path / "run")

import json

import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.config.models import ProviderConfig
from brainforge.dataset.case_builder import build_case_from_dict, case_id_for
from brainforge.dataset.dedup import deduplicate
from brainforge.dataset.split import (
    group_key,
    is_postcutoff,
    postcutoff_warning,
    split_dataset,
    split_with_postcutoff,
)
from brainforge.dataset.validation import validate_dataset, validate_record
from brainforge.dataset.writer import (
    DatasetRecord,
    build_record,
    read_jsonl,
    write_jsonl,
)
from brainforge.domains import get_pack
from brainforge.models.registry import ModelRegistry
from brainforge.pipeline.engine import PipelineEngine, run_pipeline
from brainforge.providers.base import ChatMessage, Provider
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


@pytest.fixture
def precutoff_case():
    return Case(
        id="case-pre",
        source=CaseSource(type="git", repository="repo-b", commit="def", date="2026-01-10"),
        input=CaseInput(code="y = 2", description="old case"),
        metadata={"language": "python"},
    )


class ScriptedProvider(Provider):
    def __init__(self, responses_by_model: dict[str, dict]):
        super().__init__("scripted", ProviderConfig(type="mock"))
        self.responses = responses_by_model

    def complete(self, request, model):
        raise NotImplementedError

    def structured(self, request, model, schema):
        from brainforge.providers.base import ChatResponse

        data = self.responses[model]
        return ChatResponse(content=json.dumps(data), data=data, provider=self.name, model=model)


def scripted_config():
    config_raw = raw_config()
    config_raw["pipelines"]["security_dataset"]["steps"] = [
        "security_teacher",
        "coding_teacher",
        "judge",
    ]
    from brainforge.config.models import Config

    return Config.model_validate(config_raw)


def test_engine_runs_mock_pipeline_end_to_end(postcutoff_case, tmp_path):
    config = scripted_config()
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(
        config,
        "security_dataset",
        roles,
        get_pack("security"),
        usage_logger=None,
    )
    result = engine.run_case(postcutoff_case)
    assert result.accepted, result.gate_reasons
    record = result.record
    assert record.id == "case-post"
    assert record.domain == "security"
    assert record.messages[0].role == "user"
    assert record.messages[-1].role == "assistant"
    assistant = json.loads(record.messages[-1].content)
    assert assistant["verdict"] == "confirmed"
    assert assistant["cwe"] == "CWE-78"
    metadata = record.metadata
    assert metadata["recitation_risk"] is False
    assert metadata["judge"]["provider"] == "judge_p"
    assert [t["role"] for t in metadata["teachers"]] == [
        "security_teacher",
        "coding_teacher",
    ]
    assert metadata["quality"]["passed"] is True
    assert "rag" in metadata


def test_engine_rejects_unresolved(postcutoff_case):
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
            "verdict": "insufficient_information",
            "confidence": 0.4,
            "reasoning": "not enough info",
            "disagreement": True,
        },
    }
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(
        config,
        "security_dataset",
        roles,
        get_pack("security"),
        provider_override=ScriptedProvider(responses),
    )
    result = engine.run_case(postcutoff_case)
    assert not result.accepted
    assert "unresolved_disagreement" in result.gate_reasons


def test_engine_tags_recitation_risk(precutoff_case):
    config = scripted_config()
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(config, "security_dataset", roles, get_pack("security"))
    result = engine.run_case(precutoff_case)
    assert result.metadata["recitation_risk"] is True
    assert result.metadata["recitation_reason"] == "pre_cutoff_source"


def test_engine_rejects_recitation_risk_when_configured(precutoff_case):
    config = scripted_config()
    config.pipelines["security_dataset"].reject_recitation_risk = True
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(config, "security_dataset", roles, get_pack("security"))
    result = engine.run_case(precutoff_case)
    assert not result.accepted
    assert "recitation_risk" in result.gate_reasons


def test_engine_flags_unknown_dates(postcutoff_case):
    config = scripted_config()
    postcutoff_case.source.date = None
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(config, "security_dataset", roles, get_pack("security"))
    result = engine.run_case(postcutoff_case)
    assert result.metadata["recitation_risk"] is True
    assert result.metadata["recitation_reason"] == "unknown_source_date"


def test_engine_requires_judge_last(postcutoff_case):
    config = scripted_config()
    config.pipelines["security_dataset"].steps = ["judge", "security_teacher"]
    roles = RoleRegistry(config, ModelRegistry(config))
    with pytest.raises(ValueError, match="must end with its judge role"):
        PipelineEngine(config, "security_dataset", roles, get_pack("security"))


def test_engine_mode_steps_resolution():
    config = scripted_config()
    config.pipelines["security_dataset"].steps = None
    config.pipelines["security_dataset"].mode = "cheap"
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(config, "security_dataset", roles, get_pack("security"))
    assert engine.steps == ["security_teacher", "judge"]


def test_run_pipeline_writes_accepted_and_rejected(postcutoff_case, precutoff_case, tmp_path):
    config = scripted_config()
    config.pipelines["security_dataset"].reject_recitation_risk = True
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(config, "security_dataset", roles, get_pack("security"))
    output = tmp_path / "datasets" / "security.jsonl"
    rejected = tmp_path / "rejected"
    stats = run_pipeline(engine, [postcutoff_case, precutoff_case], output, rejected)
    assert stats == {"accepted": 1, "rejected": 1}
    records = read_jsonl(output)
    assert len(records) == 1
    assert records[0]["id"] == "case-post"
    assert (rejected / "security_dataset").is_dir()


def test_run_pipeline_without_rejected_dir(postcutoff_case, precutoff_case, tmp_path):
    config = scripted_config()
    config.pipelines["security_dataset"].reject_recitation_risk = True
    roles = RoleRegistry(config, ModelRegistry(config))
    engine = PipelineEngine(config, "security_dataset", roles, get_pack("security"))
    output = tmp_path / "out.jsonl"
    stats = run_pipeline(engine, [precutoff_case], output, None)
    assert stats == {"accepted": 0, "rejected": 1}


def test_build_record_shape(postcutoff_case):
    record = build_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze this",
        canonical={"verdict": "confirmed"},
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": True},
            "recitation_risk": False,
        },
    )
    assert isinstance(record, DatasetRecord)
    assert record.messages[1].content.startswith("{")
    errors = validate_record(record.model_dump())
    assert errors == []


def test_validate_record_rejects_bad_json(postcutoff_case):
    record = build_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze",
        canonical={"verdict": "confirmed"},
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": True},
            "recitation_risk": False,
        },
    )
    record.messages[-1].content = "not json"
    errors = validate_record(record.model_dump())
    assert any("JSON" in error for error in errors)


def test_validate_record_rejects_missing_metadata(postcutoff_case):
    record = build_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze",
        canonical={"verdict": "confirmed"},
        metadata={},
    )
    errors = validate_record(record.model_dump())
    assert any("metadata" in error for error in errors)


def test_validate_dataset(postcutoff_case):
    good = build_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze",
        canonical={"verdict": "confirmed"},
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": True},
            "recitation_risk": False,
        },
    )
    result = validate_dataset([good.model_dump()])
    assert result["invalid"] == []
    assert len(result["valid"]) == 1


def test_write_jsonl_roundtrip(postcutoff_case, tmp_path):
    record = build_record(
        case=postcutoff_case,
        domain="security",
        student_input="Analyze",
        canonical={"verdict": "confirmed"},
        metadata={
            "source": {},
            "teachers": [],
            "quality": {"passed": True},
            "recitation_risk": False,
        },
    )
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [record])
    loaded = read_jsonl(path)
    assert loaded[0]["id"] == "case-post"


def test_case_builder_from_dict():
    case = build_case_from_dict(
        {
            "source": {"type": "git", "repository": "r", "commit": "c", "date": "2026-08-01"},
            "input": {"code": "print(1)", "description": "d"},
            "metadata": {"language": "python"},
        }
    )
    assert case.id.startswith("case-")
    assert case.source.date == "2026-08-01"


def test_case_id_rejects_path_traversal():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Case(
            id="../../etc/passwd",
            source=CaseSource(type="manual"),
            input=CaseInput(code="x"),
        )
    with pytest.raises(ValidationError):
        build_case_from_dict(
            {
                "id": "../../../tmp/pwn",
                "source": {"type": "manual"},
                "input": {"code": "x"},
            }
        )


def test_case_id_deterministic():
    payload = {"source": {"type": "manual"}, "input": {"code": "x"}}
    assert build_case_from_dict(payload).id == build_case_from_dict(payload).id
    assert case_id_for("a") != case_id_for("b")


def test_group_key_prefers_repository():
    record = DatasetRecord(
        id="c",
        domain="security",
        messages=[
            ChatMessage(role="user", content="u"),
            ChatMessage(role="assistant", content="a"),
        ],
        metadata={"source": {"type": "git", "repository": "repo-x", "path": "p"}},
    )
    assert group_key(record) == "repo-x"


def test_split_is_source_grouped():
    def make_record(rid: str, repo: str) -> DatasetRecord:
        return DatasetRecord(
            id=rid,
            domain="security",
            messages=[
                ChatMessage(role="user", content=rid),
                ChatMessage(role="assistant", content="a"),
            ],
            metadata={"source": {"type": "git", "repository": repo}},
        )

    records = [make_record(f"c{i}", f"repo-{i % 10}") for i in range(100)]
    splits = split_dataset(records)
    repo_splits: dict[str, set[str]] = {}
    for name, split_records in splits.items():
        for record in split_records:
            repo = record.metadata["source"]["repository"]
            repo_splits.setdefault(repo, set()).add(name)
    leaked = {repo: names for repo, names in repo_splits.items() if len(names) > 1}
    assert not leaked, f"repository leaked across splits: {leaked}"
    assert len(splits["train"]) >= 60
    assert splits["train"] or splits["test"]


def test_postcutoff_filtering(postcutoff_case, precutoff_case):
    def record_for(case: Case) -> DatasetRecord:
        return DatasetRecord(
            id=case.id,
            domain="security",
            messages=[
                ChatMessage(role="user", content="u"),
                ChatMessage(role="assistant", content="a"),
            ],
            metadata={
                "recitation_risk": case.source.date == "2026-01-10",
                "source": case.source.model_dump(),
            },
        )

    records = [record_for(postcutoff_case), record_for(precutoff_case)]
    splits = split_with_postcutoff(records)
    assert [r.id for r in splits["test_postcutoff"]] == ["case-post"]
    warning = postcutoff_warning(splits, min_postcutoff=20)
    assert warning is not None and "post-cutoff" in warning


def test_postcutoff_records_excluded_from_regular_splits():
    def make_record(rid: str, repo: str, postcutoff: bool) -> DatasetRecord:
        return DatasetRecord(
            id=rid,
            domain="security",
            messages=[
                ChatMessage(role="user", content=rid),
                ChatMessage(role="assistant", content="a"),
            ],
            metadata={
                "recitation_risk": not postcutoff,
                "source": {"type": "git", "repository": repo},
            },
        )

    # One post-cutoff record per repo so every hash bucket is covered.
    records = [make_record(f"post-{i}", f"repo-post-{i}", True) for i in range(30)]
    records += [make_record(f"pre-{i}", f"repo-pre-{i}", False) for i in range(30)]
    splits = split_with_postcutoff(records)
    for name in ("train", "validation", "test"):
        leaked = [r.id for r in splits[name] if is_postcutoff(r)]
        assert not leaked, f"{name} contains post-cutoff records: {leaked}"
    assert len(splits["test_postcutoff"]) == 30
    total = sum(len(v) for v in splits.values())
    assert total == 60, "post-cutoff records must not be duplicated across splits"


def test_dedup_exact_and_near(tmp_path):
    def make_record(rid: str, user: str, assistant: str) -> DatasetRecord:
        return DatasetRecord(
            id=rid,
            domain="security",
            messages=[
                ChatMessage(role="user", content=user),
                ChatMessage(role="assistant", content=assistant),
            ],
            metadata={},
        )

    base = make_record("a", "analyze code " + "word " * 30, json.dumps({"verdict": "confirmed"}))
    exact_dup = make_record(
        "a", "analyze code " + "word " * 30, json.dumps({"verdict": "confirmed"})
    )
    different = make_record(
        "b",
        "completely other text about kubernetes networking policies " + "cluster " * 30,
        json.dumps({"verdict": "rejected"}),
    )
    kept, duplicates = deduplicate([base, exact_dup, different])
    assert 1 in duplicates
    assert duplicates[1] == 0
    assert [r.id for r in kept] == ["a", "b"]

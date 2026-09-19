import json

import pytest
from pydantic import ValidationError

from brainforge.config import export_schema, load_config, schema_matches
from brainforge.config.loader import expand_env, write_schema
from brainforge.config.models import Config
from brainforge.errors import ConfigError
from brainforge.types import ProviderType, RoleKind


def make_raw() -> dict:
    return {
        "providers": {
            "openrouter": {"type": "openrouter", "api_key": "${OPENROUTER_API_KEY:test-key}"},
            "zen": {"type": "zen", "api_key": "${ZEN_API_KEY}"},
            "mlgw": {"type": "mlgw", "api_key": "deadbeef"},
            "mock": {"type": "mock"},
        },
        "models": {
            "security": {
                "provider": "openrouter",
                "model": "deepseek-v4-pro",
                "family": "deepseek",
                "knowledge_cutoff": "2026-06",
            },
            "coding": {"provider": "zen", "model": "kimi-k2.7-code", "family": "kimi"},
            "general": {"provider": "mlgw", "model": "qwen3.5-9b", "family": "qwen"},
            "judge": {"provider": "mock", "model": "mock-judge", "family": "judge-mock"},
        },
        "roles": {
            "security_teacher": {"model": "security", "temperature": 0.2},
            "coding_teacher": {"model": "coding", "temperature": 0.3},
            "general_teacher": {"model": "general"},
            "judge": {"model": "judge", "temperature": 0.0},
        },
        "pipelines": {
            "security_dataset": {
                "domain": "security",
                "steps": ["security_teacher", "coding_teacher", "general_teacher", "judge"],
            }
        },
    }


@pytest.fixture
def raw_config():
    return make_raw()


@pytest.fixture
def config_file(tmp_path, raw_config, monkeypatch):
    monkeypatch.setenv("ZEN_API_KEY", "zen-key")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw_config))
    return path


def test_infer_role_kind(config_file):
    config = load_config(config_file)
    assert config.infer_role_kind("security_teacher") == RoleKind.TEACHER
    assert config.infer_role_kind("general_teacher") == RoleKind.CRITIC
    assert config.infer_role_kind("judge") == RoleKind.JUDGE


def test_env_expansion_default(monkeypatch):
    monkeypatch.setenv("MY_VAR", "hello")
    assert expand_env("${MY_VAR}") == "hello"
    assert expand_env("${MISSING_VAR:fallback}") == "fallback"
    assert expand_env("${MISSING_VAR:}") == ""
    assert expand_env("a/${MY_VAR}/b") == "a/hello/b"


def test_env_expansion_missing_raises(monkeypatch):
    monkeypatch.delenv("MY_VAR", raising=False)
    with pytest.raises(ConfigError, match="MY_VAR"):
        expand_env("${MY_VAR}")


def test_env_expansion_nested():
    data = {"a": ["${V:x}", {"b": "${V:y}"}], "n": 1, "b": True}
    assert expand_env(data) == {"a": ["x", {"b": "y"}], "n": 1, "b": True}


def test_load_config_ok(config_file):
    config = load_config(config_file)
    assert config.providers["openrouter"].api_key == "test-key"
    assert config.providers["mlgw"].type == ProviderType.MLGW
    assert config.models["security"].knowledge_cutoff == "2026-06"
    assert config.judge_independence == "enforced"


def test_missing_env_var_fails(config_file, monkeypatch):
    monkeypatch.delenv("ZEN_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ZEN_API_KEY"):
        load_config(config_file)


def test_invalid_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(path)


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("/nonexistent/config.json")


def test_unknown_provider_reference(config_file):
    raw = make_raw()
    raw["models"]["security"]["provider"] = "ghost"
    path = config_file.parent / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="unknown provider 'ghost'"):
        load_config(path)


def test_unknown_role_step(config_file):
    raw = make_raw()
    raw["pipelines"]["security_dataset"]["steps"] = ["ghost_role", "judge"]
    path = config_file.parent / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="unknown role 'ghost_role'"):
        load_config(path)


def test_extra_field_forbidden(config_file):
    raw = make_raw()
    raw["providers"]["openrouter"]["ghost_option"] = True
    path = config_file.parent / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        load_config(path)


def test_judge_independence_enforced_rejects_same_provider(config_file):
    raw = make_raw()
    raw["models"]["judge"]["provider"] = "openrouter"
    path = config_file.parent / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="judge independence"):
        load_config(path)


def test_judge_independence_rejects_same_family_different_provider(config_file):
    raw = make_raw()
    raw["models"]["judge"] = {"provider": "mock", "model": "m", "family": "kimi"}
    path = config_file.parent / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="family"):
        load_config(path)


def test_judge_independence_warn_mode(config_file):
    raw = make_raw()
    raw["models"]["judge"]["provider"] = "openrouter"
    raw["judge_independence"] = "warn"
    path = config_file.parent / "warn.json"
    path.write_text(json.dumps(raw))
    config = load_config(path)
    assert config.check_judge_independence()


def test_judge_independence_off_mode(config_file):
    raw = make_raw()
    raw["models"]["judge"]["provider"] = "openrouter"
    raw["judge_independence"] = "off"
    path = config_file.parent / "off.json"
    path.write_text(json.dumps(raw))
    config = load_config(path)
    assert config.check_judge_independence()


def test_no_judge_role_passes(config_file):
    raw = make_raw()
    del raw["roles"]["judge"]
    raw["pipelines"]["security_dataset"]["steps"] = [
        "security_teacher",
        "coding_teacher",
        "general_teacher",
    ]
    path = config_file.parent / "nojudge.json"
    path.write_text(json.dumps(raw))
    config = load_config(path)
    assert config.check_judge_independence() == []


def test_explicit_kind_overrides_inference(config_file):
    raw = make_raw()
    raw["roles"]["judge"]["kind"] = "teacher"
    path = config_file.parent / "kind.json"
    path.write_text(json.dumps(raw))
    config = load_config(path)
    assert config.infer_role_kind("judge") == RoleKind.TEACHER


def test_knowledge_cutoff_format(config_file):
    raw = make_raw()
    raw["models"]["security"]["knowledge_cutoff"] = "june-2026"
    with pytest.raises(ValidationError):
        Config.model_validate(make_raw() | {"models": raw["models"]})


def test_provider_base_url_rejects_metadata_and_link_local():
    from brainforge.config.models import ProviderConfig

    for bad in (
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/",
        "https://0.0.0.0/v1",
        "ftp://openrouter.ai/v1",
    ):
        with pytest.raises(ValidationError, match="base_url"):
            ProviderConfig(type="mock", base_url=bad)
    for good in (
        "https://openrouter.ai/api/v1",
        "http://localhost:8080/v1",
        "http://127.0.0.1:11434/v1",
    ):
        ProviderConfig(type="mock", base_url=good)


def test_default_config_mlgw_key_has_no_fake_default(monkeypatch):
    from brainforge.config.defaults import DEFAULT_CONFIG_PATH

    monkeypatch.delenv("MLGW_API_KEY", raising=False)
    expanded = expand_env(json.loads(DEFAULT_CONFIG_PATH.read_text()))
    api_key = expanded["providers"]["mlgw"]["api_key"]
    assert api_key != "deadbeef"


def test_schema_roundtrip(config_file, tmp_path):
    schema_path = tmp_path / "schema.json"
    write_schema(schema_path)
    assert schema_matches(schema_path)
    assert export_schema()["title"] == "Config"
    on_disk = json.loads(schema_path.read_text())
    on_disk["properties"]["ghost"] = {}
    schema_path.write_text(json.dumps(on_disk))
    assert not schema_matches(schema_path)


def test_training_checkpoint_fields():
    from brainforge.config.models import TrainingConfig

    config = TrainingConfig()
    assert config.save_steps == 100
    assert config.seed == 42
    with pytest.raises(ValidationError):
        TrainingConfig(save_steps=0)


def test_training_defaults(config_file):
    config = load_config(config_file)
    assert config.training.base_model == "Qwen/Qwen3-8B"
    assert config.training.quantization == "4bit"

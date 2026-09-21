import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from brainforge.config.defaults import CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH, SCHEMA_FILENAME
from brainforge.config.models import Config
from brainforge.errors import ConfigError

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(_substitute, value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def _substitute(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        return default
    raise ConfigError(f"environment variable '{name}' is not set and has no default")


def resolve_config_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    from_env = os.environ.get(CONFIG_ENV_VAR)
    if from_env:
        return Path(from_env)
    return DEFAULT_CONFIG_PATH


def load_config(path: Path | str | None = None) -> Config:
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    raw = expand_env(raw)
    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {config_path}:\n{exc}") from exc
    errors = config.validate_references()
    if errors:
        raise ConfigError(
            "invalid configuration references in "
            f"{config_path}:\n" + "\n".join(f"- {error}" for error in errors)
        )
    violations = config.check_judge_independence()
    if violations and config.judge_independence == "enforced":
        raise ConfigError(
            "judge independence violated (set judge_independence to 'warn' or 'off' to allow):\n"
            + "\n".join(f"- {violation}" for violation in violations)
        )
    return config


def export_schema() -> dict[str, Any]:
    return Config.model_json_schema()


def schema_matches(schema_path: Path | str) -> bool:
    on_disk = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return bool(on_disk == export_schema())


def write_schema(schema_path: Path | str) -> Path:
    target = Path(schema_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(export_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return target


def default_schema_path(config_path: Path | str | None = None) -> Path:
    return resolve_config_path(config_path).parent / SCHEMA_FILENAME

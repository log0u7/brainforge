"""CLI providers check-logprobs: probe whether a provider serves the KD
teacher signal (per-token top-k logprobs). Validated ad-hoc on gpu3080
before landing in the framework (openrouter glm-5.3-prime serves, some
models silently return empty positions).
"""

from typer.testing import CliRunner

from brainforge.cli import app
from brainforge.config.models import ProviderConfig
from brainforge.errors import ProviderError
from brainforge.providers.base import (
    ChatRequest,
    ChatResponse,
    PositionLogprobs,
    Provider,
)

runner = CliRunner()


class _ServingProvider(Provider):
    def __init__(self):
        super().__init__("fake", ProviderConfig(type="mock"))

    def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        raise NotImplementedError

    def complete_logprobs(
        self, request: ChatRequest, model: str, top_logprobs: int = 8
    ) -> ChatResponse:
        return ChatResponse(
            provider=self.name,
            model=model,
            latency_ms=123.4,
            logprobs=[
                PositionLogprobs(token="OK", logprob=-0.05, topk=[("OK", -0.05), ("Ok", -3.1)]),
                PositionLogprobs(token=".", logprob=-0.01, topk=[(".", -0.01)]),
            ],
        )


class _EmptyProvider(_ServingProvider):
    """Serves a 200 answer but silently no positions (glm-5.3-flash case)."""

    def complete_logprobs(
        self, request: ChatRequest, model: str, top_logprobs: int = 8
    ) -> ChatResponse:
        return ChatResponse(provider=self.name, model=model, latency_ms=5.0, logprobs=[])


class _RefusingProvider(_ServingProvider):
    def complete_logprobs(
        self, request: ChatRequest, model: str, top_logprobs: int = 8
    ) -> ChatResponse:
        raise ProviderError("provider 'fake' does not support logprobs")


def _patch_build(monkeypatch, provider):
    import brainforge.cli.providers_cmd as mod

    monkeypatch.setattr(mod, "build_provider", lambda name, config: provider)


def _invoke(*args: str):
    return runner.invoke(app, ["providers", "check-logprobs", *args])


def test_check_logprobs_served(project_env, monkeypatch):
    _patch_build(monkeypatch, _ServingProvider())
    result = _invoke("--provider", "mlgw", "--model", "qwen3.5-9b")
    assert result.exit_code == 0, result.output
    assert "logprobs served: yes" in result.output
    assert "positions: 2" in result.output
    assert "123.4" in result.output  # latency_ms
    assert "OK" in result.output  # sample topk entry


def test_check_logprobs_silently_absent(project_env, monkeypatch):
    _patch_build(monkeypatch, _EmptyProvider())
    result = _invoke("--provider", "mlgw", "--model", "qwen3.5-9b")
    assert result.exit_code == 0, result.output
    assert "logprobs served: no" in result.output


def test_check_logprobs_provider_error_is_clean_red_exit(project_env, monkeypatch):
    _patch_build(monkeypatch, _RefusingProvider())
    result = _invoke("--provider", "mlgw", "--model", "qwen3.5-9b")
    assert result.exit_code == 1
    assert "check-logprobs failed:" in result.stderr
    assert "does not support logprobs" in result.stderr


def test_check_logprobs_unknown_provider(project_env):
    result = _invoke("--provider", "ghost", "--model", "qwen3.5-9b")
    assert result.exit_code == 1
    assert "check-logprobs failed:" in result.stderr
    assert "not configured" in result.stderr


def test_check_logprobs_requires_provider_and_model(project_env):
    result = _invoke("--provider", "mlgw")
    assert result.exit_code == 1
    assert "--all-configured" in result.stderr


def test_check_logprobs_all_configured_table(project_env):
    # project_env wires every model to the mock provider; base
    # complete_logprobs raises, so every row must be a clean "no".
    result = _invoke("--all-configured")
    assert result.exit_code == 0, result.output
    assert "mock_only" in result.output
    assert "qwen3.5-9b" in result.output
    assert "no" in result.output


def test_check_logprobs_all_configured_with_serving_provider(project_env, monkeypatch):
    _patch_build(monkeypatch, _ServingProvider())
    result = _invoke("--all-configured")
    assert result.exit_code == 0, result.output
    assert "yes" in result.output

"""Tests for the served-model task-eval generator (OpenAI-compatible endpoint)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from brainforge.errors import ProviderError
from brainforge.providers.base import ChatResponse, Usage
from brainforge.training.task_eval import build_remote_generator


class _FakeProvider:
    """Canned ChatResponse provider keyed by the user prompt."""

    def __init__(self, responses: dict[str, str], fail_on: set[str] | None = None):
        self.responses = responses
        self.fail_on = fail_on or set()
        self.requests: list[dict] = []
        self.chat_requests: list[Any] = []

    def complete(self, request: Any, model: str) -> ChatResponse:
        prompt = request.messages[-1].content
        self.chat_requests.append(request)
        self.requests.append({"prompt": prompt, "model": model, "max_tokens": request.max_tokens})
        if prompt in self.fail_on:
            raise ProviderError(f"server error on {prompt!r}")
        content = self.responses.get(prompt)
        if content is None:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return ChatResponse(
            content=content,
            provider="fake",
            model=model,
            usage=Usage(input_tokens=10, output_tokens=20),
        )


def _install(monkeypatch: Any, provider: _FakeProvider) -> None:
    import brainforge.training.task_eval as te

    monkeypatch.setattr(te, "_build_provider", lambda base_url, api_key: provider)


def test_remote_generator_maps_prompts_in_order(monkeypatch: Any):
    provider = _FakeProvider(
        {
            "p1": json.dumps({"verdict": "solved", "flag": "FLAG{a}"}),
            "p2": json.dumps({"verdict": "insufficient_information"}),
        }
    )
    _install(monkeypatch, provider)
    generate = build_remote_generator("http://localhost:9933/v1", "mymodel")
    texts = generate([["p1"], ["p2"]])
    assert "FLAG{a}" in texts[0]
    assert "insufficient_information" in texts[1]
    assert [r["prompt"] for r in provider.requests] == ["p1", "p2"]
    assert all(r["model"] == "mymodel" for r in provider.requests)


def test_remote_generator_passes_max_tokens(monkeypatch: Any):
    provider = _FakeProvider({"p": json.dumps({"verdict": "solved"})})
    _install(monkeypatch, provider)
    generate = build_remote_generator("http://localhost:9933/v1", "mymodel", max_tokens=777)
    generate([["p"]])
    assert provider.requests[0]["max_tokens"] == 777


def test_remote_generator_surfaces_provider_error(monkeypatch: Any):
    provider = _FakeProvider({"boom": json.dumps({"verdict": "solved"})}, fail_on={"boom"})
    _install(monkeypatch, provider)
    generate = build_remote_generator("http://localhost:9933/v1", "mymodel")
    with pytest.raises(ProviderError):
        generate([["boom"]])


def test_remote_generator_uses_flat_user_message(monkeypatch: Any):
    provider = _FakeProvider({"line1\n\nline2": json.dumps({"verdict": "solved"})})
    _install(monkeypatch, provider)
    generate = build_remote_generator("http://localhost:9933/v1", "mymodel")
    generate([["line1", "line2"]])
    assert provider.requests[0]["prompt"] == "line1\n\nline2"


def test_evaluate_on_records_with_remote_generator(tmp_path: Any, monkeypatch: Any):
    """run_task_eval through a remote generator writes no local task_eval.json."""
    from pathlib import Path

    import brainforge.training.task_eval as te

    records = [
        {
            "id": "r1",
            "domain": "ctf",
            "messages": [
                {"role": "user", "content": "case prompt"},
                {
                    "role": "assistant",
                    "content": json.dumps({"verdict": "solved", "flag": "FLAG{a}"}),
                },
            ],
        }
    ]
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    provider = _FakeProvider({"case prompt": json.dumps({"verdict": "solved", "flag": "FLAG{a}"})})
    _install(monkeypatch, provider)
    result = te.evaluate_model_on_records(
        Path("models/export"),
        dataset,
        generator=te.build_remote_generator("http://localhost:9933/v1", "mymodel"),
    )
    assert result["n_records"] == 1
    assert result["accuracy"] == 1.0
    assert not (Path("models/export") / "task_eval.json").exists()


def test_cli_endpoint_requires_model_name(tmp_path: Any, monkeypatch: Any):
    from typer.testing import CliRunner

    from brainforge.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "task-eval",
            "--endpoint",
            "http://localhost:9933/v1",
            "-d",
            str(tmp_path / "ds.jsonl"),
        ],
    )
    assert result.exit_code == 1
    assert "--model-name is required" in result.output


def test_remote_generator_wires_enable_thinking(monkeypatch: Any):
    provider = _FakeProvider({"p": json.dumps({"verdict": "solved"})})
    _install(monkeypatch, provider)
    generate = build_remote_generator("http://localhost:9933/v1", "mymodel", enable_thinking=False)
    generate([["p"]])
    request = provider.chat_requests[-1]
    assert request.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_remote_generator_default_sends_no_extra_body(monkeypatch: Any):
    provider = _FakeProvider({"p": json.dumps({"verdict": "solved"})})
    _install(monkeypatch, provider)
    generate = build_remote_generator("http://localhost:9933/v1", "mymodel")
    generate([["p"]])
    request = provider.chat_requests[-1]
    assert not request.extra_body


def test_generation_template_kwargs():
    from brainforge.training.task_eval import _generation_template_kwargs

    assert _generation_template_kwargs(False) == {"enable_thinking": False}
    assert _generation_template_kwargs(True) == {"enable_thinking": True}
    assert _generation_template_kwargs(None) == {}


def test_cli_default_thinking_is_none_not_false(tmp_path: Any, monkeypatch: Any):
    """The default task-eval thinking mode is None (render like training):
    the --no-thinking default of v0.8 inserted an empty think block into the
    prompt that TRL prompt-completion training never showed, so every
    generation was measured against a template mismatch."""
    import json as _json
    from pathlib import Path as _Path

    from typer.testing import CliRunner

    import brainforge.cli.train_cmd as tc
    import brainforge.training.task_eval as te_module
    from brainforge.cli import app

    captured: dict = {}

    def fake_eval(
        model_path, dataset, quantization, max_new_tokens, batch_size=None, enable_thinking=None
    ):
        captured["enable_thinking"] = enable_thinking
        return {"n_records": 1, "accuracy": 1.0}

    monkeypatch.setattr(te_module, "evaluate_model_on_records", fake_eval)
    monkeypatch.setattr(tc, "latest_run_dir", lambda: _Path(tmp_path))
    monkeypatch.setattr(
        "brainforge.config.load_config",
        lambda *a, **k: type("C", (), {"training": type("T", (), {"generate_batch_size": 4})()})(),
    )
    dataset = tmp_path / "ds.jsonl"

    def fake_record() -> dict:
        return {
            "id": "rec-001",
            "domain": "security",
            "messages": [
                {"role": "user", "content": "case prompt"},
                {
                    "role": "assistant",
                    "content": _json.dumps({"verdict": "confirmed", "cwe": "CWE-78"}),
                },
            ],
        }

    dataset.write_text(_json.dumps(fake_record()) + "\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["train", "task-eval", "-d", str(dataset)])
    assert result.exit_code == 0, result.output
    assert captured["enable_thinking"] is None

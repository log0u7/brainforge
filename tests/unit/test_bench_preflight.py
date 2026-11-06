"""Bench preflight: endpoint protocol checks for the final benchmark."""

import json

from brainforge.experiments import preflight


def _json_response(payload: dict, code: int = 200) -> tuple[int, bytes]:
    return code, json.dumps(payload).encode()


def test_constrained_output_check_passes_when_the_endpoint_honours_the_schema(monkeypatch) -> None:
    def fake_http_json(url, payload=None, timeout=10.0):
        assert payload["response_format"]["type"] == "json_schema"
        # a thinking model streams the answer into reasoning_content and
        # leaves content empty: the probe must disable thinking too
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return _json_response({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(preflight, "_http_json", fake_http_json)
    check = preflight.check_constrained_output("http://host:9999/v1", "model")
    assert check["status"] == "pass"


def test_constrained_output_check_fails_when_the_endpoint_ignores_the_schema(monkeypatch) -> None:
    def fake_http_json(url, payload=None, timeout=10.0):
        return _json_response({"choices": [{"message": {"content": "not json"}}]})

    monkeypatch.setattr(preflight, "_http_json", fake_http_json)
    check = preflight.check_constrained_output("http://host:9999/v1", "model")
    assert check["status"] == "fail"


def test_constrained_output_check_fails_on_an_http_error(monkeypatch) -> None:
    def fake_http_json(url, payload=None, timeout=10.0):
        return 400, b'{"error":"response_format unsupported"}'

    monkeypatch.setattr(preflight, "_http_json", fake_http_json)
    check = preflight.check_constrained_output("http://host:9999/v1", "model")
    assert check["status"] == "fail"


def test_run_preflight_includes_the_constrained_output_check_for_both_endpoints(
    monkeypatch,
) -> None:
    seen: list[str] = []

    def fake_http_json(url, payload=None, timeout=10.0):
        seen.append(url)
        return _json_response({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(preflight, "_http_json", fake_http_json)
    monkeypatch.setattr(preflight, "check_alias", lambda *a, **k: {"id": "alias", "status": "pass"})
    monkeypatch.setattr(
        preflight, "check_chat_template", lambda *a, **k: {"id": "template", "status": "pass"}
    )
    monkeypatch.setattr(
        preflight, "check_latency", lambda *a, **k: {"id": "latency", "status": "pass"}
    )

    result = preflight.run_preflight(
        "http://base:1/v1", "http://trained:2/v1", "base-model", "trained-model"
    )

    ids = [check["id"] for check in result.checks]
    assert "constrained_output.base-model" in ids
    assert "constrained_output.trained-model" in ids
    assert result.ok

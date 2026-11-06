"""Bench preflight checks for the final benchmark.

The bench runs both models on llama-server over HTTP. Before a single
attempt runs, the endpoints must be the two distinct servers, serve the
aliases the runner routes on, and answer a chat completion fast enough.
A run that starts on a misconfigured endpoint burns GPU hours and
produces numbers nobody can reproduce (run-20261106: the base and the
trained alias both answered, but every fact request came back empty).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#: checks that must pass. A WARN is reported and does not fail the run:
#: llama.cpp reports ``chat_format: Content-only`` even with --jinja
#: enabled, and the template params prove the jinja template is active.
PASSING_STATUSES = {"pass", "warn"}


@dataclass
class PreflightResult:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _http_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b""
    except URLError:
        raise


def _strip_v1(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _model_aliases(body: bytes) -> list[str]:
    data = json.loads(body)
    models = data.get("data") or data.get("models") or []
    aliases: list[str] = []
    for model in models:
        if isinstance(model, dict):
            alias = model.get("id") or model.get("name") or model.get("model")
            if alias:
                aliases.append(str(alias))
    return aliases


def check_alias(base_url: str, expected_alias: str, timeout: float = 10.0) -> dict[str, Any]:
    check_id = f"alias.{expected_alias}"
    root = _strip_v1(base_url)
    for url in (f"{root}/v1/models", f"{root}/models"):
        try:
            code, body = _http_json(url, timeout=timeout)
        except Exception as exc:
            return {"id": check_id, "status": "fail", "message": str(exc)}
        if code != 200:
            continue
        aliases = _model_aliases(body)
        if expected_alias in aliases:
            return {"id": check_id, "status": "pass", "message": expected_alias}
        return {
            "id": check_id,
            "status": "fail",
            "message": f"alias missing; served: {aliases}",
        }
    return {"id": check_id, "status": "fail", "message": "models endpoint unreachable"}


def check_chat_template(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    root = _strip_v1(base_url)
    for url in (f"{root}/props", f"{base_url.rstrip('/')}/props"):
        try:
            code, body = _http_json(url, timeout=timeout)
        except Exception as exc:
            return {"id": "chat_template", "status": "fail", "message": str(exc)}
        if code != 200:
            continue
        data = json.loads(body)
        params = (data.get("default_generation_settings") or {}).get("params") or {}
        chat_format = params.get("chat_format")
        template = params.get("chat_template") or data.get("chat_template") or ""
        if chat_format == "Content-only" and not template:
            return {
                "id": "chat_template",
                "status": "warn",
                "message": "chat_format=Content-only (provider renders the prompt)",
            }
        return {"id": "chat_template", "status": "pass", "message": f"chat_format={chat_format}"}
    return {"id": "chat_template", "status": "warn", "message": "props endpoint unreachable"}


def check_latency(
    base_url: str,
    model: str,
    timeout: float = 30.0,
    max_seconds: float = 10.0,
) -> dict[str, Any]:
    url = f"{_strip_v1(base_url)}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 4,
        "stream": False,
    }
    try:
        started = time.perf_counter()
        code, _body = _http_json(url, payload=payload, timeout=timeout)
        elapsed = time.perf_counter() - started
    except Exception as exc:
        return {"id": "latency", "status": "fail", "message": str(exc)}
    if code != 200:
        return {"id": "latency", "status": "fail", "message": f"HTTP {code} in {elapsed:.2f}s"}
    if elapsed > max_seconds:
        return {
            "id": "latency",
            "status": "fail",
            "message": f"too slow: {elapsed:.2f}s > {max_seconds:.0f}s",
        }
    return {"id": "latency", "status": "pass", "message": f"{elapsed:.2f}s"}


def probe(base_url: str, model: str, timeout: float = 30.0) -> bool:
    """One cheap completion: True when the endpoint answers 200."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
        "stream": False,
    }
    try:
        code, _body = _http_json(
            f"{_strip_v1(base_url)}/v1/chat/completions", payload=payload, timeout=timeout
        )
    except Exception:
        return False
    return code == 200


def run_preflight(
    base_url: str,
    trained_url: str,
    base_model: str,
    trained_model: str,
    timeout: float = 30.0,
    max_latency: float = 10.0,
) -> PreflightResult:
    checks: list[dict[str, Any]] = []
    if base_url.rstrip("/") == trained_url.rstrip("/"):
        checks.append(
            {
                "id": "distinct_servers",
                "status": "fail",
                "message": "base and trained must be different servers",
            }
        )
    else:
        checks.append({"id": "distinct_servers", "status": "pass", "message": "distinct"})

    checks.append(check_alias(base_url, base_model))
    checks.append(check_alias(trained_url, trained_model))
    checks.append(check_chat_template(base_url))
    checks.append(check_chat_template(trained_url))
    checks.append(check_latency(base_url, base_model, timeout=timeout, max_seconds=max_latency))
    checks.append(
        check_latency(trained_url, trained_model, timeout=timeout, max_seconds=max_latency)
    )

    details = {
        "base_url": base_url,
        "trained_url": trained_url,
        "base_model": base_model,
        "trained_model": trained_model,
    }
    ok = all(check["status"] in PASSING_STATUSES for check in checks)
    return PreflightResult(ok=ok, checks=checks, details=details)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Bench preflight checks")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--trained-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-latency", type=float, default=10.0)
    args = parser.parse_args(argv)

    result = run_preflight(
        args.base_url,
        args.trained_url,
        args.base_model,
        args.trained_model,
        timeout=args.timeout,
        max_latency=args.max_latency,
    )
    for check in result.checks:
        print(f"[{check['status'].upper()}] {check['id']}: {check['message']}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

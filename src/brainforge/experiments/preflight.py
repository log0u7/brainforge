"""Bench preflight checks for the final benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class PreflightResult:
    ok: bool
    checks: list[dict[str, Any]]
    details: dict[str, Any]


def _http_get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b""
    except URLError:
        raise


def _http_post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b""
    except URLError:
        raise


def _norm_base(base_url: str) -> str:
    return base_url.rstrip('/')


def check_alias(base_url: str, expected_alias: str, timeout: float = 10.0) -> dict[str, Any]:
    b = _norm_base(base_url)
    # try different paths
    candidates = []
    if b.endswith('/v1'):
        candidates = [f"{b}/models", f"{b[:-3]}/v1/models"]
    else:
        candidates = [f"{b}/v1/models", f"{b}/models"]
    for url in candidates:
        try:
            code, body = _http_get(url, timeout=timeout)
            if code == 200:
                data = json.loads(body)
                models = data.get("data") or data.get("models") or []
                aliases = []
                for m in models:
                    if isinstance(m, dict):
                        aliases.append(m.get("id") or m.get("name") or m.get("model"))
                if expected_alias in aliases or any(expected_alias == str(a) or expected_alias in str(a) for a in aliases):
                    return {"id": f"alias.{expected_alias}", "status": "pass", "message": expected_alias}
                return {"id": f"alias.{expected_alias}", "status": "fail", "message": f"alias not found; got {aliases}"}
        except Exception:
            continue
    return {"id": f"alias.{expected_alias}", "status": "fail", "message": "could not reach models endpoint"}


def check_chat_template(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    b = _norm_base(base_url)
    root = b[:-3] if b.endswith('/v1') else b
    for url in [f"{root}/props", f"{b}/props"]:
        try:
            code, body = _http_get(url, timeout=timeout)
            if code == 200:
                data = json.loads(body)
                gen = data.get("default_generation_settings") or {}
                params = gen.get("params") or {}
                chat_format = params.get("chat_format")
                gen_prompt = params.get("generation_prompt") or gen.get("generation_prompt") or ""
                if chat_format == "Content-only":
                    return {"id": "chat_template", "status": "warn", "message": f"chat_format={chat_format} (will rely on provider)"}
                if chat_format is None and gen_prompt == "":
                    return {"id": "chat_template", "status": "warn", "message": "template info missing"}
                return {"id": "chat_template", "status": "pass", "message": f"chat_format={chat_format}"}
        except Exception:
            continue
    return {"id": "chat_template", "status": "warn", "message": "props not reachable"}


def check_latency(base_url: str, model: str, timeout: float = 30.0, max_seconds: float = 10.0) -> dict[str, Any]:
    b = _norm_base(base_url)
    for url in [f"{b}/v1/chat/completions" if not b.endswith('/v1') else f"{b}/chat/completions", f"{b}/chat/completions"]:
        try:
            start = time.perf_counter()
            code, _ = _http_post_json(url, {"model": model, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 4, "stream": False}, timeout=timeout)
            elapsed = time.perf_counter() - start
            if code in (200, 201):
                if elapsed > max_seconds:
                    return {"id": "latency", "status": "fail", "message": f"too slow: {elapsed:.2f}s > {max_seconds}s"}
                return {"id": "latency", "status": "pass", "message": f"{elapsed:.2f}s"}
        except Exception:
            continue
    return {"id": "latency", "status": "fail", "message": "could not reach chat completions"}


def probe(base_url: str, model: str, timeout: float = 30.0) -> bool:
    b = _norm_base(base_url)
    for url in [f"{b}/v1/chat/completions" if not b.endswith('/v1') else f"{b}/chat/completions", f"{b}/chat/completions"]:
        try:
            code, _ = _http_post_json(url, {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stream": False}, timeout=timeout)
            if code in (200, 201):
                return True
        except Exception:
            continue
    return False


def run_preflight(base_url: str, trained_url: str, base_model: str, trained_model: str, timeout: float = 30.0, max_latency: float = 10.0) -> PreflightResult:
    checks = []
    details = {"base_url": base_url, "trained_url": trained_url, "base_model": base_model, "trained_model": trained_model}
    if _norm_base(base_url) == _norm_base(trained_url):
        checks.append({"id": "distinct_servers", "status": "fail", "message": "base and trained must use different servers"})
    else:
        checks.append({"id": "distinct_servers", "status": "pass", "message": "distinct"})
    checks.append(check_alias(base_url, base_model))
    checks.append(check_alias(trained_url, trained_model))
    checks.append(check_chat_template(base_url))
    checks.append(check_chat_template(trained_url))
    checks.append(check_latency(base_url, base_model, timeout=timeout, max_seconds=max_latency))
    checks.append(check_latency(trained_url, trained_model, timeout=timeout, max_seconds=max_latency))
    ok = all(c["status"] in ("pass", "warn") for c in checks) and not any(c["status"] == "fail" for c in checks)
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
    result = run_preflight(args.base_url, args.trained_url, args.base_model, args.trained_model, timeout=args.timeout, max_latency=args.max_latency)
    for c in result.checks:
        print(f"[{c['status'].upper()}] {c['id']}: {c['message']}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

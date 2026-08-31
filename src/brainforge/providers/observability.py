import json
import time
from pathlib import Path

from brainforge.config.models import ModelDef
from brainforge.providers.base import ChatResponse


class UsageLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "usage.jsonl"

    def log(
        self,
        response: ChatResponse,
        role: str,
        cost_usd: float | None = None,
        error: str | None = None,
    ) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "provider": response.provider,
            "model": response.model,
            "role": role,
            "latency_ms": round(response.latency_ms, 2),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "retries": response.retries,
            "cache_hit": response.cached,
            "cost_usd": cost_usd,
            "error": error,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def estimate_cost(model: ModelDef, response: ChatResponse) -> float | None:
    if model.pricing is None:
        return None
    cost = (
        response.usage.input_tokens * model.pricing.input_per_mtok
        + response.usage.output_tokens * model.pricing.output_per_mtok
    ) / 1_000_000
    return round(cost, 6)

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from brainforge.config.models import Config


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "experiment"


def compute_metrics(records: list[dict]) -> dict:
    total = len(records)
    recitation_risk = sum(1 for r in records if r.get("metadata", {}).get("recitation_risk"))
    postcutoff = total - recitation_risk
    agreement_totals: dict[str, list[bool]] = {}
    for record in records:
        for role, agreed in record.get("metadata", {}).get("teacher_agreement", {}).items():
            agreement_totals.setdefault(role, []).append(bool(agreed))
    agreement_rates = {
        role: round(sum(values) / len(values), 3)
        for role, values in agreement_totals.items()
        if values
    }
    domains: dict[str, int] = {}
    for record in records:
        domain = record.get("domain", "unknown")
        domains[domain] = domains.get(domain, 0) + 1
    return {
        "total": total,
        "recitation_risk": recitation_risk,
        "recitation_risk_ratio": round(recitation_risk / total, 3) if total else 0.0,
        "postcutoff": postcutoff,
        "postcutoff_ratio": round(postcutoff / total, 3) if total else 0.0,
        "teacher_judge_agreement_rate": agreement_rates,
        "domains": domains,
    }


def high_agreement_warning(metrics: dict, threshold: float = 0.95) -> list[str]:
    warnings = []
    for role, rate in metrics.get("teacher_judge_agreement_rate", {}).items():
        if rate >= threshold:
            warnings.append(
                f"teacher '{role}' agrees with the judge on {rate:.0%} of records"
                " (>= 95%): check for correlated bias"
            )
    return warnings


def create_experiment(
    name: str,
    config: Config,
    dataset_path: Path | str | None = None,
    records: list[dict] | None = None,
    output_root: Path | str = Path("experiments"),
) -> Path:
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    run_id = f"{date}-{slugify(name)}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics = compute_metrics(records or [])
    if dataset_path is not None and Path(dataset_path).is_file():
        shutil.copy(Path(dataset_path), run_dir / "dataset.jsonl")
        if records is None:
            metrics = compute_metrics(_read_jsonl(Path(dataset_path)))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(_render_report(run_id, metrics), encoding="utf-8")
    return run_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _render_report(run_id: str, metrics: dict) -> str:
    lines = [
        f"# Experiment {run_id}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Dataset metrics",
        "",
        f"- total records: {metrics['total']}",
        f"- recitation risk: {metrics['recitation_risk']} ({metrics['recitation_risk_ratio']:.1%})",
        f"- post-cutoff (primary eval): {metrics['postcutoff']}"
        f" ({metrics['postcutoff_ratio']:.1%})",
        f"- domains: {metrics['domains']}",
        "",
        "## Teacher/judge agreement",
        "",
    ]
    if metrics["teacher_judge_agreement_rate"]:
        for role, rate in metrics["teacher_judge_agreement_rate"].items():
            lines.append(f"- {role}: {rate:.1%}")
    else:
        lines.append("- no agreement data")
    warnings = high_agreement_warning(metrics)
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {warning}" for warning in warnings]
    lines += [
        "",
        "## Contamination note",
        "",
        "Records flagged `recitation_risk` come from sources dated before the teachers'",
        "knowledge cutoffs and may reflect memorized CVEs/patches rather than analysis.",
        "Benchmark the student primarily on `test_postcutoff.jsonl`.",
        "",
    ]
    return "\n".join(lines)

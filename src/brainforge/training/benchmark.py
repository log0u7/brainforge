"""Three-arm benchmark orchestrator (MR4, ADR-0018/0020 evidence-first).

Arms on identical splits and holdout:

* ``no-kd``: plain QLoRA SFT, the baseline every distillation variant
  must beat (the first benchmark showed plain SFT ahead).
* ``sidecar``: the ADR-0015 top-k sidecar path (``train run --kd``),
  requires ``kd.jsonl`` in the dataset directory.
* ``gkd``: the ADR-0020 on-policy backend (``strategy: gkd``), requires
  ``training.gkd_teacher``.
* ``lora64``: opportunistic no-KD arm at LoRA rank 64, probing the
  student-capacity hypothesis against the noisy-signal hypothesis.

Each arm trains into its own directory (nothing shared), then evaluates
on the same frozen post-cutoff holdout. The orchestrator renders a
markdown comparison table for `docs/explanation/benchmark.md`.

Heavy imports are NOT needed at module level: the trainer and the gate
are injected callables, so the unit suite runs without the training
extra.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError

#: fixed arm order: the baseline first, then distillation variants
ARMS: dict[str, dict[str, Any]] = {
    "no-kd": {"updates": {"strategy": "qlora"}},
    "sidecar": {
        "updates": {"strategy": "qlora"},
        "train_kwargs": {"kd": True},
        "requires_sidecar": True,
    },
    "gkd": {"updates": {"strategy": "gkd"}, "requires_teacher": True},
    "lora64": {"updates": {"strategy": "qlora", "lora_rank": 64}},
}


def run_arm(
    config: TrainingConfig,
    arm: str,
    dataset_dir: Path,
    output_dir: Path,
    gate_dataset: Path,
    train_fn: Callable[..., dict[str, Any]] | None = None,
    gate_fn: Callable[[Path, Path], dict[str, Any]] | None = None,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Train one benchmark arm and evaluate it on the shared holdout.

    Fail fast before any heavy import: the sidecar arm refuses to run
    without ``kd.jsonl``, the gkd arm without ``gkd_teacher``.
    ``max_new_tokens`` rides the gate evaluation: 512 (the task_eval
    default) cut ground-truth completions short and scored them
    unparseable, so the benchmark defaults to the task-eval CLI value.
    """
    if arm not in ARMS:
        raise BrainforgeError(f"unknown benchmark arm: {arm} (known: {', '.join(ARMS)})")
    spec = ARMS[arm]
    dataset_dir = Path(dataset_dir)
    gate_dataset = Path(gate_dataset)

    if spec.get("requires_sidecar") and not (dataset_dir / "kd.jsonl").is_file():
        raise BrainforgeError(
            f"Sidecar arm requires kd.jsonl in {dataset_dir}. Run 'brainforge pipeline kd-pass'."
        )
    updates: dict[str, Any] = spec.get("updates", {})
    arm_config = config.model_copy(update=updates) if updates else config
    if spec.get("requires_teacher") and not getattr(arm_config, "gkd_teacher", ""):
        raise BrainforgeError(
            "no gkd_teacher: the gkd arm needs training.gkd_teacher (a local model"
            " sharing the student's tokenizer, e.g. Qwen/Qwen3-4B)"
        )

    out_dir = Path(output_dir) / arm
    trainer = train_fn
    if trainer is None:
        from brainforge.cli.train_cmd import train_student

        trainer = train_student
    train_kwargs: dict[str, Any] = spec.get("train_kwargs", {})
    summary = trainer(arm_config, dataset_dir, out_dir, **train_kwargs)

    gate = gate_fn
    if gate is None:
        from brainforge.training.task_eval import evaluate_model_on_records

        gate = evaluate_model_on_records
    metrics = gate(out_dir, gate_dataset, max_new_tokens=max_new_tokens)

    return {"arm": arm, "metrics": metrics, "output_dir": str(out_dir), "train_summary": summary}


def run_three_arm_benchmark(
    config: TrainingConfig,
    dataset_dir: Path,
    gate_dataset: Path,
    output_dir: Path,
    train_fn: Callable[..., dict[str, Any]] | None = None,
    gate_fn: Callable[[Path, Path], dict[str, Any]] | None = None,
    max_new_tokens: int = 2048,
) -> dict[str, dict[str, Any]]:
    """Run every arm on the same splits; returns results keyed by arm name.

    ``max_new_tokens`` rides every arm's gate evaluation (default 2048,
    matching the task-eval CLI): the 512 task_eval default truncates
    ground-truth completions and scores them unparseable.
    """
    results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        results[arm] = run_arm(
            config=config,
            arm=arm,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            gate_dataset=gate_dataset,
            train_fn=train_fn,
            gate_fn=gate_fn,
            max_new_tokens=max_new_tokens,
        )
    return results


def render_markdown_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the comparison table for docs/explanation/benchmark.md.

    n_records rides every row: the honesty rule of the first benchmark
    (report n with every number) applies here too.
    """
    header = "| arm | accuracy | flag_accuracy | n_records |"
    separator = "|---|---|---|---|"
    rows: list[str] = []
    for arm in ARMS:
        if arm not in results:
            continue
        metrics = results[arm].get("metrics", {})
        accuracy = metrics.get("accuracy")
        flag_accuracy = metrics.get("flag_accuracy")
        n_records = metrics.get("n_records", "-")
        acc = f"{accuracy:.4f}" if isinstance(accuracy, float) else "-"
        flag = f"{flag_accuracy:.4f}" if isinstance(flag_accuracy, float) else "-"
        rows.append(f"| {arm} | {acc} | {flag} | {n_records} |")
    return "\n".join([header, separator, *rows])

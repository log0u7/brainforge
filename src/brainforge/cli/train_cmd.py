import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer

from brainforge.cli._common import clamp_errors, console
from brainforge.errors import BrainforgeError

if TYPE_CHECKING:
    from brainforge.config.models import TrainingConfig
    from brainforge.training.preflight import CheckReport

app = typer.Typer(
    help="Training: prepare datasets, run QLoRA or full fine-tuning, evaluate, export.",
    no_args_is_help=True,
)


def resolve_run_output_dir(
    resume: bool,
    output: Path | str | None,
    training: "TrainingConfig",
    run_name: str,
) -> Path:
    """Pick the output dir for a training run.

    --resume reuses the latest run dir; on a fresh install (no run dirs
    yet) it falls back to a new run dir instead of crashing - service
    supervisors (nssm restart loops) must boot clean.
    """
    if resume and output is None:
        try:
            return latest_run_dir(base=Path(training.output_dir))
        except BrainforgeError:
            return Path(training.output_dir) / run_name
    return Path(output) if output else Path(training.output_dir) / run_name


def latest_run_dir(base: Path | None = None) -> Path:
    """Resolve the most recent training run directory, by modification time."""
    if base is None:
        from brainforge.config import load_config

        base = Path(load_config().training.output_dir)
    base = Path(base).resolve()
    candidates = [d for d in base.glob("run-*") if d.is_dir()]
    if not candidates:
        raise BrainforgeError(
            f"no training runs found under {base}; run 'brainforge train run' first"
        )
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _print_check_report(report: "CheckReport") -> None:
    if report.model is not None:
        console.print(f"model: {report.model.id} ({report.model.source})")
    if report.hardware is not None:
        console.print(f"hardware: {report.hardware.kind} ({report.hardware.name})")
    if report.plan is not None:
        plan = report.plan
        console.print(
            f"plan: backend={plan.backend} device={plan.device} dtype={plan.dtype}"
            f" quantization={plan.quantization} strategy={plan.effective_strategy}"
        )
    if report.estimate is None:
        console.print("memory: estimate unavailable")
    else:
        estimate = report.estimate
        available = (
            f"{estimate.available_gb:.1f} GB" if estimate.available_gb is not None else "unknown"
        )
        console.print(
            f"memory: ~{estimate.total_gb:.1f} GB total, {available} available"
            " (estimate, not a guarantee)"
        )
    for check in report.checks:
        console.print(f"  {check.status:<4} {check.id}: {check.message}")
    counts = {
        status: sum(1 for check in report.checks if check.status == status)
        for status in ("pass", "warn", "fail", "skip")
    }
    console.print(
        f"{counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail,"
        f" {counts['skip']} skip"
    )


def train_student(
    training: "TrainingConfig",
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
    kd: bool = False,
) -> dict:
    """Dispatch to the configured training backend (kd, unsloth or trl)."""
    from brainforge.training.prepare import tool_turn_warning

    # KD consumes the dataset differently (it rejects tool messages outright
    # and ignores assistant_only_loss), so the flag advice would be inert.
    warning = (
        None
        if kd
        else tool_turn_warning(Path(dataset_dir) / "train.jsonl", training.assistant_only_loss)
    )
    if warning:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    from brainforge.training.preflight import auto_fit_config

    training, fit_warnings = auto_fit_config(training, dataset_dir, kd=kd)
    for line in fit_warnings:
        console.print(f"[yellow]auto-fit:[/yellow] {line}")
    if kd:
        from brainforge.training.kd import train_kd as train_fn
    elif training.backend == "unsloth":
        from brainforge.training.unsloth_backend import train_unsloth as train_fn
    else:
        from brainforge.training.qlora import train_qlora as train_fn
    return train_fn(training, dataset_dir, output_dir, resume=resume)


@app.command("prepare")
@clamp_errors("prepare")
def prepare(
    dataset: Path = typer.Argument(..., help="Path to dataset.jsonl"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Validate, split and export a dataset for TRL training."""
    from brainforge.training.prepare import prepare

    target = output_dir or Path("datasets") / "prepared"
    stats = prepare(dataset, target, train_ratio, val_ratio, seed)
    console.print(f"[green]prepared dataset[/green] -> {target}")
    for name, count in stats.items():
        console.print(f"  {name}: {count}")


@app.command("check")
@clamp_errors("training check")
def check(
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Hugging Face id or local model directory (default: config base_model)",
    ),
    strategy: Literal["qlora", "full"] | None = typer.Option(
        None, "--strategy", help="Override config training strategy"
    ),
    quantization: Literal["auto", "4bit", "8bit", "none"] | None = typer.Option(
        None, "--quantization", help="Override config quantization"
    ),
    backend: Literal["auto", "trl", "unsloth"] | None = typer.Option(
        None, "--backend", help="Override config training backend"
    ),
    device: Literal["auto", "cuda", "xpu", "mps", "cpu"] | None = typer.Option(
        None, "--device", help="Override config device"
    ),
    max_length: int = typer.Option(0, "--max-length", help="Override config max length"),
    batch_size: int = typer.Option(0, "--batch-size", help="Override config batch size"),
    lora_rank: int = typer.Option(0, "--lora-rank", help="Override config LoRA rank"),
    dataset_dir: Path = typer.Option(
        None, "--dataset-dir", help="Optional prepared split to validate"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON report"),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as failures"),
    config: str = typer.Option(None, "--config", "-c"),
) -> None:
    """Check hardware, model compatibility and estimated memory before training."""
    from brainforge.config import load_config
    from brainforge.training.preflight import report_to_dict, run_preflight

    training = load_config(config).training
    report = run_preflight(
        training,
        model=model or None,
        strategy=strategy,
        quantization=quantization,
        backend=backend,
        device=device,
        max_length=max_length or None,
        batch_size=batch_size or None,
        lora_rank=lora_rank or None,
        dataset_dir=dataset_dir,
        strict=strict,
    )
    if json_output:
        typer.echo(json.dumps(report_to_dict(report), indent=2))
    else:
        _print_check_report(report)
    raise typer.Exit(code=0 if report.ok else 1)


@app.command("run")
@clamp_errors("training")
def run(
    dataset_dir: Path = typer.Option(
        Path("datasets/prepared"),
        "--dataset-dir",
        "-d",
        help="Directory with train/validation splits",
    ),
    epochs: int = typer.Option(
        0, "--epochs", "-e", help="Override config epochs (0 = keep config)"
    ),
    base_model: str = typer.Option("", "--base-model", "-b", help="Override config base model"),
    strategy: Literal["qlora", "full"] | None = typer.Option(
        None, "--strategy", help="Override config training strategy (qlora or full)"
    ),
    backend: Literal["auto", "trl", "unsloth"] | None = typer.Option(
        None, "--backend", help="Override config training backend"
    ),
    device: Literal["auto", "cuda", "xpu", "mps", "cpu"] | None = typer.Option(
        None, "--device", help="Override config device"
    ),
    quantization: Literal["auto", "4bit", "8bit", "none"] | None = typer.Option(
        None, "--quantization", help="Override config quantization"
    ),
    kd: bool = typer.Option(
        False,
        "--kd",
        help=(
            "Distillation training: SFT + KL to teacher top-k logprobs"
            " (requires a kd.jsonl sidecar in the dataset dir)"
        ),
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output dir (default: <config.output_dir>/<run name>)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Resume from the latest checkpoint in the output dir",
    ),
    config: str = typer.Option(None, "--config", "-c"),
) -> None:
    """Train the student: QLoRA, full fine-tuning or KD distillation.

    Requires the training extra; the device and quantization resolve automatically.
    """
    from brainforge.config import load_config

    cfg = load_config(config)
    training = cfg.training
    updates: dict[str, Any] = {}
    if epochs:
        updates["epochs"] = epochs
    if base_model:
        updates["base_model"] = base_model
    if strategy:
        updates["strategy"] = strategy
    if backend:
        updates["backend"] = backend
    if device:
        updates["device"] = device
    if quantization:
        updates["quantization"] = quantization
    if updates:
        training = training.model_copy(update=updates)
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
    output_dir = resolve_run_output_dir(resume, output, training, run_name)
    summary = train_student(training, dataset_dir, output_dir, resume=resume, kd=kd)
    console.print(f"[green]training done[/green] -> {summary['output_dir']}")
    for key, value in summary.items():
        console.print(f"  {key}: {value}")


@app.command("evaluate")
@clamp_errors("evaluation")
def evaluate(
    model: Path = typer.Option(
        None, "--model", "-m", help="Trained adapter directory (default: latest run)"
    ),
    dataset: Path = typer.Option(
        Path("datasets/prepared/test_postcutoff.jsonl"),
        "--dataset",
        "-d",
        help="Eval split (JSONL)",
    ),
    config: str = typer.Option(None, "--config", "-c"),
    batch_size: int = typer.Option(
        0, "--batch-size", help="Eval forward batch size (0 = config default)"
    ),
) -> None:
    """Evaluate a trained student (loss + perplexity, written to eval.json)."""
    from brainforge.config import load_config
    from brainforge.training.qlora import evaluate as evaluate_model

    training = load_config(config).training
    result = evaluate_model(
        model or latest_run_dir(),
        dataset,
        batch_size=batch_size or training.eval_batch_size,
        attn_implementation=training.attn_implementation,
    )
    console.print(f"[green]evaluated[/green] {result['n_records']} records")
    console.print(f"  eval_loss: {result['eval_loss']:.4f}")
    console.print(f"  perplexity: {result['perplexity']:.4f}")


@app.command("chat")
@clamp_errors("chat")
def chat(
    model: Path = typer.Option(
        None, "--model", "-m", help="Trained adapter or merged model (default: latest run)"
    ),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
) -> None:
    """Interactive chat with a trained student model (requires the training extra)."""
    from brainforge.training.chat import chat_loop

    chat_loop(model or latest_run_dir(), max_new_tokens=max_new_tokens)


@app.command("task-eval")
@clamp_errors("task evaluation")
def task_eval(
    model: Path = typer.Option(
        None, "--model", "-m", help="Trained adapter or merged model (default: latest run)"
    ),
    dataset: Path = typer.Option(
        Path("datasets/prepared/test_postcutoff.jsonl"),
        "--dataset",
        "-d",
        help="Eval split (JSONL)",
    ),
    quantization: Literal["auto", "4bit", "8bit", "none"] = typer.Option(
        "auto", "--quantization", help="4bit, 8bit, none or auto (device default)"
    ),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    config: str = typer.Option(None, "--config", "-c"),
    batch_size: int = typer.Option(
        0, "--batch-size", help="Generation batch size (0 = config default)"
    ),
) -> None:
    """Task-level evaluation: verdict + CWE accuracy (security) or verdict + flag accuracy (CTF)."""
    from brainforge.config import load_config
    from brainforge.training.task_eval import evaluate_model_on_records

    training = load_config(config).training
    result = evaluate_model_on_records(
        model or latest_run_dir(),
        dataset,
        quantization,
        max_new_tokens,
        batch_size=batch_size or training.generate_batch_size,
    )
    console.print(f"[green]task evaluation done[/green] ({result['n_records']} records)")
    for key in (
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "cwe_accuracy",
        "flag_accuracy",
    ):
        value = result.get(key)
        console.print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


@app.command("export")
@clamp_errors("export")
def export(
    model: Path = typer.Option(
        None, "--model", "-m", help="Trained adapter directory (default: latest run)"
    ),
    output: Path = typer.Option(
        Path("models/export"), "--output", "-o", help="Export output directory"
    ),
    format: Literal["merged", "gguf"] = typer.Option(
        "merged", "--format", "-f", help="merged: HF weights; gguf: quantized GGUF + Modelfile"
    ),
    quant: str = typer.Option("Q4_K_M", "--quant", help="GGUF quantization type (gguf format)"),
    ollama: str = typer.Option(
        None, "--ollama", help="Register the GGUF with Ollama under this name (gguf format)"
    ),
    llama_cpp_path: Path = typer.Option(
        None, "--llama-cpp-path", help="Path to a llama.cpp checkout (gguf format)"
    ),
) -> None:
    """Export a trained student: merged HF weights or quantized GGUF."""
    from brainforge.training.qlora import export as export_merged

    if format == "merged":
        result = export_merged(model or latest_run_dir(), output)
        console.print(f"[green]exported[/green] -> {result['output_dir']}")
        return
    from brainforge.training.gguf import export_gguf

    merged_dir = Path(output) / "merged"
    export_merged(model or latest_run_dir(), merged_dir)
    result = export_gguf(
        merged_dir,
        output,
        quant=quant,
        llama_cpp_path=llama_cpp_path,
        ollama_name=ollama,
    )
    console.print(f"[green]exported gguf[/green] -> {result['output_dir']}")
    console.print(f"  quant: {result['quant']}")
    console.print(f"  modelfile: {result['modelfile']}")
    if result["ollama"]:
        console.print(f"  ollama: {result['ollama']}")

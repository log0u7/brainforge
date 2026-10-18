from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import typer

from brainforge.cli._common import clamp_errors, console
from brainforge.errors import BrainforgeError

app = typer.Typer(
    help="Training: prepare datasets, run QLoRA or full fine-tuning, evaluate, export.",
    no_args_is_help=True,
)


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
    backend: Literal["trl", "unsloth"] | None = typer.Option(
        None, "--backend", help="Override config training backend (trl or unsloth)"
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
    """Run QLoRA or full fine-tuning (requires a CUDA GPU and the training extra)."""
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
    if updates:
        training = training.model_copy(update=updates)
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
    if resume and output is None:
        output_dir = latest_run_dir(base=Path(training.output_dir))
    else:
        output_dir = output or Path(training.output_dir) / run_name
    if training.backend == "unsloth":
        from brainforge.training.unsloth_backend import train_unsloth as train_fn
    else:
        from brainforge.training.qlora import train_qlora as train_fn
    summary = train_fn(training, dataset_dir, output_dir, resume=resume)
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
    """Interactive chat with a trained student model (requires CUDA + training extra)."""
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
    quantization: str = typer.Option("4bit", "--quantization", help="4bit, 8bit or none"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    config: str = typer.Option(None, "--config", "-c"),
    batch_size: int = typer.Option(
        0, "--batch-size", help="Generation batch size (0 = config default)"
    ),
) -> None:
    """Task-level evaluation: verdict + CWE accuracy on a held-out split."""
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
    for key in ("accuracy", "false_positive_rate", "false_negative_rate", "cwe_accuracy"):
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

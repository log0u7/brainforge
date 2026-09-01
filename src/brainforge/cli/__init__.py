import typer

from brainforge.cli import (
    config_cmd,
    dataset_cmd,
    models_cmd,
    pipeline_cmd,
    rag_cmd,
    roles_cmd,
    teacher_cmd,
    train_cmd,
)

app = typer.Typer(
    name="brainforge",
    help=(
        "Build security & coding datasets with multi-teacher LLM pipelines"
        " and train small local models."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

for sub_app, name, help_text in [
    (config_cmd, "config", "Configuration utilities."),
    (models_cmd, "models", "Model registry operations."),
    (roles_cmd, "roles", "Role inspection."),
    (teacher_cmd, "teacher", "Run teachers on cases."),
    (pipeline_cmd, "pipeline", "Run dataset generation pipelines."),
    (rag_cmd, "rag", "Local RAG index and search."),
    (dataset_cmd, "dataset", "Dataset utilities."),
    (train_cmd, "train", "Training: prepare datasets, run QLoRA, evaluate, export."),
]:
    app.add_typer(sub_app.app, name=name, help=help_text)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

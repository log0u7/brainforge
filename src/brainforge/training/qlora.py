from brainforge.config.models import TrainingConfig
from brainforge.errors import BrainforgeError


def train_qlora(config: TrainingConfig, dataset_dir, output_dir) -> None:
    raise BrainforgeError(
        "QLoRA training is planned for phase 2; see ROADMAP.md. "
        "Use 'brainforge train prepare' to export a TRL-ready dataset in the meantime."
    )


def evaluate(model_path, eval_dataset) -> None:
    raise BrainforgeError(
        "Student evaluation (including the post-cutoff benchmark) is planned for "
        "phase 2; see ROADMAP.md."
    )


def export(model_path, output_dir) -> None:
    raise BrainforgeError("Model export (merged weights, GGUF) is planned for phase 2.")

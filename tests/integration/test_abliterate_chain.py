"""Abliteration integration test: real weight surgery on a tiny random model.

Opt-in via BRAINFORCE_IT=1 (marker ``integration``). Builds a tiny random
Llama locally (no network) and verifies the refusal direction shrinks after
abliteration and the modified model round-trips.
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("BRAINFORCE_IT") != "1",
        reason="integration tests are opt-in via BRAINFORCE_IT=1",
    ),
]


def _tiny_model(tmp_path: Path, prompts: list[str]) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    model_dir = tmp_path / "tiny-model"
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
    )
    LlamaForCausalLM(config).save_pretrained(model_dir)

    vocab: dict[str, int] = {"[UNK]": 0, "[PAD]": 1}
    for prompt in prompts:
        for word in prompt.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.save_pretrained(model_dir)
    return model_dir


def _calibration(tmp_path: Path) -> Path:
    path = tmp_path / "calibration.jsonl"
    rows = [{"prompt": f"harmful pattern {index}", "label": "harmful"} for index in range(4)] + [
        {"prompt": f"benign pattern {index}", "label": "benign"} for index in range(4)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _calibration_prompts() -> list[str]:
    return [f"harmful pattern {index}" for index in range(4)] + [
        f"benign pattern {index}" for index in range(4)
    ]


def test_abliteration_shrinks_refusal_direction(tmp_path):
    from brainforge.training.abliterate import abliterate

    model_dir = _tiny_model(tmp_path, _calibration_prompts())
    stats = abliterate(model_dir, _calibration(tmp_path), tmp_path / "out", layers="all")
    assert stats["direction_norm_before"] > 0
    assert stats["direction_norm_after"] < stats["direction_norm_before"] * 0.5
    assert stats["suppression_ratio"] < 0.5
    assert stats["layers_edited"] == [0, 1, 2, 3]

    stored = json.loads((tmp_path / "out" / "ablation_stats.json").read_text())
    assert stored["suppression_ratio"] == stats["suppression_ratio"]

    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "out")
    assert reloaded.config.hidden_size == 32


def test_abliteration_layer_subset(tmp_path):
    from brainforge.training.abliterate import abliterate

    model_dir = _tiny_model(tmp_path, _calibration_prompts())
    stats = abliterate(model_dir, _calibration(tmp_path), tmp_path / "out", layers="0,2")
    assert stats["layers_edited"] == [0, 2]

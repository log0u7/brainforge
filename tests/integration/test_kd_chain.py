"""KD integration tests: loss math, real-tokenizer alignment, CLI chain.

Opt-in via BRAINFORCE_IT=1 (marker ``integration``). The loss math and
alignment run on CPU with a locally built tiny model (no network); the CLI
chain uses the mock provider.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("BRAINFORCE_IT") != "1",
        reason="integration tests are opt-in via BRAINFORCE_IT=1",
    ),
]


class FakeVocabTokenizer:
    def get_vocab(self):
        return {"a": 0, "b": 1, "c": 2}


def test_compute_kd_loss_prefers_matching_distribution():
    import torch

    from brainforge.training.kd import compute_kd_loss

    tokenizer = FakeVocabTokenizer()
    topk = [("a", -0.1), ("b", -3.0)]
    # student puts mass on "a" (matches teacher)
    good = torch.tensor([5.0, 0.0, 0.0])
    # student puts mass on "b" (contradicts teacher)
    bad = torch.tensor([0.0, 5.0, 0.0])
    good_loss = compute_kd_loss(good, topk, tokenizer, temperature=1.0)
    bad_loss = compute_kd_loss(bad, topk, tokenizer, temperature=1.0)
    assert good_loss < bad_loss
    assert good_loss.item() > 0


def test_compute_kd_loss_skips_unknown_tokens():
    import torch

    from brainforge.training.kd import compute_kd_loss

    tokenizer = FakeVocabTokenizer()
    # all top-k tokens outside the student vocabulary
    loss = compute_kd_loss(torch.tensor([1.0, 2.0, 3.0]), [("zz", -0.1)], tokenizer, 1.0)
    assert loss is None


def _real_tokenizer():
    """A real BPE tokenizer built locally (byte-level, exact reconstruction)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300, special_tokens=["[UNK]", "[PAD]"], show_progress=False
    )
    corpus = [
        "<u>hi</u><a>hello world</a>",
        "<u>say it</u><a>hello world</a>",
        "user: say it\nassistant: hello world\n",
        "hello world this is a test of the tokenizer",
    ]
    backend.train_from_iterator(corpus, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    )
    return tokenizer


def test_align_teacher_positions_with_real_tokenizer():
    from brainforge.training.kd import align_teacher_positions

    tokenizer = _real_tokenizer()
    content = "hello world"
    full_text = "<u>hi</u><a>" + content
    content_start = full_text.rfind(content)
    teacher_positions = [
        {"token": "hello ", "logprob": -0.1, "topk": [["hello", -0.1]]},
        {"token": "world", "logprob": -0.2, "topk": [["world", -0.2]]},
    ]
    aligned = align_teacher_positions(
        teacher_positions, content, content_start, full_text, tokenizer
    )
    assert len(aligned) == 2
    indices = [index for index, _ in aligned]
    assert len(set(indices)) == 2
    assert all(index >= 0 for index in indices)
    # each predictor position decodes to a token inside the content region
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    for index in indices:
        decoded = tokenizer.decode([input_ids[index + 1]])
        assert decoded in content or content in full_text


def test_tokenize_row_marks_assistant_and_aligns():
    from brainforge.training.kd import _tokenize_row

    tokenizer = _real_tokenizer()
    content = "hello world"
    record = {
        "id": "r1",
        "messages": [
            {"role": "user", "content": "say it"},
            {"role": "assistant", "content": content},
        ],
    }
    teacher = {
        "positions": [
            {"token": "hello ", "logprob": -0.1, "topk": [["hello", -0.1]]},
            {"token": "world", "logprob": -0.2, "topk": [["world", -0.2]]},
        ]
    }
    row = _tokenize_row(record, teacher, tokenizer)
    assert row["content_token_count"] >= 2
    trained = [token for token in row["labels"] if token != -100]
    assert len(trained) == row["content_token_count"]
    assert row["kd_targets"]
    for index, _topk in row["kd_targets"]:
        assert 0 <= index < len(row["input_ids"])


def test_cli_chain_pipeline_kd_sidecar(workspace: Path):
    dataset = workspace / "datasets" / "security_dataset.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from brainforge.cli import app; app()",
            "pipeline",
            "run",
            "security_dataset",
            "--input",
            str(workspace / "data" / "raw"),
            "--output",
            str(dataset),
            "--provider",
            "mock",
            "--kd",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    sidecar = workspace / "datasets" / "security_dataset.kd.jsonl"
    assert sidecar.is_file()
    records = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    sidecar_records = [
        json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()
    ]
    assert len(sidecar_records) == len(records)
    by_id = {record["id"]: record for record in sidecar_records}
    for record in records:
        entry = by_id[record["id"]]
        assistant = record["messages"][-1]["content"]
        rebuilt = "".join(position["token"] for position in entry["positions"])
        assert rebuilt == assistant


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    config = json.loads((REPO_ROOT / "config" / "config.json").read_text())
    config["providers"]["mock_only"] = {"type": "mock"}
    for model in config["models"].values():
        model["provider"] = "mock_only"
    config["judge_independence"] = "off"
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.json").write_text(json.dumps(config))
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for case_file in (REPO_ROOT / "examples" / "cases").glob("case-*.json"):
        shutil.copy(case_file, raw / case_file.name)
    return tmp_path

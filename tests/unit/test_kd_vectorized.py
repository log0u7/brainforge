"""Equivalence and speed of the vectorized KD loss (issue #57).

The vectorized path must produce the same loss as the per-position reference
implementation (< 1e-5), including the None-when-no-valid-token semantics.
CI's unit job runs without the training extra, so skip cleanly when torch is
absent (the real training path is covered by the opt-in integration suite).
"""

import json
import time

import pytest

torch = pytest.importorskip("torch")

from brainforge.training.kd import (  # noqa: E402
    KDDataCollator,
    compute_kd_loss,
    compute_kd_loss_batch,
)

VOCAB_SIZE = 128
OOV_TOKEN = "\u00ff"  # absent from the vocab: must be dropped by both paths


class _VocabTokenizer:
    """Vocabulary chr(0)..chr(127): keeps the loss math small and exact."""

    pad_token_id = 0

    def get_vocab(self):
        return {chr(i): i for i in range(VOCAB_SIZE)}


def _random_targets(rng, count, k):
    """(position, [[token, logprob], ...]) rows mixing in-vocab and OOV tokens."""
    targets = []
    for position in rng.sample(range(VOCAB_SIZE - 1), count):
        topk = []
        for _ in range(rng.randint(0, k)):
            token = OOV_TOKEN if rng.random() < 0.2 else chr(rng.randrange(VOCAB_SIZE))
            topk.append([token, float(rng.uniform(-10.0, -0.01))])
        targets.append((position, topk))
    return targets


def _features(targets_per_row):
    return [
        {
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "kd_targets": json.dumps(targets, default=list),
        }
        for targets in targets_per_row
    ]


def _reference_loss(logits, targets_per_row, tokenizer, temperature):
    """The pre-vectorization aggregation: one loss call per position."""
    terms = []
    for row, targets in enumerate(targets_per_row):
        for token_index, topk in targets:
            if not topk:
                continue
            term = compute_kd_loss(logits[row, token_index], topk, tokenizer, temperature)
            if term is not None:
                terms.append(term)
    return torch.stack(terms).mean() if terms else None


def _vectorized_loss(logits, targets_per_row, tokenizer, temperature):
    batch = KDDataCollator(tokenizer)(_features(targets_per_row))
    return compute_kd_loss_batch(
        logits,
        batch["kd_positions"],
        batch["kd_topk_ids"],
        batch["kd_topk_logprobs"],
        batch["kd_topk_mask"],
        batch["kd_position_mask"],
        temperature,
    )


def test_equivalence_random_batches():
    import random

    tokenizer = _VocabTokenizer()
    for seed in range(5):
        rng = random.Random(seed)
        targets_per_row = [_random_targets(rng, rng.randint(1, 12), 8) for _ in range(3)]
        logits = torch.randn(3, VOCAB_SIZE, 4 * VOCAB_SIZE)
        for temperature in (1.0, 2.0):
            reference = _reference_loss(logits, targets_per_row, tokenizer, temperature)
            vectorized = _vectorized_loss(logits, targets_per_row, tokenizer, temperature)
            if reference is None:
                assert vectorized is None
            else:
                assert vectorized is not None
                difference = (reference - vectorized).abs().item()
                assert difference < 1e-5, f"seed={seed} T={temperature}: diff={difference}"


def test_all_oov_positions_match_reference_none():
    tokenizer = _VocabTokenizer()
    targets_per_row = [[(3, [[OOV_TOKEN, -0.5]])], [(1, [[OOV_TOKEN, -0.1]])]]
    logits = torch.randn(2, VOCAB_SIZE, 4 * VOCAB_SIZE)
    assert _reference_loss(logits, targets_per_row, tokenizer, 2.0) is None
    assert _vectorized_loss(logits, targets_per_row, tokenizer, 2.0) is None


def test_collator_builds_padded_tensors():
    tokenizer = _VocabTokenizer()
    targets_per_row = [
        [(5, [["a", -0.1], [OOV_TOKEN, -2.0]]), (9, [])],  # second position: empty topk
        [(2, [["b", -0.3]])],
        [],  # row without any target
    ]
    batch = KDDataCollator(tokenizer)(_features(targets_per_row))
    positions = batch["kd_positions"]
    topk_ids = batch["kd_topk_ids"]
    topk_logprobs = batch["kd_topk_logprobs"]
    topk_mask = batch["kd_topk_mask"]
    position_mask = batch["kd_position_mask"]
    assert positions.shape == (3, 2)
    assert topk_ids.shape == (3, 2, 2)
    assert topk_logprobs.dtype == torch.float64
    assert positions.dtype == torch.long
    # position 5 keeps its single in-vocab pair; empty-topk position 9 is dropped
    assert position_mask[0].tolist() == [True, False]
    assert positions[0, 0].item() == 5
    assert topk_ids[0, 0].tolist() == [ord("a"), 0]
    assert topk_mask[0, 0].tolist() == [True, False]  # OOV entry masked out
    assert topk_logprobs[0, 0, 0].item() == pytest.approx(-0.1)
    assert not position_mask[2].any()  # empty row contributes nothing


def test_vectorized_speed_smoke():
    """Timing smoke only: CI machines vary, so print the numbers, no assert.

    The production pathology is GPU-bound (one kernel launch + host-to-device
    tensor build per position over a 150k vocab, issue #57); a CPU smoke
    therefore shows a conservative speedup.
    """
    tokenizer = _VocabTokenizer()
    rng = __import__("random").Random(0)
    batch_size, positions, k, vocab_size, sequence_length = 2, 300, 8, 32768, 512
    targets_per_row = [
        [
            (
                position,
                [
                    [chr(rng.randrange(VOCAB_SIZE)), float(rng.uniform(-10.0, -0.01))]
                    for _ in range(k)
                ],
            )
            for position in rng.sample(range(sequence_length - 1), positions)
        ]
        for _ in range(batch_size)
    ]
    logits = torch.randn(batch_size, sequence_length, vocab_size)

    start = time.perf_counter()
    reference = _reference_loss(logits, targets_per_row, tokenizer, 2.0)
    reference_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    vectorized = _vectorized_loss(logits, targets_per_row, tokenizer, 2.0)
    vectorized_ms = (time.perf_counter() - start) * 1000

    assert reference is not None and vectorized is not None
    assert (reference - vectorized).abs().item() < 1e-5
    print(
        f"\nKD loss on {batch_size * positions} positions (vocab {vocab_size}): "
        f"reference {reference_ms:.1f} ms, vectorized {vectorized_ms:.1f} ms "
        f"(x{reference_ms / max(vectorized_ms, 1e-9):.1f})"
    )


class _FakeOutput:
    def __init__(self, logits):
        self.loss = torch.tensor(0.5)
        self.logits = logits


class _FakeModel:
    """Stands in for the CausalLM forward: no transformers needed."""

    def __init__(self, batch, sequence_length):
        self.batch = batch
        self.sequence_length = sequence_length

    def __call__(self, input_ids, attention_mask, labels):
        return _FakeOutput(torch.randn(self.batch, self.sequence_length, 4 * VOCAB_SIZE))


def _bare_trainer():
    """A KDTrainer without the transformers Trainer base: compute_loss only
    needs kd_alpha, kd_temperature and the tokenizer attribute."""
    from brainforge.training.kd import KDTrainer

    trainer = KDTrainer.__new__(KDTrainer)
    trainer.kd_alpha = 0.5
    trainer.kd_temperature = 2.0
    trainer.processing_class = _VocabTokenizer()
    return trainer


def test_compute_loss_uses_the_encoded_targets():
    features = [
        {
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
            "kd_targets": json.dumps([(1, [["a", -0.1]])]),
        }
    ]
    inputs = KDDataCollator(_VocabTokenizer())(features)
    loss = _bare_trainer().compute_loss(_FakeModel(1, 3), inputs)
    assert loss.ndim == 0


def test_compute_loss_falls_back_to_the_reference_loop():
    inputs = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.zeros(1, 3, dtype=torch.long),
        "kd_targets": [[(1, [["a", -0.1]])]],
    }
    loss = _bare_trainer().compute_loss(_FakeModel(1, 3), inputs)
    assert loss.ndim == 0


def test_compute_loss_without_tokenizer_raises():
    from brainforge.errors import BrainforgeError

    inputs = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.zeros(1, 3, dtype=torch.long),
        "kd_targets": [[(1, [["a", -0.1]])]],
    }
    trainer = _bare_trainer()
    trainer.processing_class = None
    trainer.tokenizer = None
    with pytest.raises(BrainforgeError):
        trainer.compute_loss(_FakeModel(1, 3), inputs)

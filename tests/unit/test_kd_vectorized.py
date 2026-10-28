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
    _compute_kd_loss_reference,
    compute_kd_loss,
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
            term = _compute_kd_loss_reference(
                logits[row, token_index], topk, tokenizer, temperature
            )
            if term is not None:
                terms.append(term)
    return torch.stack(terms).mean() if terms else None


def _vectorized_loss(logits, targets_per_row, tokenizer, temperature):
    batch = KDDataCollator(tokenizer)(_features(targets_per_row))
    return compute_kd_loss(
        logits,
        batch["kd_position_ids"],
        batch["kd_topk_ids"],
        batch["kd_topk_logprobs"],
        batch["kd_mask"],
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
    position_ids = batch["kd_position_ids"]
    topk_ids = batch["kd_topk_ids"]
    topk_logprobs = batch["kd_topk_logprobs"]
    mask = batch["kd_mask"]
    assert position_ids.shape == (3, 2)
    assert topk_ids.shape == (3, 2, 2)
    assert topk_logprobs.dtype == torch.float32
    assert position_ids.dtype == torch.long
    # row 0 keeps only its in-vocab pair at position slot 0; position 9 is gone
    assert position_ids[0].tolist() == [5, 2]
    assert topk_ids[0, 0].tolist() == [ord("a"), 0]
    assert mask[0, 0].tolist() == [True, False]  # OOV entry masked out
    assert not mask[2].any()  # empty row contributes nothing


def test_vectorized_speed_smoke():
    """Timing smoke only: CI machines vary, so print the numbers, no assert."""
    tokenizer = _VocabTokenizer()
    rng = __import__("random").Random(0)
    positions, k, vocab_size, sequence_length = 100, 8, 4096, 256
    targets_per_row = [_random_targets(rng, positions, k) for _ in range(2)]
    # remap positions into the sequence length and force in-vocab tokens
    targets_per_row = [
        [(p % (sequence_length - 1), topk) for p, topk in row] for row in targets_per_row
    ]
    logits = torch.randn(2, sequence_length, vocab_size)

    start = time.perf_counter()
    reference = _reference_loss(logits, targets_per_row, tokenizer, 2.0)
    reference_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    vectorized = _vectorized_loss(logits, targets_per_row, tokenizer, 2.0)
    vectorized_ms = (time.perf_counter() - start) * 1000

    assert reference is not None and vectorized is not None
    assert (reference - vectorized).abs().item() < 1e-5
    print(
        f"\nKD loss on {2 * positions} positions: "
        f"reference {reference_ms:.1f} ms, vectorized {vectorized_ms:.1f} ms "
        f"(x{reference_ms / max(vectorized_ms, 1e-9):.1f})"
    )

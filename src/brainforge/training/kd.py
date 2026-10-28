"""Knowledge distillation training: SFT + KL to the teacher's top-k logprobs.

The dataset (``train.jsonl``) plus a ``kd.jsonl`` sidecar (produced by
``pipeline run --kd``, one record of per-token teacher top-k logprobs per
dataset id) are tokenized with the student's own chat template. Teacher
positions are mapped onto student tokens by character midpoint, so the
student and teacher may use different tokenizers. Top-k tokens that are
absent from the student vocabulary are dropped and the teacher distribution
is renormalized over the remaining tokens.

Loss per position: ``alpha * T^2 * CE(student_T, teacher_topk_T)
+ (1 - alpha) * CE(gold)``.

Experimental. Heavy dependencies (torch, transformers, datasets) are
imported lazily so the package and the unit test suite run without the
``training`` extra.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from brainforge.config.models import TrainingConfig
from brainforge.dataset.writer import file_digest, iter_jsonl, read_jsonl
from brainforge.errors import BrainforgeError
from brainforge.training.hardware import torch_dtype
from brainforge.training.qlora import (
    _latest_checkpoint,
    _lora_config,
    _quantization_config,
    _trainer_kwargs,
)


def align_teacher_positions(
    teacher_positions: list[dict],
    content: str,
    content_start: int,
    full_text: str,
    tokenizer: Any,
) -> list[tuple[int, list[tuple[str, float]]]]:
    """Map teacher top-k positions onto student token indices by char midpoint.

    ``content_start`` is the char offset of the assistant content inside the
    fully rendered chat text. Each teacher token (tokens concatenate back to
    the content) is attributed to the student token whose char span contains
    its midpoint. Returns (student_token_index, topk) pairs; the index is the
    predictor position (token_index - 1) used by the KD loss.
    """
    if not teacher_positions:
        return []
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    spans: list[tuple[int, int]] = []
    offset = 0
    for token_id in input_ids:
        text = tokenizer.decode([token_id])
        spans.append((offset, offset + len(text)))
        offset += len(text)

    aligned: list[tuple[int, list[tuple[str, float]]]] = []
    teacher_offset = 0
    for position in teacher_positions:
        token = position.get("token", "")
        midpoint = content_start + teacher_offset + len(token) / 2
        teacher_offset += len(token)
        for index, (start, end) in enumerate(spans):
            if start <= midpoint < end:
                if index > 0:
                    aligned.append((index - 1, position.get("topk", [])))
                break
    return aligned


def _compute_kd_loss_reference(
    logits: Any,
    topk: list[tuple[str, float]],
    tokenizer: Any,
    temperature: float,
) -> Any | None:
    """Pre-vectorization per-position loss, kept as the equivalence reference
    (issue #57 tests): one log_softmax over the full vocabulary per call."""
    import torch

    pairs: list[tuple[int, float]] = []
    # get_vocab() builds a 250k-entry dict: cache it on the tokenizer or the
    # per-position loop costs minutes per optimizer step (issue #57).
    vocab = getattr(tokenizer, "_kd_vocab_cache", None)
    if vocab is None:
        vocab = tokenizer.get_vocab()
        tokenizer._kd_vocab_cache = vocab
    for token, logprob in topk:
        token_id = vocab.get(token)
        if token_id is not None:
            pairs.append((token_id, logprob))
    if not pairs:
        return None
    student_logprobs = torch.log_softmax(logits / temperature, dim=-1)
    teacher_lps = torch.tensor(
        [logprob for _, logprob in pairs], device=logits.device, dtype=logits.dtype
    )
    teacher_probs = torch.softmax(teacher_lps / temperature, dim=-1)
    token_ids = torch.tensor([token_id for token_id, _ in pairs], device=logits.device)
    student_q = student_logprobs[token_ids]
    return -(teacher_probs * student_q).sum() * (temperature**2)


_KD_LOG_SOFTMAX_CHUNK = 1024
"""Positions per log_softmax chunk in compute_kd_loss_batch: bounds the
[chunk, vocab] peak memory for 150k-vocab models (issue #57) while keeping
one kernel call per chunk instead of one per position."""


def compute_kd_loss_batch(
    logits: Any,
    positions: Any,
    topk_ids: Any,
    topk_logprobs: Any,
    topk_mask: Any,
    position_mask: Any,
    temperature: float,
) -> Any | None:
    """Vectorized batch form of compute_kd_loss over collator-encoded targets.

    ``positions``/``position_mask`` are [B, P]; ``topk_ids``/``topk_logprobs``/
    ``topk_mask`` are [B, P, K]. Returns the mean loss over the positions with
    at least one in-vocabulary top-k token (None when there is none), i.e.
    exactly the value KDTrainer.compute_loss used to obtain by looping
    compute_kd_loss over (row, position) pairs: teacher top-k renormalized
    over the surviving tokens, student log-softmax at the temperature, T^2
    scaling per position.
    """
    import torch

    if not bool(position_mask.any()):
        return None
    batch_index, position_index = position_mask.nonzero(as_tuple=True)
    keep = topk_mask[batch_index, position_index]
    ids = topk_ids[batch_index, position_index]
    # invalid top-k entries get teacher probability exactly 0 (that is the
    # renormalization); the cast to the logits dtype rounds the float64
    # logprobs once, like the reference loss builds its tensor.
    teacher = topk_logprobs[batch_index, position_index].to(logits.dtype)
    teacher = teacher.masked_fill(~keep, float("-inf"))
    terms: list[Any] = []
    for start in range(0, ids.shape[0], _KD_LOG_SOFTMAX_CHUNK):
        rows = slice(start, start + _KD_LOG_SOFTMAX_CHUNK)
        chunk_rows = batch_index[rows]
        # index per chunk: the gathered [chunk, vocab] copy, not the full
        # [N, vocab] one, must be the materialized tensor.
        flat_logits = logits[chunk_rows, positions[chunk_rows, position_index[rows]]]
        student_q = (
            torch.log_softmax(flat_logits / temperature, dim=-1)
            .gather(1, ids[rows])
            .masked_fill(~keep[rows], 0.0)
        )
        teacher_probs = torch.softmax(teacher[rows] / temperature, dim=-1)
        terms.append(-(teacher_probs * student_q).sum(dim=-1) * (temperature**2))
    return torch.cat(terms).mean()


class KDDataCollator:
    """Pads input_ids/labels and encodes per-row teacher targets into tensors.

    The KD targets are pre-encoded here (issue #57): token strings are mapped
    to student vocabulary ids once per batch, so ``KDTrainer.compute_loss``
    runs one vectorized kernel sequence instead of a Python loop per position.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        # get_vocab() builds a 250k-entry dict: build it once here and share
        # the same cache the per-position reference loss uses (issue #57).
        vocab = getattr(tokenizer, "_kd_vocab_cache", None)
        if vocab is None:
            vocab = tokenizer.get_vocab()
            tokenizer._kd_vocab_cache = vocab
        self._vocab: dict[str, int] = vocab

    def __call__(self, features: list[dict]) -> dict:
        import torch

        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths)
        pad_id = self.tokenizer.pad_token_id or 0
        batch: dict[str, Any] = {
            "input_ids": torch.full((len(features), max_length), pad_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(features), max_length), dtype=torch.long),
            "labels": torch.full((len(features), max_length), -100, dtype=torch.long),
        }
        for row, feature in enumerate(features):
            ids = torch.tensor(feature["input_ids"], dtype=torch.long)
            batch["input_ids"][row, : len(ids)] = ids
            batch["attention_mask"][row, : len(ids)] = 1
            labels = torch.tensor(feature["labels"], dtype=torch.long)
            batch["labels"][row, : len(labels)] = labels
        # kd_targets arrive as JSON strings (arrow cannot type the
        # heterogeneous [str, float] topk pairs) - decode here.
        kd_targets = [
            json.loads(feature["kd_targets"])
            if isinstance(feature["kd_targets"], str)
            else feature["kd_targets"]
            for feature in features
        ]
        batch["kd_targets"] = kd_targets
        batch.update(self._encode_kd_targets(kd_targets))
        return batch

    def _encode_kd_targets(self, kd_targets: list[Any]) -> dict[str, Any]:
        """Encode per-row (position, topk) targets into padded tensors.

        Rows are padded to [B, max_positions] and [B, max_positions, K].
        Top-k tokens missing from the student vocabulary are masked out: the
        loss renormalizes the teacher over the surviving tokens, exactly like
        ``compute_kd_loss`` drops them. Logprobs are kept in float64 so the
        cast to the logits dtype rounds once, like the reference loss.
        """
        import torch

        rows = len(kd_targets)
        max_positions = max((len(targets) for targets in kd_targets), default=0)
        max_topk = max((len(topk) for targets in kd_targets for _, topk in targets), default=0)
        positions = torch.zeros((rows, max_positions), dtype=torch.long)
        topk_ids = torch.zeros((rows, max_positions, max_topk), dtype=torch.long)
        topk_logprobs = torch.zeros((rows, max_positions, max_topk), dtype=torch.float64)
        topk_mask = torch.zeros((rows, max_positions, max_topk), dtype=torch.bool)
        position_mask = torch.zeros((rows, max_positions), dtype=torch.bool)
        for row, targets in enumerate(kd_targets):
            for index, (token_index, topk) in enumerate(targets):
                valid = [
                    (token_id, logprob)
                    for token, logprob in topk
                    if (token_id := self._vocab.get(token)) is not None
                ]
                if not valid:
                    continue
                position_mask[row, index] = True
                positions[row, index] = token_index
                topk_ids[row, index, : len(valid)] = torch.tensor(
                    [token_id for token_id, _ in valid], dtype=torch.long
                )
                topk_logprobs[row, index, : len(valid)] = torch.tensor(
                    [logprob for _, logprob in valid], dtype=torch.float64
                )
                topk_mask[row, index, : len(valid)] = True
        return {
            "kd_positions": positions,
            "kd_topk_ids": topk_ids,
            "kd_topk_logprobs": topk_logprobs,
            "kd_topk_mask": topk_mask,
            "kd_position_mask": position_mask,
        }


class KDTrainer:
    """Mixin adding the distillation term to a transformers Trainer."""

    def __init__(
        self, *args: Any, kd_alpha: float = 0.5, kd_temperature: float = 2.0, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature

    def compute_loss(
        self, model: Any, inputs: dict, return_outputs: bool = False, **kwargs: Any
    ) -> Any:
        kd_targets = inputs.pop("kd_targets")
        encoded = inputs.pop("kd_positions", None)
        if encoded is not None:
            kd_topk_ids = inputs.pop("kd_topk_ids")
            kd_topk_logprobs = inputs.pop("kd_topk_logprobs")
            kd_topk_mask = inputs.pop("kd_topk_mask")
            kd_position_mask = inputs.pop("kd_position_mask")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        ce_loss = outputs.loss
        logits = outputs.logits
        kd_terms = []
        if encoded is not None:
            # vectorized path (issue #57): the collator pre-encoded the
            # teacher targets; one kernel sequence replaces the per-position
            # Python loop.
            term = compute_kd_loss_batch(
                logits,
                encoded,
                kd_topk_ids,
                kd_topk_logprobs,
                kd_topk_mask,
                kd_position_mask,
                self.kd_temperature,
            )
            if term is not None:
                kd_terms.append(term)
        else:
            # fallback for batches produced without the collator encoding
            tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            if tokenizer is None:
                raise BrainforgeError("KD trainer has no tokenizer; cannot align teacher tokens")
            for row, targets in enumerate(kd_targets):
                for token_index, topk in targets:
                    if not topk:
                        continue
                    term = compute_kd_loss(
                        logits[row, token_index], topk, tokenizer, self.kd_temperature
                    )
                    if term is not None:
                        kd_terms.append(term)
        if kd_terms:
            import torch

            kd_loss = torch.stack(kd_terms).mean()
            loss = self.kd_alpha * kd_loss + (1 - self.kd_alpha) * ce_loss
        else:
            loss = ce_loss
        return (loss, outputs) if return_outputs else loss


def _kd_row_stream(
    records_path: Path, teacher_by_id: dict, tokenizer: Any, content_digest: str = ""
) -> Iterator[dict]:
    """Tokenize records one at a time, for Dataset.from_generator (bounded RAM).

    ``content_digest`` is unused here: it only feeds the datasets cache
    fingerprint so a regenerated file invalidates the cached arrow table.
    """
    for record in iter_jsonl(records_path):
        yield _tokenize_row(record, teacher_by_id[record["id"]], tokenizer)


def _tokenize_row(record: dict, teacher: dict, tokenizer: Any) -> dict:
    messages = record["messages"]
    if any(message.get("role") == "tool" for message in messages):
        raise BrainforgeError(
            f"dataset record '{record['id']}' contains tool messages; knowledge "
            "distillation supports single-turn records only"
        )
    content = messages[-1]["content"]
    if tokenizer.chat_template is None:
        raise BrainforgeError(
            "tokenizer for KD training has no chat template; cannot align teacher positions"
        )
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    content_start = full_text.rfind(content)
    if content_start < 0:
        raise BrainforgeError(
            f"assistant content of record '{record['id']}' not found in the rendered chat"
        )
    input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(input_ids)
    offset = 0
    content_token_count = 0
    for index, token_id in enumerate(input_ids):
        text = tokenizer.decode([token_id])
        start, end = offset, offset + len(text)
        offset = end
        if start >= content_start:
            labels[index] = token_id
            content_token_count += 1
    kd_targets = align_teacher_positions(
        teacher["positions"], content, content_start, full_text, tokenizer
    )
    return {
        "id": record["id"],
        "input_ids": input_ids,
        "labels": labels,
        "kd_targets": kd_targets,
        "content_token_count": content_token_count,
    }


def train_kd(
    config: TrainingConfig,
    dataset_dir: Path | str,
    output_dir: Path | str,
    resume: bool | str | Path = False,
) -> dict:
    """Fine-tune with SFT + knowledge distillation from the teacher sidecar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _latest_checkpoint(output_dir) if resume else None
    dataset_dir = Path(dataset_dir)
    sidecar_path = dataset_dir / "kd.jsonl"
    if not sidecar_path.exists():
        raise BrainforgeError(
            f"KD training requires {sidecar_path}; regenerate the dataset with "
            "'brainforge pipeline run --kd'"
        )
    if config.packing:
        raise BrainforgeError("KD training does not support packing; set packing to false")
    train_path = dataset_dir / "train.jsonl"
    if not train_path.is_file():
        raise BrainforgeError(f"missing {train_path}")
    if train_path.stat().st_size == 0:
        # Before any heavy import: unit-testable and fails fast.
        raise BrainforgeError(f"{train_path} is empty; regenerate the dataset first")
    from brainforge.training.hardware import plan_for_config

    plan = plan_for_config(config, kd=True)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    teacher_by_id = {entry["id"]: entry for entry in read_jsonl(sidecar_path)}

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    validation_path = dataset_dir / "validation.jsonl"
    splits = [train_path, *((validation_path,) if validation_path.exists() else ())]
    for split_path in splits:
        missing = next(
            (
                record["id"]
                for record in iter_jsonl(split_path)
                if record["id"] not in teacher_by_id
            ),
            None,
        )
        if missing is not None:
            raise BrainforgeError(
                f"dataset record '{missing}' has no teacher positions in {sidecar_path}"
            )

    # Eager materialization (Dataset.from_list): KD datasets are tiny by
    # construction (teacher positions live in RAM anyway), and from_generator
    # fingerprints closures unstably - train and eval builders collided on
    # the same cache dir and self-locked on Windows (WinError 32, observed
    # on gpu3080).
    def _materialize(records_path: Path) -> list[dict]:
        rows = list(
            _kd_row_stream(
                records_path=records_path,
                teacher_by_id=teacher_by_id,
                tokenizer=tokenizer,
                content_digest=file_digest(records_path),
            )
        )
        # Arrow cannot type the heterogeneous [str, float] topk pairs inside
        # kd_targets (WinError-free but ArrowInvalid "cannot mix list and
        # non-list"): the column travels as a JSON string and the data
        # collator decodes it.
        for row in rows:
            row["kd_targets"] = json.dumps(row["kd_targets"], default=list)
        return rows

    train_dataset = Dataset.from_list(_materialize(train_path))
    eval_dataset = (
        Dataset.from_list(_materialize(validation_path))
        if validation_path.exists() and any(True for _ in iter_jsonl(validation_path))
        else None
    )

    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype(plan)}
    if plan.quantization != "none":
        load_kwargs["quantization_config"] = _quantization_config(plan.quantization)
    model: Any = AutoModelForCausalLM.from_pretrained(config.base_model, **load_kwargs)
    if plan.peft:
        from peft import get_peft_model

        model = get_peft_model(model, _lora_config(config))

    class _KDTrainer(KDTrainer, Trainer):  # type: ignore[misc]  # mixin: compute_loss signature differs across transformers versions
        pass

    trainer = _KDTrainer(
        model=model,
        args=TrainingArguments(
            **{
                **_trainer_kwargs(config, output_dir, eval_dataset is not None, plan),
                # the KDDataCollator needs the kd_targets column: the default
                # remove_unused_columns=True strips it before the collator
                "remove_unused_columns": False,
            }
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=KDDataCollator(tokenizer),
        processing_class=tokenizer,
        kd_alpha=config.kd_alpha,
        kd_temperature=config.kd_temperature,
    )
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(output_dir))
    summary = {
        "base_model": config.base_model,
        "backend": "kd",
        "strategy": config.strategy,
        "quantization": plan.quantization,
        "device": plan.device,
        "device_kind": plan.device_kind,
        "dtype": plan.dtype,
        "attn_implementation": plan.attn_implementation,
        "effective_strategy": plan.effective_strategy,
        "kd_alpha": config.kd_alpha,
        "kd_temperature": config.kd_temperature,
        "kd_positions": sum(len(row["kd_targets"]) for row in train_dataset),
        "train_rows": len(train_dataset),
        "validation_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps({**summary, "log_history": trainer.state.log_history}, indent=2),
        encoding="utf-8",
    )
    return summary

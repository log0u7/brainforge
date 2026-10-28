import contextlib
import json
import math
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest


class _FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def to(self, device: object) -> "_FakeTensor":
        return self

    def clone(self) -> "_FakeTensor":
        return _FakeTensor([list(row) for row in self.rows])

    def __eq__(self, other: object) -> object:
        return _FakeTensor([[1 if value == other else 0 for value in row] for row in self.rows])

    def __setitem__(self, mask: "_FakeTensor", value: int) -> None:
        for row, flags in zip(self.rows, mask.rows, strict=True):
            for index, flag in enumerate(flags):
                if flag:
                    row[index] = value

    def __getitem__(self, key: tuple[slice, slice]) -> "_FakeTensor":
        _rows, cols = key
        return _FakeTensor([row[cols] for row in self.rows])

    def sum(self) -> int:
        return sum(sum(row) for row in self.rows)


class _FakeBaseModel:
    def __init__(self, losses: list[float]) -> None:
        self.losses = list(losses)
        self.device = "cpu"
        self.calls: list[dict] = []

    def eval(self) -> "_FakeBaseModel":
        return self

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return types.SimpleNamespace(loss=self.losses.pop(0))


class _FakeTokenizer:
    chat_template: str | None = "tpl"
    specs: ClassVar[list[dict]] = []
    instances: ClassVar[list["_FakeTokenizer"]] = []

    def __init__(self) -> None:
        self._applied = 0
        type(self).instances.append(self)

    @classmethod
    def from_pretrained(cls, name: str, **kwargs: object) -> "_FakeTokenizer":
        instance = cls()
        instance.name = name
        return instance

    def apply_chat_template(
        self,
        messages: object,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = True,
    ) -> dict:
        assert tokenize and return_dict and not add_generation_prompt
        spec = type(self).specs[self._applied]
        self._applied += 1
        return {
            "input_ids": list(spec["input_ids"]),
            "attention_mask": list(spec["attention_mask"]),
        }

    def pad(self, features: list[dict], padding: bool = True, return_tensors: str = "pt") -> dict:
        assert padding and return_tensors == "pt"
        return {
            "input_ids": _FakeTensor([list(f["input_ids"]) for f in features]),
            "attention_mask": _FakeTensor([list(f["attention_mask"]) for f in features]),
        }


class _FakeAutoModelForCausalLM:
    models: ClassVar[list[_FakeBaseModel]] = []
    load_calls: ClassVar[list[tuple[str, dict]]] = []
    losses: ClassVar[list[float]] = []

    @classmethod
    def from_pretrained(cls, name: str, **kwargs: object) -> _FakeBaseModel:
        model = _FakeBaseModel(cls.losses)
        cls.models.append(model)
        cls.load_calls.append((name, kwargs))
        return model


@pytest.fixture
def fake_inference_modules(monkeypatch):
    """Fake torch/transformers modules: evaluate_base runs without the extra."""
    _FakeTokenizer.specs = []
    _FakeTokenizer.instances = []
    _FakeAutoModelForCausalLM.models = []
    _FakeAutoModelForCausalLM.load_calls = []
    _FakeAutoModelForCausalLM.losses = []
    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = contextlib.nullcontext
    torch_mod.float32 = "float32"
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = _FakeTokenizer
    transformers_mod.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    transformers_mod.BitsAndBytesConfig = lambda **kwargs: {"quant": kwargs}
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    return _FakeAutoModelForCausalLM


def _eval_dataset(tmp_path: Path, count: int = 1) -> Path:
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(json.dumps({"messages": messages}) for _ in range(count)) + "\n",
        encoding="utf-8",
    )
    return path


def test_messages_dataset_tolerates_empty_validation_file(tmp_path):
    """prepare can legitimately emit a 0-record split (tiny datasets): an
    existing-but-empty validation.jsonl must behave like a missing file
    (no eval dataset), not raise 'empty dataset split'."""
    import pytest

    from brainforge.training.qlora import _messages_dataset

    empty = tmp_path / "validation.jsonl"
    empty.write_text("", encoding="utf-8")
    try:
        result = _messages_dataset(empty)
    except Exception as exc:
        pytest.fail(f"empty validation file raised: {exc}")
    assert result is None


def test_model_init_kwargs_pins_device_map():
    """The default auto dispatch sends quantized modules to the CPU on a
    busy 10GB card; the trainer kwargs pin the resolved device instead."""
    from types import SimpleNamespace

    from brainforge.training.qlora import _model_init_kwargs

    plan = SimpleNamespace(attn_implementation="sdpa", device="cuda:0")
    kwargs = _model_init_kwargs(plan)
    assert kwargs == {"attn_implementation": "sdpa", "device_map": "cuda:0"}


def test_evaluate_base_requires_chat_template(fake_inference_modules, monkeypatch, tmp_path):
    """Same tokenizer gate as evaluate(): missing chat template fails before
    any model load."""
    from brainforge.errors import BrainforgeError
    from brainforge.training.qlora import evaluate_base

    monkeypatch.setattr(_FakeTokenizer, "chat_template", None)
    with pytest.raises(BrainforgeError, match="no chat template"):
        evaluate_base("base/model", _eval_dataset(tmp_path))
    assert _FakeAutoModelForCausalLM.load_calls == []


def test_evaluate_base_result_shape(fake_inference_modules, tmp_path):
    """evaluate_base mirrors evaluate()'s result keys and 'none' plan values."""
    from brainforge.training.qlora import evaluate_base

    _FakeTokenizer.specs = [{"input_ids": [9, 8, 7, 3], "attention_mask": [1, 1, 1, 0]}]
    _FakeAutoModelForCausalLM.losses = [0.25]
    dataset = _eval_dataset(tmp_path)
    result = evaluate_base("base/model", dataset)
    assert set(result) == {
        "model",
        "dataset",
        "n_records",
        "eval_loss",
        "perplexity",
        "device",
        "device_kind",
        "dtype",
        "quantization",
        "attn_implementation",
    }
    assert result["model"] == "base/model"
    assert result["dataset"] == str(dataset)
    assert result["n_records"] == 1
    assert result["eval_loss"] == pytest.approx(0.25)
    assert result["perplexity"] == pytest.approx(math.exp(0.25))
    assert result["quantization"] == "none"
    assert result["dtype"] == "float32"
    assert result["device"] == "cpu"
    assert result["device_kind"] == "cpu"
    assert result["attn_implementation"] == "sdpa"


def test_evaluate_base_none_quantization_skips_quantization_config(
    fake_inference_modules, tmp_path
):
    """quantization='none' passes no quantization_config to from_pretrained."""
    from brainforge.training.qlora import evaluate_base

    _FakeTokenizer.specs = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]
    _FakeAutoModelForCausalLM.losses = [0.0]
    evaluate_base("base/model", _eval_dataset(tmp_path))
    name, kwargs = _FakeAutoModelForCausalLM.load_calls[0]
    assert name == "base/model"
    assert "quantization_config" not in kwargs
    assert kwargs["device_map"] == "auto"
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["torch_dtype"] == "float32"


def test_evaluate_base_token_weighted_loss(fake_inference_modules, tmp_path):
    """Corpus perplexity: 3 shifted tokens at loss 1.0 plus 7 at loss 2.0 give
    the token-weighted mean 1.7 (an unweighted mean would give 1.5)."""
    from brainforge.training.qlora import evaluate_base

    _FakeTokenizer.specs = [
        {"input_ids": [5, 6, 7, 8, 0, 0], "attention_mask": [1, 1, 1, 1, 0, 0]},
        {"input_ids": list(range(1, 9)), "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1]},
    ]
    _FakeAutoModelForCausalLM.losses = [1.0, 2.0]
    result = evaluate_base("base/model", _eval_dataset(tmp_path, count=2), batch_size=1)
    assert result["eval_loss"] == pytest.approx(1.7)
    assert result["perplexity"] == pytest.approx(math.exp(1.7))
    model = _FakeAutoModelForCausalLM.models[0]
    assert len(model.calls) == 2
    assert model.calls[0]["labels"].rows[0] == [5, 6, 7, 8, -100, -100]
    assert model.calls[1]["labels"].rows[0] == list(range(1, 9))

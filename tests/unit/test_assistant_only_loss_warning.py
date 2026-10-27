"""Warning for agentic multi-turn datasets trained without assistant-only loss.

The agentic pack emits records that interleave assistant tool calls with
`tool` observations. Training such a dataset with
`training.assistant_only_loss=false` puts tool observations inside the
loss; the train entry point warns instead of silently training on them.
"""

import json
from pathlib import Path

from brainforge.training.prepare import tool_turn_warning


def _record(*roles: str) -> dict:
    return {
        "id": "case-1",
        "domain": "agentic",
        "messages": [{"role": role, "content": "x"} for role in roles],
        "metadata": {},
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_warns_on_tool_turns_without_assistant_only_loss(tmp_path):
    path = _write(tmp_path, [_record("user", "assistant", "tool", "assistant")])
    warning = tool_turn_warning(path, assistant_only_loss=False)
    assert warning is not None
    assert "1" in warning
    assert "assistant_only_loss" in warning


def test_warns_on_multi_assistant_without_tool_role(tmp_path):
    path = _write(tmp_path, [_record("user", "assistant", "user", "assistant")])
    warning = tool_turn_warning(path, assistant_only_loss=False)
    assert warning is not None


def test_silent_when_assistant_only_loss_enabled(tmp_path):
    path = _write(tmp_path, [_record("user", "assistant", "tool", "assistant")])
    assert tool_turn_warning(path, assistant_only_loss=True) is None


def test_silent_on_single_turn_records(tmp_path):
    path = _write(tmp_path, [_record("user", "assistant"), _record("user", "assistant")])
    assert tool_turn_warning(path, assistant_only_loss=False) is None


def test_silent_on_missing_train_split(tmp_path):
    assert tool_turn_warning(tmp_path / "does-not-exist.jsonl", assistant_only_loss=False) is None


def test_counts_across_records(tmp_path):
    path = _write(
        tmp_path,
        [
            _record("user", "assistant", "tool", "assistant"),
            _record("user", "assistant", "user", "assistant"),
        ],
    )
    warning = tool_turn_warning(path, assistant_only_loss=False)
    assert warning is not None
    assert "2" in warning


def test_train_student_prints_warning_for_agentic_split(
    tmp_path, monkeypatch, capsys, auto_fit_stub
):
    import brainforge.cli.train_cmd as train_cmd

    path = _write(tmp_path, [_record("user", "assistant", "tool", "assistant")])

    class _Training:
        assistant_only_loss = False
        backend = "trl"

    captured = {}

    def fake_train_fn(training, dataset_dir, output_dir, resume=False):
        captured["dataset_dir"] = dataset_dir
        return {"output_dir": str(output_dir)}

    monkeypatch.setattr(
        "brainforge.training.prepare.tool_turn_warning",
        lambda dataset_path, assistant_only_loss: tool_turn_warning(path, assistant_only_loss),
    )
    monkeypatch.setattr("brainforge.training.qlora.train_qlora", fake_train_fn)
    summary = train_cmd.train_student(_Training(), tmp_path, tmp_path / "out")
    assert captured["dataset_dir"] == tmp_path
    output = capsys.readouterr().out
    assert "assistant_only_loss" in output
    assert summary["output_dir"].endswith("out")


def test_train_student_suppresses_warning_on_kd_path(tmp_path, monkeypatch, capsys, auto_fit_stub):
    import brainforge.cli.train_cmd as train_cmd

    path = _write(tmp_path, [_record("user", "assistant", "tool", "assistant")])

    class _Training:
        assistant_only_loss = False
        backend = "trl"

    monkeypatch.setattr(
        "brainforge.training.prepare.tool_turn_warning",
        lambda dataset_path, assistant_only_loss: tool_turn_warning(path, assistant_only_loss),
    )
    monkeypatch.setattr("brainforge.training.kd.train_kd", lambda *a, **k: {"output_dir": "x"})
    train_cmd.train_student(_Training(), tmp_path, tmp_path / "out", kd=True)
    assert "assistant_only_loss" not in capsys.readouterr().out

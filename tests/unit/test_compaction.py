"""Tool-observation compaction (maki-thread Phase 1): tool outputs are
59.3% of the mr22 corpus chars; ingest excerpts them instead of storing
raw dumps. Head + dropped-marker + tail, bounded and deterministic."""

from brainforge.domains.ctf.flag_ground_truth import (
    TOOL_OBS_HEAD,
    TOOL_OBS_MAX_CHARS,
    TOOL_OBS_TAIL,
    compact_tool_text,
    ingest_trajectories,
)


def test_short_text_unchanged():
    assert compact_tool_text("short") == "short"
    assert compact_tool_text("x" * TOOL_OBS_MAX_CHARS) == "x" * TOOL_OBS_MAX_CHARS


def test_long_text_bounded_with_marker():
    text = "HEAD" + "y" * 2000 + "TAIL"
    out = compact_tool_text(text)
    assert len(out) < 450
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert f"[... {len(text) - TOOL_OBS_HEAD - TOOL_OBS_TAIL} chars truncated ...]" in out


def test_single_long_line_bounded():
    out = compact_tool_text("z" * 5000)
    assert len(out) < TOOL_OBS_HEAD + TOOL_OBS_TAIL + 60
    assert f"{5000 - TOOL_OBS_HEAD - TOOL_OBS_TAIL} chars truncated" in out


def test_record_compacts_only_tool_messages():
    from brainforge.domains.ctf.flag_ground_truth import compact_tool_observations

    record = {
        "id": "r1",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "T" * 1000},
            {"role": "assistant", "content": "A" * 1000},
        ],
    }
    out = compact_tool_observations(record)
    assert len(out["messages"][1]["content"]) < 450
    assert out["messages"][2]["content"] == "A" * 1000
    assert out["messages"][0]["content"] == "q"


def test_ingest_applies_compaction():
    record = {
        "id": "case-1",
        "messages": [
            {"role": "user", "content": "solve"},
            {"role": "tool", "content": "O" * 1000},
            {"role": "assistant", "content": "{}"},
        ],
    }
    case = {"id": "case-1", "metadata": {"expected_flag": "flag{x}"}}
    rows, flags = ingest_trajectories([record], [case])
    assert len(rows[0]["messages"][1]["content"]) < 450
    assert flags == [{"id": "case-1", "expected_flag": "flag{x}"}]

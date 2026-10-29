"""Verified-tier records must carry the real flag string.

The pipeline verifies teacher flags against ``metadata.expected_flag``
but the stored record's assistant JSON keeps ``flag: null``: the ground
truth never reaches training targets or the holdout, so flag_accuracy
is uncomputable (measured 2026-10-29: None on every dataset).
"""

import json

import pytest


def make_record(record_id: str, verdict: str = "partial", flag=None) -> dict:
    assistant = {"verdict": verdict, "technique": "t", "exploit_path": [], "flag": flag}
    return {
        "id": record_id,
        "domain": "ctf",
        "messages": [
            {"role": "user", "content": "Solve this CTF challenge."},
            {"role": "assistant", "content": json.dumps(assistant)},
        ],
        "metadata": {"pipeline": "ctf"},
    }


def make_case(case_id: str, expected_flag: str | None, tier: str = "verified") -> dict:
    metadata = {"tier": tier}
    if expected_flag is not None:
        metadata["expected_flag"] = expected_flag
    return {"id": case_id, "metadata": metadata}


def test_enrich_stamps_verified_flag():
    from brainforge.dataset.flag_ground_truth import enrich_flags

    records = [make_record("ctf-lictor-fmtstr-001")]
    cases = [make_case("ctf-lictor-fmtstr-001", "FLAG{the_format_is_the_weapon}")]
    out = enrich_flags(records, cases)
    data = json.loads(out[0]["messages"][-1]["content"])
    assert data["flag"] == "FLAG{the_format_is_the_weapon}"
    assert out[0]["metadata"]["flag_verified"] is True


def test_enrich_skips_tier_b_and_mismatch():
    from brainforge.dataset.flag_ground_truth import enrich_flags

    tier_b = make_record("ctf-wqb-raw-001")
    cases = [make_case("ctf-wqb-raw-001", None, tier="unverified")]
    out = enrich_flags([tier_b], cases)
    assert json.loads(out[0]["messages"][-1]["content"])["flag"] is None
    assert "flag_verified" not in out[0]["metadata"]


def test_enrich_leaves_existing_flag_alone():
    from brainforge.dataset.flag_ground_truth import enrich_flags

    record = make_record("ctf-lictor-orw-001", flag="FLAG{orw_reads_the_flag}")
    cases = [make_case("ctf-lictor-orw-001", "FLAG{orw_reads_the_flag}")]
    out = enrich_flags([record], cases)
    assert out[0]["metadata"]["flag_verified"] is True


def test_enrich_rejects_conflicting_existing_flag():
    """A record that already claims a different flag is a data defect, not
    something enrichment silently overwrites."""
    from brainforge.dataset.flag_ground_truth import enrich_flags
    from brainforge.errors import BrainforgeError

    record = make_record("ctf-lictor-fmtstr-001", flag="FLAG{wrong}")
    cases = [make_case("ctf-lictor-fmtstr-001", "FLAG{the_format_is_the_weapon}")]
    with pytest.raises(BrainforgeError):
        enrich_flags([record], cases)

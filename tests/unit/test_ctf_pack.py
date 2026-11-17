import json

import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_pack, pack_names
from brainforge.domains.base import GeneralCritique
from brainforge.domains.ctf.schemas import CtfTeacherAnalysis, CtfVerdict
from brainforge.types import Mode, RoleKind


@pytest.fixture
def ctf_case():
    return Case(
        id="case-pwn-001",
        source=CaseSource(type="ctfd", repository="example-ctf", date="2026-08-15"),
        input=CaseInput(
            code="readelf -h ./chall\nProgram header: GNU_STACK RW",
            description="Pwn challenge: a 64-bit ELF with a gets() call and no canary.",
            question="Solve this CTF challenge.",
        ),
    )


def test_ctf_registered():
    assert "ctf" in pack_names()


def test_output_schemas():
    pack = get_pack("ctf")
    assert pack.output_schema(RoleKind.TEACHER) is CtfTeacherAnalysis
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is CtfVerdict


def test_recommended_steps():
    pack = get_pack("ctf")
    assert pack.steps_for_mode(Mode.CHEAP) == ["ctf_teacher", "judge"]
    assert pack.steps_for_mode(Mode.STANDARD) == [
        "ctf_teacher",
        "coding_teacher",
        "judge",
    ]
    assert pack.steps_for_mode(Mode.MAXIMUM) == [
        "ctf_teacher",
        "coding_teacher",
        "general_teacher",
        "judge",
    ]


def test_teacher_prompts_contain_case_and_rag(ctf_case):
    pack = get_pack("ctf")

    class Chunk:
        text = "gets() is unsafe: no bounds check"
        source = "man"

    teacher = pack.teacher_prompt("ctf_teacher", ctf_case, [Chunk()])
    assert "gets()" in teacher
    assert "no bounds check" in teacher
    judge = pack.judge_prompt(ctf_case, [], {"ctf_teacher": {"summary": "s"}})
    assert "canonical verdict" in judge


# Arbitration-by-ground-truth tests (moved from the engine, ADR-0022):
# disagreements must train, not discard. The pack arbitrates, the engine
# no longer knows the flag concept.


def _arb_case(expected_flag):
    return Case(
        id="case-arb-001",
        source=CaseSource(type="ctfd", repository="example-ctf", date="2026-08-15"),
        input=CaseInput(code="x", description="d", question="Solve."),
        metadata={"expected_flag": expected_flag} if expected_flag else {},
    )


def test_arbitrate_accepts_when_correct_flag_in_teacher_output():
    pack = get_pack("ctf")
    teacher = {"ctf_teacher": {"verdict": "confirmed", "flag": "WQB{s0lved}"}}
    judge = {"verdict": "insufficient_information"}
    assert pack.arbitrate(_arb_case("WQB{s0lved}"), judge, teacher) is True


def test_arbitrate_accepts_case_insensitive_match():
    pack = get_pack("ctf")
    teacher = {"ctf_teacher": {"flag": "wqb{s0lved}"}}
    judge = {"verdict": "insufficient_information"}
    assert pack.arbitrate(_arb_case("WQB{S0LVED}"), judge, teacher) is True


def test_arbitrate_rejects_when_no_flag_matches():
    pack = get_pack("ctf")
    teacher = {"ctf_teacher": {"flag": "WQB{wrong}"}}
    judge = {"verdict": "insufficient_information"}
    assert pack.arbitrate(_arb_case("WQB{right}"), judge, teacher) is False


def test_arbitrate_uses_judge_reasoning_too():
    """The judge contested but quoted the right flag in its reasoning."""
    pack = get_pack("ctf")
    judge = {"verdict": "insufficient_information", "reasoning": "candidate WQB{maybe}"}
    assert pack.arbitrate(_arb_case("WQB{maybe}"), judge, {}) is True


def test_arbitrate_no_ground_truth_returns_false():
    """Tier-B: nothing to verify against - no arbitration, gate decides."""
    pack = get_pack("ctf")
    teacher = {"ctf_teacher": {"flag": "WQB{anything}"}}
    judge = {"verdict": "insufficient_information"}
    assert pack.arbitrate(_arb_case(None), judge, teacher) is False


def test_teacher_gate_passes_valid():
    pack = get_pack("ctf")
    data = {
        "summary": "Classic ret2win via gets() overflow",
        "technique": "pwn/buffer-overflow",
        "difficulty": "easy",
        "exploit_path": ["overflow the return address", "jump to win()"],
        "confidence": 0.9,
        "reasoning": "gets() with a 64-byte buffer and no canary.",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data, min_confidence=0.5)
    assert result.passed


def test_teacher_gate_rejects_missing_technique():
    pack = get_pack("ctf")
    data = {
        "summary": "s",
        "technique": "",
        "difficulty": "easy",
        "exploit_path": ["x"],
        "confidence": 0.9,
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("technique" in reason for reason in result.reasons)


def test_teacher_gate_rejects_missing_exploit_path():
    pack = get_pack("ctf")
    data = {
        "summary": "s",
        "technique": "pwn/buffer-overflow",
        "difficulty": "easy",
        "exploit_path": [],
        "confidence": 0.9,
        "reasoning": "r",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("exploit_path" in reason for reason in result.reasons)


def test_judge_gate_passes_solved():
    pack = get_pack("ctf")
    data = {
        "verdict": "solved",
        "technique": "pwn/buffer-overflow",
        "exploit_path": ["overflow", "ret2win"],
        "flag": "flag{r3t2w1n}",
        "writeup": "Overflow the buffer and jump to win().",
        "confidence": 0.9,
        "reasoning": "The binary prints the flag from win().",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert result.passed


def test_judge_gate_rejects_solved_without_flag():
    pack = get_pack("ctf")
    data = {
        "verdict": "solved",
        "technique": "pwn/buffer-overflow",
        "exploit_path": ["overflow"],
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("flag" in reason for reason in result.reasons)


def test_judge_gate_rejects_solved_without_exploit_path():
    pack = get_pack("ctf")
    data = {
        "verdict": "solved",
        "technique": "pwn/buffer-overflow",
        "exploit_path": [],
        "flag": "flag{x}",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("exploit_path" in reason for reason in result.reasons)


def test_student_input(ctf_case):
    pack = get_pack("ctf")
    text = pack.student_input(ctf_case)
    assert "Solve this CTF challenge." in text
    assert "gets()" in text


def test_system_prompt_kinds():
    pack = get_pack("ctf")
    assert "CTF teacher" in pack.system_prompt("ctf_teacher", RoleKind.TEACHER)
    assert "judge" in pack.system_prompt("judge", RoleKind.JUDGE).lower()
    assert "devil" in pack.system_prompt("critic", RoleKind.CRITIC).lower()


def test_critic_prompt(ctf_case):
    pack = get_pack("ctf")
    text = pack.critic_prompt("critic", ctf_case, [], {})
    assert "gets()" in text


def test_quality_gate_low_confidence():
    pack = get_pack("ctf")
    result = pack.quality_gate(
        RoleKind.TEACHER,
        {"confidence": 0.1, "technique": "x", "exploit_path": [], "reasoning": "r"},
    )
    assert not result.passed
    assert any("confidence" in reason for reason in result.reasons)


def test_quality_gate_missing_summary_and_reasoning():
    pack = get_pack("ctf")
    result = pack.quality_gate(
        RoleKind.TEACHER, {"confidence": 0.9, "technique": "x", "exploit_path": []}
    )
    assert not result.passed
    assert any("missing summary or reasoning" in reason for reason in result.reasons)


def test_quality_gate_teacher_missing_reasoning_only():
    pack = get_pack("ctf")
    result = pack.quality_gate(
        RoleKind.TEACHER,
        {"confidence": 0.9, "technique": "x", "exploit_path": [], "summary": "s"},
    )
    assert not result.passed
    assert any("missing reasoning" in reason for reason in result.reasons)


def test_quality_gate_judge_missing_reasoning():
    pack = get_pack("ctf")
    result = pack.quality_gate(
        RoleKind.JUDGE, {"confidence": 0.9, "verdict": "unsolved"}
    )
    assert not result.passed
    assert any("missing reasoning" in reason for reason in result.reasons)


def _judge_payload(flag="FLAG{known}"):
    return {
        "verdict": "solved",
        "technique": "pwn/buffer-overflow",
        "exploit_path": ["overflow", "ret2win"],
        "flag": flag,
        "writeup": "Overflow the buffer and jump to win().",
        "confidence": 0.9,
        "reasoning": "The binary prints the flag from win().",
        "disagreement": False,
    }


def _judge_payload(
    flag="FLAG{known}", verdict="solved", technique="pwn/buffer-overflow ret2win"
):
    return {
        "verdict": verdict,
        "technique": technique,
        "exploit_path": ["overflow", "ret2win"],
        "flag": flag,
        "writeup": "Overflow the buffer and jump to win().",
        "confidence": 0.9,
        "reasoning": "The binary prints the flag from win().",
        "disagreement": False,
    }


def test_judge_gate_passes_matching_expected_flag(ctf_case):
    pack = get_pack("ctf")
    verified = ctf_case.model_copy(
        update={"metadata": {"expected_flag": "FLAG{known}"}}
    )
    result = pack.quality_gate(
        RoleKind.JUDGE, _judge_payload(), min_confidence=0.5, case=verified
    )
    assert result.passed


def test_judge_gate_accepts_technique_keyword_inside_prose(ctf_case):
    pack = get_pack("ctf")
    verified = ctf_case.model_copy(
        update={"metadata": {"expected_technique": ["ret2func", "ret2win"]}}
    )
    result = pack.quality_gate(
        RoleKind.JUDGE, _judge_payload(), min_confidence=0.5, case=verified
    )
    assert result.passed


def test_judge_gate_rejects_wrong_technique(ctf_case):
    pack = get_pack("ctf")
    verified = ctf_case.model_copy(
        update={"metadata": {"expected_technique": ["fmtstr", "format string"]}}
    )
    result = pack.quality_gate(
        RoleKind.JUDGE,
        _judge_payload(technique="pwn/heap overflow tcache"),
        min_confidence=0.5,
        case=verified,
    )
    assert not result.passed
    assert any("expected_technique" in reason for reason in result.reasons)


def test_judge_gate_rejects_hallucinated_flag(ctf_case):
    pack = get_pack("ctf")
    verified = ctf_case.model_copy(
        update={"metadata": {"expected_flag": "FLAG{known}"}}
    )
    result = pack.quality_gate(
        RoleKind.JUDGE,
        _judge_payload(flag="FLAG{invented}"),
        min_confidence=0.5,
        case=verified,
    )
    assert not result.passed
    assert any("expected_flag" in reason for reason in result.reasons)


def test_judge_gate_flag_absence_is_not_an_error(ctf_case):
    pack = get_pack("ctf")
    verified = ctf_case.model_copy(
        update={"metadata": {"expected_flag": "FLAG{known}"}}
    )
    result = pack.quality_gate(
        RoleKind.JUDGE, _judge_payload(flag=None), min_confidence=0.5, case=verified
    )
    assert result.passed


def test_judge_gate_tier_b_unaffected(ctf_case):
    pack = get_pack("ctf")
    result = pack.quality_gate(
        RoleKind.JUDGE, _judge_payload(), min_confidence=0.5, case=ctf_case
    )
    assert result.passed


def test_judge_gate_non_string_expected_metadata_does_not_crash(ctf_case):
    pack = get_pack("ctf")
    verified = ctf_case.model_copy(
        update={"metadata": {"expected_flag": 12345, "expected_technique": 99}}
    )
    result = pack.quality_gate(
        RoleKind.JUDGE, _judge_payload(), min_confidence=0.5, case=verified
    )
    assert result.passed


def test_judge_gate_flag_comparison_strips_whitespace(ctf_case):
    pack = get_pack("ctf")
    verified = ctf_case.model_copy(
        update={"metadata": {"expected_flag": "FLAG{known} "}}
    )
    result = pack.quality_gate(
        RoleKind.JUDGE,
        _judge_payload(flag=" FLAG{known}\n"),
        min_confidence=0.5,
        case=verified,
    )
    assert result.passed


def test_ctf_eval_shape_maps_verdict_and_flag():
    pack = get_pack("ctf")
    assert pack.eval_shape({"verdict": "solved", "flag": "FLAG{a}"}) == {
        "verdict": "solved",
        "abstain": False,
        "flag": "FLAG{a}",
    }
    assert pack.eval_shape({"verdict": "insufficient_information"}) == {
        "verdict": "insufficient_information",
        "abstain": True,
        "flag": None,
    }
    assert pack.eval_unparseable() == {"verdict": None, "abstain": False, "flag": None}


def test_ctf_eval_score_outcomes():
    pack = get_pack("ctf")
    solved = {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"}
    partial = {"verdict": "partial", "abstain": False, "flag": None}
    assert pack.eval_score(solved, solved)["outcome"] == "correct"
    # a partial solve does not pass as a full one: miss on a solved judge
    assert pack.eval_score(partial, solved)["outcome"] == "false_negative"
    assert pack.eval_score(solved, partial)["outcome"] == "false_positive"
    unparseable = pack.eval_score(
        {"verdict": None, "abstain": False, "flag": None}, solved
    )
    assert unparseable["predicted"] == "unparseable"
    assert unparseable["outcome"] == "false_negative"


def test_ctf_eval_secondary_metrics_flag_accuracy():
    pack = get_pack("ctf")
    pairs = [
        (
            {"verdict": "solved", "abstain": False, "flag": "FLAG{bad}"},
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
        ),
        (
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
            {"verdict": "solved", "abstain": False, "flag": "FLAG{a}"},
        ),
        (
            {"verdict": "solved", "abstain": False, "flag": None},
            {"verdict": "partial", "abstain": False, "flag": None},
        ),
    ]
    assert pack.eval_secondary_metrics(pairs) == {"flag_accuracy": 0.5}


def test_ctf_pack_declares_the_training_contract():
    """ADR-0022: the CTF pack owns its GRPO dataset contract - the reward
    name, the reward-server columns, the sidecar file and the gate metrics."""
    pack = get_pack("ctf")
    assert pack.grpo_reward() == "ctf_flag"
    assert pack.grpo_dataset_columns() == ("port", "flag", "facts")
    assert pack.gate_metrics() == ("flag_accuracy", "cwe_accuracy", "accuracy")
    assert pack.grpo_sidecar("/tmp/nowhere") == {}


def test_ctf_pack_sidecar_reads_expected_flags(tmp_path):
    pack = get_pack("ctf")
    sidecar = tmp_path / "grpo_flags.jsonl"
    sidecar.write_text(
        '{"id": "lab-1", "expected_flag": "FLAG{abc}"}\n'
        '{"id": "lab-2", "expected_flag": ""}\n'
        '{"id": "lab-3"}\n',
        encoding="utf-8",
    )
    assert pack.grpo_sidecar(tmp_path) == {"lab-1": {"expected_flag": "FLAG{abc}"}}


def test_ctf_pack_ingest_extracts_verdict_flags_and_merges_sidecar(tmp_path):
    pack = get_pack("ctf")
    records = [
        {
            "id": "lab-1",
            "messages": [
                {"role": "user", "content": "solve"},
                {
                    "role": "assistant",
                    "content": '{"verdict": "solved", "flag": "FLAG{read}"}',
                },
            ],
        },
        # a case carries the ground truth: the trajectory is skipped
        {"id": "lab-2", "messages": [{"role": "user", "content": "solve"}]},
    ]
    cases = [{"id": "lab-2", "metadata": {"expected_flag": "FLAG{cased}"}}]
    rows, sidecar_rows = pack.ingest_trajectories(records, cases)
    assert [row["id"] for row in rows] == ["lab-1", "lab-2"]
    assert sidecar_rows == [
        {"id": "lab-1", "expected_flag": "FLAG{read}"},
        {"id": "lab-2", "expected_flag": "FLAG{cased}"},
    ]
    # merge_sidecar is id-keyed and cumulative: a stale row is replaced
    dataset_dir = tmp_path / "grpo"
    dataset_dir.mkdir()
    (dataset_dir / "grpo_flags.jsonl").write_text(
        '{"id": "lab-1", "expected_flag": "FLAG{stale}"}\n', encoding="utf-8"
    )
    pack.merge_sidecar(dataset_dir, sidecar_rows)
    entries = [
        json.loads(line)
        for line in (dataset_dir / "grpo_flags.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert entries == [
        {"id": "lab-1", "expected_flag": "FLAG{read}"},
        {"id": "lab-2", "expected_flag": "FLAG{cased}"},
    ]

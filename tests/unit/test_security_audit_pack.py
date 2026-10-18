import pytest

from brainforge.case import Case, CaseInput, CaseSource
from brainforge.domains import get_pack, pack_names
from brainforge.domains.base import GeneralCritique
from brainforge.domains.security_audit.schemas import (
    AuditFinding,
    SecurityAuditAnalysis,
    SecurityAuditVerdict,
)
from brainforge.types import Mode, RoleKind


@pytest.fixture
def audit_case():
    return Case(
        id="auth-service-001",
        source=CaseSource(type="system", repository="auth-service", date="2026-07-10"),
        input=CaseInput(
            code=(
                '@app.post("/login")\ndef login(username, password):\n'
                "    if session_store.get(username) == password:\n"
                "        return create_token(username)\n"
                "    return None\n\n"
                "def create_token(username):\n"
                '    return jwt.encode({"user": username}, SECRET_KEY)\n'
            ),
            description=(
                "Authentication service: a Flask app that validates credentials by "
                "comparing the password to a value stored in an in-memory session "
                "store and issues a JWT without an expiry claim."
            ),
            question="Audit this system for security issues.",
        ),
    )


def test_security_audit_registered():
    assert "security_audit" in pack_names()


def test_output_schemas():
    pack = get_pack("security_audit")
    assert pack.output_schema(RoleKind.TEACHER) is SecurityAuditAnalysis
    assert pack.output_schema(RoleKind.CRITIC) is GeneralCritique
    assert pack.output_schema(RoleKind.JUDGE) is SecurityAuditVerdict


def test_recommended_steps():
    pack = get_pack("security_audit")
    assert pack.steps_for_mode(Mode.CHEAP) == ["audit_teacher", "judge"]
    assert pack.steps_for_mode(Mode.STANDARD) == ["audit_teacher", "security_teacher", "judge"]
    assert pack.steps_for_mode(Mode.MAXIMUM) == [
        "audit_teacher",
        "security_teacher",
        "coding_teacher",
        "judge",
    ]


def test_teacher_prompts_contain_case_and_rag(audit_case):
    pack = get_pack("security_audit")

    class Chunk:
        text = "Org policy: all JWTs must carry an expiry claim."
        source = "kb"

    teacher = pack.teacher_prompt("audit_teacher", audit_case, [Chunk()])
    assert "create_token" in teacher
    assert "Org policy" in teacher
    judge = pack.judge_prompt(audit_case, [], {"audit_teacher": {"summary": "s"}})
    assert "canonical security audit verdict" in judge


def test_teacher_gate_passes_valid():
    pack = get_pack("security_audit")
    data = {
        "summary": "Token lacks an expiry claim.",
        "findings": [
            {
                "area": "crypto",
                "severity": "high",
                "evidence": "jwt.encode without exp claim",
                "recommendation": "add an exp claim",
                "priority": "p1",
            }
        ],
        "scope_covered": ["auth"],
        "overall_risk": "high",
        "confidence": 0.85,
        "reasoning": "Unsigned expiry means tokens stay valid forever.",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data, min_confidence=0.5)
    assert result.passed


def test_teacher_gate_rejects_empty_reasoning():
    pack = get_pack("security_audit")
    data = {
        "summary": "s",
        "findings": [],
        "scope_covered": [],
        "overall_risk": "low",
        "confidence": 0.9,
        "reasoning": "",
    }
    result = pack.quality_gate(RoleKind.TEACHER, data)
    assert not result.passed
    assert any("reasoning" in reason for reason in result.reasons)


def test_judge_gate_passes_audit_complete():
    pack = get_pack("security_audit")
    data = {
        "verdict": "audit_complete",
        "findings": [
            {
                "area": "crypto",
                "severity": "high",
                "evidence": "jwt.encode without exp claim",
                "recommendation": "add an exp claim",
                "priority": "p1",
            }
        ],
        "overall_risk": "high",
        "remediation_priority": "p1",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data, min_confidence=0.5)
    assert result.passed


def test_judge_gate_rejects_audit_complete_without_findings():
    pack = get_pack("security_audit")
    data = {
        "verdict": "audit_complete",
        "findings": [],
        "overall_risk": "low",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("findings" in reason for reason in result.reasons)


def test_judge_gate_rejects_p0_with_low_priority():
    pack = get_pack("security_audit")
    data = {
        "verdict": "audit_complete",
        "findings": [
            {
                "area": "authz",
                "severity": "critical",
                "evidence": "IDOR on /admin/users",
                "recommendation": "enforce ownership checks",
                "priority": "p3",
            }
        ],
        "overall_risk": "critical",
        "remediation_priority": "p3",
        "confidence": 0.9,
        "reasoning": "r",
        "disagreement": False,
    }
    result = pack.quality_gate(RoleKind.JUDGE, data)
    assert not result.passed
    assert any("p0" in reason or "priority" in reason for reason in result.reasons)


def test_student_input(audit_case):
    pack = get_pack("security_audit")
    text = pack.student_input(audit_case)
    assert "Audit this system" in text
    assert "create_token" in text


def test_finding_schema():
    finding = AuditFinding(area="authn", severity="medium", evidence="no rate limiting")
    assert finding.recommendation is None
    assert finding.priority is None

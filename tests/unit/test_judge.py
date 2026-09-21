from brainforge.pipeline.judge import compute_agreement, is_unresolved, teacher_signal


def test_teacher_signal_vulnerability_found():
    assert teacher_signal({"vulnerability_found": True}) is True
    assert teacher_signal({"vulnerability_found": False}) is False


def test_teacher_signal_issues():
    assert teacher_signal({"issues": ["a"]}) is True
    assert teacher_signal({"issues": []}) is False


def test_teacher_signal_none():
    assert teacher_signal({}) is None
    assert teacher_signal({"other": 1}) is None


def test_is_unresolved():
    assert is_unresolved({"verdict": "insufficient_information"}) is True
    assert is_unresolved({"verdict": "confirmed"}) is False
    assert is_unresolved({}) is False


def test_compute_agreement_confirmed_and_agreeing_teacher():
    agreement = compute_agreement({"verdict": "confirmed"}, {"t1": {"vulnerability_found": True}})
    assert agreement == {"t1": True}


def test_compute_agreement_refuted_and_disagreeing_teacher():
    agreement = compute_agreement({"verdict": "refuted"}, {"t1": {"vulnerability_found": True}})
    assert agreement == {"t1": False}


def test_compute_agreement_skips_teachers_without_signal():
    agreement = compute_agreement(
        {"verdict": "confirmed"},
        {"t1": {"vulnerability_found": True}, "t2": {}, "t3": {"issues": []}},
    )
    assert agreement == {"t1": True, "t3": False}


def test_compute_agreement_empty():
    assert compute_agreement({"verdict": "confirmed"}, {}) == {}

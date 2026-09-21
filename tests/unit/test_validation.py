from brainforge.dataset.validation import validate_dataset, validate_record


def make_record(**overrides):
    record = {
        "id": "case-001",
        "domain": "security",
        "messages": [
            {"role": "user", "content": "review"},
            {"role": "assistant", "content": '{"verdict": "confirmed"}'},
        ],
        "metadata": {
            "source": {"type": "git", "date": "2026-08-01"},
            "teachers": [],
            "quality": {"passed": True},
            "recitation_risk": False,
        },
    }
    record.update(overrides)
    return record


def test_valid_record_passes():
    assert validate_record(make_record()) == []


def test_schema_validation_failure():
    errors = validate_record({"id": "x"})
    assert errors and "schema validation failed" in errors[0]


def test_no_messages():
    errors = validate_record(make_record(messages=[]))
    assert errors == ["record has no messages"]


def test_wrong_first_and_last_roles():
    record = make_record(
        messages=[
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": '{"verdict": "confirmed"}'},
        ]
    )
    errors = validate_record(record)
    assert "first message must be from the user" in errors
    assert "last message must be from the assistant" in errors


def test_unexpected_role_and_empty_content():
    record = make_record(
        messages=[
            {"role": "user", "content": "  "},
            {"role": "tool", "content": "data"},
            {"role": "assistant", "content": '{"ok": true}'},
        ]
    )
    errors = validate_record(record)
    assert any("unexpected message role 'tool'" in error for error in errors)
    assert any("empty content in 'user' message" in error for error in errors)


def test_assistant_content_not_json():
    record = make_record(
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "nope"}]
    )
    errors = validate_record(record)
    assert any("not valid JSON" in error for error in errors)


def test_missing_metadata_keys():
    record = make_record(metadata={"source": {}, "teachers": []})
    errors = validate_record(record)
    assert any("metadata is missing 'quality'" in error for error in errors)
    assert any("metadata is missing 'recitation_risk'" in error for error in errors)


def test_quality_gate_not_passed():
    record = make_record(metadata={**make_record()["metadata"], "quality": {"passed": False}})
    errors = validate_record(record)
    assert "metadata quality gate did not pass" in errors


def test_validate_dataset_splits_valid_and_invalid():
    result = validate_dataset([make_record(), {"id": "bad"}])
    assert len(result["valid"]) == 1
    assert result["invalid"][0]["index"] == 1
    assert result["invalid"][0]["id"] == "bad"

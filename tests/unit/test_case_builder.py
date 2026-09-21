import json
from pathlib import Path

from brainforge.dataset.case_builder import (
    build_case_from_dict,
    build_case_from_file,
    build_cases_from_path,
    case_id_for,
)


def write(tmp_path, name, content):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_case_id_deterministic():
    assert case_id_for("payload") == case_id_for("payload")
    assert case_id_for("a") != case_id_for("b")


def test_build_case_from_dict_explicit_id():
    case = build_case_from_dict({"id": "case-x", "input": {"code": "x()"}})
    assert case.id == "case-x"
    assert case.source.type == "manual"
    assert case.input.code == "x()"


def test_build_case_from_dict_generates_stable_id():
    data = {"input": {"code": "x()"}}
    case = build_case_from_dict(data)
    assert case.id.startswith("case-")
    assert case.id == build_case_from_dict(data).id


def test_build_case_from_file(tmp_path):
    path = write(tmp_path, "src/tool.py", "print('hi')\n")
    case = build_case_from_file(path, tmp_path)
    assert case is not None
    assert case.source.type == "file"
    assert case.source.path == str(Path("src/tool.py"))
    assert case.input.code == "print('hi')\n"
    assert case.metadata["language"] == "python"
    assert case.source.date is None


def test_build_case_from_file_empty_returns_none(tmp_path):
    path = write(tmp_path, "empty.py", "   \n")
    assert build_case_from_file(path, tmp_path) is None


def test_build_case_from_file_oversized_returns_none(tmp_path, monkeypatch):
    path = write(tmp_path, "big.py", "x = 1\n")
    monkeypatch.setattr("brainforge.dataset.case_builder._MAX_FILE_BYTES", 1)
    assert build_case_from_file(path, tmp_path) is None


def test_build_cases_from_path_mixed(tmp_path):
    write(tmp_path, "case-a.json", json.dumps({"id": "case-a", "input": {"code": "a"}}))
    write(tmp_path, "broken.json", "{not json")
    write(tmp_path, "noinput.json", json.dumps({"id": "case-b"}))
    write(tmp_path, "script.sh", "echo hi\n")
    write(tmp_path, "note.txt", "text only")
    write(tmp_path, ".hidden.py", "secret()")
    write(tmp_path, "sub/__pycache__/mod.py", "pass")
    cases = build_cases_from_path(tmp_path)
    assert "case-a" in [case.id for case in cases]
    assert any(case.metadata.get("language") == "shell" for case in cases)
    assert len(cases) == 2


def test_build_cases_from_path_single_file(tmp_path):
    path = write(tmp_path, "case-a.json", json.dumps({"id": "case-a", "input": {"code": "a"}}))
    cases = build_cases_from_path(path)
    assert [case.id for case in cases] == ["case-a"]

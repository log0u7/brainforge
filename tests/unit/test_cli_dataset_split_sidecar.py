import json
from pathlib import Path

from typer.testing import CliRunner

from brainforge.cli import app

runner = CliRunner()

# Groups picked so sha256("42:<group>") buckets them into the three splits with
# the default ratios (0.8/0.1, seed 42): train < 80, validation 80-89, test >= 90.
TRAIN_GROUP = "r-train-0"  # bucket 76
VAL_GROUP = "r-val-3"  # bucket 89
TEST_GROUP = "r-test-9"  # bucket 96


def _record(rid: str, repo: str) -> dict:
    return {
        "id": rid,
        "domain": "security",
        "messages": [
            {"role": "user", "content": f"question {rid}"},
            {"role": "assistant", "content": json.dumps({"answer": rid})},
        ],
        "metadata": {
            "source": {"type": "synthetic", "repository": repo},
            "teachers": ["teacher_a"],
            "quality": {"passed": True},
            "recitation_risk": True,
        },
    }


def _write_dataset(env: Path) -> None:
    records = [_record(f"train-{i}", TRAIN_GROUP) for i in range(6)]
    records += [_record(f"val-{i}", VAL_GROUP) for i in range(2)]
    records += [_record(f"test-{i}", TEST_GROUP) for i in range(2)]
    path = env / "datasets" / "ctf.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _kd_entry(rid: str) -> dict:
    return {"id": rid, "kd_targets": f"[{rid}-targets]"}


def _write_sidecar(env: Path, entries: list[dict]) -> Path:
    path = env / "datasets" / "ctf.kd.jsonl"
    # blank lines between entries must be skipped by the sidecar reader
    path.write_text("\n\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    return path


def _ids(path: Path) -> list[str]:
    return [
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_split_sidecar_writes_train_ids_in_sidecar_order(project_env):
    _write_dataset(project_env)
    sidecar = _write_sidecar(
        project_env,
        [
            _kd_entry("test-0"),
            _kd_entry("train-3"),
            _kd_entry("val-0"),
            _kd_entry("train-0"),
            _kd_entry("ghost-1"),
            _kd_entry("train-5"),
        ],
    )
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 0, result.output
    out = project_env / "datasets" / "ctf_split"
    assert _ids(out / "train.jsonl") == [f"train-{i}" for i in range(6)]
    assert _ids(out / "validation.jsonl") == ["val-0", "val-1"]
    assert _ids(out / "test.jsonl") == ["test-0", "test-1"]
    got = [
        json.loads(line)
        for line in (out / "kd.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert got == [
        _kd_entry("train-3"),
        _kd_entry("train-0"),
        _kd_entry("train-5"),
    ]
    assert "sidecar: 3/6" in result.output


def test_split_sidecar_default_output_path(project_env):
    _write_dataset(project_env)
    sidecar = _write_sidecar(project_env, [_kd_entry("train-0")])
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 0, result.output
    assert (project_env / "datasets" / "ctf_split" / "kd.jsonl").is_file()


def test_split_sidecar_explicit_output_path(project_env):
    _write_dataset(project_env)
    sidecar = _write_sidecar(project_env, [_kd_entry("train-0")])
    custom = project_env / "datasets" / "kd-custom.jsonl"
    result = runner.invoke(
        app,
        [
            "dataset",
            "split",
            "datasets/ctf.jsonl",
            "--sidecar",
            str(sidecar),
            "--sidecar-out",
            str(custom),
        ],
    )
    assert result.exit_code == 0, result.output
    assert custom.is_file()
    assert not (project_env / "datasets" / "ctf_split" / "kd.jsonl").exists()


def test_split_without_sidecar_outputs_unchanged(project_env):
    _write_dataset(project_env)
    out = project_env / "datasets" / "ctf_split"
    result = runner.invoke(app, ["dataset", "split", "datasets/ctf.jsonl"])
    assert result.exit_code == 0, result.output
    assert not (out / "kd.jsonl").exists()
    names = ["train.jsonl", "validation.jsonl", "test.jsonl", "test_postcutoff.jsonl", "stats.json"]
    before = {name: (out / name).read_bytes() for name in names}
    sidecar = _write_sidecar(project_env, [_kd_entry("train-0")])
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 0, result.output
    after = {name: (out / name).read_bytes() for name in names}
    assert after == before
    assert (out / "kd.jsonl").is_file()


def test_split_sidecar_malformed_line_fails(project_env):
    _write_dataset(project_env)
    sidecar = project_env / "datasets" / "ctf.kd.jsonl"
    sidecar.write_text('{"id": "train-0"}\nnot json\n', encoding="utf-8")
    out = project_env / "datasets" / "ctf_split"
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 1
    assert "sidecar" in result.output
    assert not (out / "kd.jsonl").exists()


def test_split_sidecar_non_dict_line_fails(project_env):
    _write_dataset(project_env)
    sidecar = project_env / "datasets" / "ctf.kd.jsonl"
    sidecar.write_text('{"id": "train-0"}\n[1, 2]\n', encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 1
    assert "sidecar" in result.output


def test_split_sidecar_entry_without_id_dropped(project_env):
    _write_dataset(project_env)
    sidecar = project_env / "datasets" / "ctf.kd.jsonl"
    sidecar.write_text('{"id": "train-0"}\n{"no_id": true}\n', encoding="utf-8")
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 0, result.output
    out = project_env / "datasets" / "ctf_split"
    lines = [
        json.loads(line)
        for line in (out / "kd.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines == [{"id": "train-0"}]


def test_split_sidecar_missing_file_fails(project_env):
    _write_dataset(project_env)
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", "datasets/ghost.kd.jsonl"]
    )
    assert result.exit_code == 1
    assert "sidecar not found" in result.output


def test_split_sidecar_non_string_id_dropped(project_env):
    _write_dataset(project_env)
    sidecar = _write_sidecar(project_env, [_kd_entry("train-0"), {"id": ["bad"]}])
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 0, result.output
    out = project_env / "datasets" / "ctf_split"
    got = [
        json.loads(line)
        for line in (out / "kd.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert got == [_kd_entry("train-0")]


def test_split_sidecar_empty_file_writes_empty_kd(project_env):
    _write_dataset(project_env)
    sidecar = _write_sidecar(project_env, [])
    result = runner.invoke(
        app, ["dataset", "split", "datasets/ctf.jsonl", "--sidecar", str(sidecar)]
    )
    assert result.exit_code == 0, result.output
    out = project_env / "datasets" / "ctf_split" / "kd.jsonl"
    assert out.is_file()
    assert out.read_text(encoding="utf-8").strip() == ""

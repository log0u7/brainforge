import json

from brainforge.config.models import Config
from brainforge.experiments.manager import (
    compute_metrics,
    create_experiment,
    high_agreement_warning,
)
from tests.conftest import raw_config


def sample_record(recitation_risk: bool, agreements: dict) -> dict:
    return {
        "id": "case-x",
        "domain": "security",
        "metadata": {"recitation_risk": recitation_risk, "teacher_agreement": agreements},
    }


def test_compute_metrics():
    records = [
        sample_record(True, {"security_teacher": True}),
        sample_record(False, {"security_teacher": True, "coding_teacher": False}),
        sample_record(False, {"security_teacher": False, "coding_teacher": True}),
    ]
    metrics = compute_metrics(records)
    assert metrics["total"] == 3
    assert metrics["recitation_risk"] == 1
    assert metrics["postcutoff"] == 2
    assert metrics["teacher_judge_agreement_rate"]["security_teacher"] == 0.667
    assert metrics["teacher_judge_agreement_rate"]["coding_teacher"] == 0.5


def test_high_agreement_warning():
    metrics = {"teacher_judge_agreement_rate": {"security_teacher": 0.98, "coding_teacher": 0.7}}
    warnings = high_agreement_warning(metrics)
    assert len(warnings) == 1
    assert "security_teacher" in warnings[0]


def test_create_experiment(tmp_path):
    config = Config.model_validate(raw_config())
    records = [sample_record(False, {"security_teacher": True})]
    run_dir = create_experiment("demo run", config, records=records, output_root=tmp_path)
    assert run_dir.is_dir()
    assert "demo-run" in run_dir.name
    saved_config = json.loads((run_dir / "config.json").read_text())
    assert saved_config["providers"]["teacher_a"]["type"] == "mock"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["total"] == 1
    report = (run_dir / "report.md").read_text()
    assert "Contamination note" in report
    assert "post-cutoff" in report

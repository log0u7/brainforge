import pytest

from brainforge.config.models import Config


def raw_config():
    return {
        "providers": {
            "teacher_a": {"type": "mock"},
            "teacher_b": {"type": "mock"},
            "local": {"type": "mock"},
            "judge_p": {"type": "mock"},
        },
        "models": {
            "security": {
                "provider": "teacher_a",
                "model": "mock-security",
                "family": "a",
                "knowledge_cutoff": "2026-06",
            },
            "coding": {
                "provider": "teacher_b",
                "model": "mock-coding",
                "family": "b",
                "knowledge_cutoff": "2026-05",
            },
            "general": {"provider": "local", "model": "mock-general", "family": "g"},
            "judge": {"provider": "judge_p", "model": "mock-judge", "family": "j"},
        },
        "roles": {
            "security_teacher": {"model": "security", "temperature": 0.2},
            "coding_teacher": {"model": "coding", "temperature": 0.3},
            "general_teacher": {"model": "general"},
            "judge": {"model": "judge", "temperature": 0.0},
        },
        "pipelines": {"security_dataset": {"domain": "security"}},
    }


@pytest.fixture
def config():
    return Config.model_validate(raw_config())

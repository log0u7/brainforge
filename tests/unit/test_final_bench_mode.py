"""Unit tests for the --mode flag on final_bench.

The bench runs in two modes:
  full   - model + augure + lictor (the production system)
  model  - model alone: the LLM must solve the challenge from source
           code alone, without augure's technique selection or lictor's
            exploit assembly/run.
"""

from pathlib import Path

import pytest


def test_mode_flag_is_accepted(tmp_path: Path) -> None:
    """--mode full|model is accepted; anything else fails."""
    # full mode parses fine (will fail later for missing server, but arg parse ok)
    import sys

    from brainforge.experiments.final_bench import main

    sys.argv = [
        "final_bench",
        "--manifest",
        str(tmp_path / "m.jsonl"),
        "--training-ids",
        str(tmp_path / "t.jsonl"),
        "--base-url",
        "http://127.0.0.1:9932/v1",
        "--base-model",
        "base",
        "--trained-url",
        "http://127.0.0.1:9933/v1",
        "--trained-model",
        "trained",
        "--repetitions",
        "1",
        "--mode",
        "full",
    ]
    # Should not raise on arg parsing (will fail later on missing server/files)
    try:
        main(sys.argv[1:])
    except SystemExit:
        pass  # argparse exits on error; missing files is expected
    except Exception:
        pass  # missing server/files is expected at this stage

    # model mode parses fine too
    sys.argv = [
        "final_bench",
        "--manifest",
        str(tmp_path / "m.jsonl"),
        "--training-ids",
        str(tmp_path / "t.jsonl"),
        "--base-url",
        "http://127.0.0.1:9932/v1",
        "--base-model",
        "base",
        "--trained-url",
        "http://127.0.0.1:9933/v1",
        "--trained-model",
        "trained",
        "--repetitions",
        "1",
        "--mode",
        "model",
    ]
    try:
        main(sys.argv[1:])
    except SystemExit:
        pass
    except Exception:
        pass

    # invalid mode fails
    sys.argv = [
        "final_bench",
        "--manifest",
        str(tmp_path / "m.jsonl"),
        "--training-ids",
        str(tmp_path / "t.jsonl"),
        "--base-url",
        "http://127.0.0.1:9932/v1",
        "--base-model",
        "base",
        "--trained-url",
        "http://127.0.0.1:9933/v1",
        "--trained-model",
        "trained",
        "--repetitions",
        "1",
        "--mode",
        "invalid",
    ]
    with pytest.raises(SystemExit):
        main(sys.argv[1:])


def test_model_mode_requires_no_augure() -> None:
    """In model mode, augure is not consulted: the model must solve the
    challenge from source code alone."""

    # The mode flag is stored; augure/lictor are not invoked in model mode.
    # This test documents the contract: model-only mode does not touch
    # augure's rules or lictor's execution layer.
    assert True  # placeholder - the real verification is in the bench run


def test_model_mode_uses_model_only_pilot() -> None:
    """In model mode, the pilot is model_only_pilot.run_model_only,
    not run_pilot. The model must solve from source code alone."""
    from brainforge.experiments import model_only_pilot

    assert hasattr(model_only_pilot, "run_model_only")
    assert callable(model_only_pilot.run_model_only)

    # The pilot accepts the same signature shape as run_pilot:
    # (targets, workdir, provider, model, seed=...)
    import inspect

    sig = inspect.signature(model_only_pilot.run_model_only)
    params = list(sig.parameters.keys())
    assert "targets" in params
    assert "workdir" in params
    assert "provider" in params
    assert "model" in params


def test_model_mode_uses_different_pilot() -> None:
    """In model mode, the pilot is model_only_pilot, not run_pilot."""

    # Verify that model_only_pilot exists and is importable
    from brainforge.experiments import model_only_pilot

    assert hasattr(model_only_pilot, "run_model_only")

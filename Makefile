.PHONY: install test lint format mypy secrets audit build ci mutate train-smoke train-smoke-cpu train-check train-run install-rocm install-xpu config models rag dataset validate train evaluate clean

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

secrets:
	@command -v gitleaks >/dev/null 2>&1 || { echo "gitleaks not found: install v8.30.1 from https://github.com/gitleaks/gitleaks/releases"; exit 1; }
	gitleaks git --redact -v .

audit:
	uv export --format requirements-txt --no-dev --no-hashes --no-emit-project -o requirements-audit.txt
	# PYSEC-2026-4217 (fsspec<=2026.2.0, transitive via torch/datasets http extra):
	# unfixable while the unsloth extra requires datasets<5.0 and datasets 4.8.x
	# caps fsspec at 2026.2.0. Re-check when unsloth supports datasets>=5.1.
	# --no-deps --disable-pip: audit the uv lock as-is; pip would reject the
	# pillow override against fastembed's <13 metadata.
	uvx pip-audit -r requirements-audit.txt --strict --ignore-vuln PYSEC-2026-4217 --no-deps --disable-pip

build:
	uv build

# Local equivalent of the CI pipeline (GitHub Actions / GitLab CI / Forgejo).
ci: lint secrets test config build audit

# GPU targets: need the training extra and an accelerator (CUDA/ROCm/XPU/MPS).
train-smoke:
	uv sync --extra training
	uv run python -m brainforge.training.smoke

# Preflight: hardware, model compatibility and memory estimate; CI-friendly.
train-check:
	uv sync --extra training
	uv run brainforge train check --strict

# Accelerator-specific torch builds replace the PyPI (CUDA) default wheel.
# Indexes verified for torch 2.13.0 + cp312 (2026-10).
ROCM_INDEX ?= https://download.pytorch.org/whl/rocm7.1
XPU_INDEX ?= https://download.pytorch.org/whl/xpu

install-rocm:
	uv sync --extra training
	uv pip install --reinstall torch==2.13.0 --index-url $(ROCM_INDEX)

install-xpu:
	uv sync --extra training
	uv pip install --reinstall torch==2.13.0 --index-url $(XPU_INDEX)

# CPU smoke: tiny model, no GPU/bitsandbytes needed; skips gracefully offline.
train-smoke-cpu:
	uv sync --extra training
	uv run python -m brainforge.training.smoke_cpu

train-run:
	uv run brainforge train run

config:
	uv run brainforge config validate
	uv run brainforge config schema --check

models:
	uv run brainforge models list

roles:
	uv run brainforge roles list

rag:
	uv run brainforge rag index data/raw

dataset:
	uv run brainforge pipeline run security_dataset --input data/raw

validate:
	uv run brainforge dataset validate datasets/security_dataset.jsonl

train:
	uv run brainforge train prepare datasets/security_dataset.jsonl

evaluate:
	uv run brainforge train evaluate

mypy:
	uv run mypy

# Mutation testing: run after touching core logic; triage surviving mutants.
mutate:
	uv run mutmut run

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

.PHONY: install test lint format secrets audit build ci config models rag dataset validate train evaluate clean

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
	uvx pip-audit -r requirements-audit.txt --strict

build:
	uv build

# Local equivalent of the CI pipeline (GitHub Actions / GitLab CI / Forgejo).
ci: lint secrets test config build audit

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

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

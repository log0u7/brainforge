.PHONY: install test lint format config models rag dataset validate train evaluate clean docs-serve docs-build

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

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

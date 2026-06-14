# Convenience targets. All commands run through uv (never pip).
# On Windows, run these via `make <target>` if GNU Make is installed,
# or copy the underlying `uv run ...` command directly.

.PHONY: help sync lint format test smoke check clean

help:  ## Show this help.
	@echo "Targets: sync lint format test smoke check clean"

sync:  ## Resolve and install all dependencies into the venv.
	uv sync

lint:  ## Lint the codebase with ruff.
	uv run ruff check .

format:  ## Auto-format the codebase with ruff.
	uv run ruff format .
	uv run ruff check . --fix

test:  ## Run the test suite.
	uv run pytest

smoke:  ## End-to-end SSL + FixMatch + eval on synthetic data (CPU).
	uv run btssl smoke

check: lint test  ## Lint + test (CI gate).

clean:  ## Remove caches and generated outputs.
	rm -rf .pytest_cache .ruff_cache .mypy_cache results

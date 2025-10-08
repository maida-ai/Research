# =========
# Defaults & env (override in .env or on CLI)
# =========
-include .env

UV_VENV_CLEAR       ?= 1
export VIRTUAL_ENV	:= $(PWD)/.venv
export UV_PROJECT_ENVIRONMENT	:= $(VIRTUAL_ENV)

SHELL 				:= /bin/bash
.SHELLFLAGS 		:= -eu -o pipefail -c


.PHONY: setup test lint format clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup    - Set up development environment"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linting checks"
	@echo "  format   - Format code"
	@echo "  clean    - Clean up generated files"
	@echo "  help     - Show this help message"

# Set up development environment
setup:
	@UV_VENV_CLEAR=$(UV_VENV_CLEAR) uv venv "$(VIRTUAL_ENV)"
	@uv sync --group dev --all-packages --frozen
	@uv run pre-commit install \
		--hook-type pre-commit \
		--hook-type pre-push
	@git config --local core.commentChar ';'

# Run tests
test:
	uv run pytest -q --cov

# Run linting checks
lint:
	uv run ruff check .
	uv run isort --check-only .

# Format code
format:
	uv run ruff check --fix .
	uv run isort .

# Clean up generated files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

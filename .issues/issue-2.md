Create repo skeleton and dev environment

Bootstrap a clean Python repo and GPU-ready environment to train and benchmark efficient long-context blocks.

## Description

We need a reproducible repo with a minimal Python package, tests, formatting/linting, and CUDA-ready deps (PyTorch + FlashAttention). This establishes a stable base for all subsequent work.

## Implementation Suggestions

* Create the structure:
  ```
  efficient-longctx/
    efficient_longctx/
      __init__.py
      blocks/
      training/
      evals/
      utils/
    scripts/
    tests/
    examples/
    docs/
    reports/
    pyproject.toml
    README.md
    LICENSE
    .gitignore
    .pre-commit-config.yaml
    Makefile
  ```

* Add a `Makefile`:
  ```
  setup:
    python -m venv .venv && . .venv/bin/activate && pip install -U pip
    . .venv/bin/activate && pip install -e ".[dev]"

  test:
    . .venv/bin/activate && pytest -q

  lint:
    . .venv/bin/activate && ruff check . && black --check . && isort --check-only .

  format:
    . .venv/bin/activate && ruff check --fix . && black . && isort .
  ```

* In `pyproject.toml`, configure Black, Ruff, isort, and project metadata.
* Create `.pre-commit-config.yaml` with hooks for Black/Ruff/isort and run:
  ```
  pip install pre-commit
  pre-commit install
  ```

* Install GPU deps (adjust CUDA version if needed):
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install flash-attn==2.*
  pip install numpy tqdm datasets transformers huggingface_hub accelerate einops triton matplotlib pytest pytest-cov ruff black isort
  ```

* Add tests:
  ```python
  # tests/test_cuda.py
  import torch

  def test_cuda_available():
      assert torch.cuda.is_available(), "CUDA GPU not detected"
  ```

  ```python
  # tests/test_flashattn_import.py
  def test_flash_attn_import():
      import flash_attn  # noqa: F401
  ```

## Acceptance Criteria

* [ ] `make setup` installs deps and `pytest -q` passes locally.
* [ ] FlashAttention imports successfully.
* [ ] Pre-commit hooks (Black/Ruff/isort) run on commit.

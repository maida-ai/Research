Create repo skeleton and dev environment

Fixes #2

Bootstrap a clean Python repo and GPU-ready environment to train and benchmark efficient long-context blocks.

**Changes:**
- Created complete Python package structure with efficient_longctx module
- Added pyproject.toml with project metadata, dependencies, and tool configurations
- Implemented Makefile with setup, test, lint, and format targets using uv
- Configured pre-commit hooks for Black, Ruff, and isort
- Added test suite for CUDA availability and FlashAttention validation
- Created comprehensive README.md with setup instructions
- Added .gitignore and LICENSE files

---

## Summary

This implementation creates a complete, reproducible development environment for efficient long-context research. The setup uses modern Python tooling (uv, pytest, ruff, black, isort) and includes GPU-ready dependencies (PyTorch + FlashAttention).

## Implementation Details

- **Package Structure**: Follows the exact structure specified in the issue with efficient_longctx/ containing blocks/, training/, evals/, and utils/ submodules
- **Dependencies**: Core ML dependencies (torch, transformers, datasets, etc.) with FlashAttention installed separately to handle build requirements
- **Development Tools**: Complete linting and formatting setup with pre-commit hooks
- **Testing**: CUDA availability tests and FlashAttention import validation with graceful error handling

## Test Coverage

The test suite includes:
- CUDA availability and device count validation
- FlashAttention import testing with error handling for version compatibility issues
- All tests pass locally with `make test`

## Modified Files

- `pyproject.toml` - Project configuration and dependencies
- `Makefile` - Development commands using uv
- `.pre-commit-config.yaml` - Git hooks configuration
- `efficient_longctx/__init__.py` - Main package initialization
- `efficient_longctx/{blocks,training,evals,utils}/__init__.py` - Submodule initialization
- `tests/test_cuda.py` - CUDA availability tests
- `tests/test_flashattn_import.py` - FlashAttention import tests
- `tests/__init__.py` - Test package initialization
- `README.md` - Project documentation
- `.gitignore` - Python project gitignore
- `LICENSE` - MIT license

## Risk Assessment

- **Low Risk**: All changes are additive and create a new development environment
- **FlashAttention Compatibility**: Tests handle version compatibility issues gracefully
- **Rollback**: Can be easily reverted by removing the new files

## Commands to Validate

```bash
make setup    # Set up development environment
make test     # Run test suite
make lint     # Run linting checks
make format   # Format code
```

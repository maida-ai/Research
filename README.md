# Efficient Long-Context Blocks

A Python package for training and benchmarking efficient long-context attention blocks.

## Overview

This repository provides a clean, reproducible environment for developing and testing efficient long-context attention mechanisms. It includes GPU-ready dependencies (PyTorch + FlashAttention) and a comprehensive development setup.

## Features

- **GPU-Ready**: Pre-configured with CUDA support and FlashAttention
- **Development Tools**: Black, Ruff, isort for code formatting and linting
- **Testing**: Pytest with coverage reporting
- **Pre-commit Hooks**: Automated code quality checks
- **Reproducible**: Locked dependencies and clear setup instructions

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maida-ai/research.git
   cd research
   ```

2. Set up the development environment:
   ```bash
   make setup
   ```

3. Verify installation:
   ```bash
   make test
   ```

## Development

### Available Commands

- `make setup` - Set up development environment
- `make test` - Run tests
- `make lint` - Run linting checks
- `make format` - Format code
- `make clean` - Clean up generated files

### Project Structure

```
efficient-longctx/
├── efficient_longctx/          # Main package
│   ├── blocks/                 # Attention blocks
│   ├── training/               # Training utilities
│   ├── evals/                  # Evaluation utilities
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
├── examples/                   # Example notebooks
├── docs/                       # Documentation
├── reports/                    # Analysis and benchmarks
└── scripts/                    # Utility scripts
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **isort**: Import sorting
- **Pre-commit**: Git hooks for automated checks

### Testing

Run the test suite:

```bash
make test
```

The test suite includes:
- CUDA availability checks
- FlashAttention import validation
- Basic package functionality

## Dependencies

### Core Dependencies

- `torch>=2.0.0` - PyTorch for deep learning
- `flash-attn>=2.0.0` - FlashAttention for efficient attention
- `numpy>=1.21.0` - Numerical computing
- `transformers>=4.20.0` - Hugging Face transformers
- `datasets>=2.0.0` - Dataset utilities
- `accelerate>=0.20.0` - Training acceleration
- `einops>=0.6.0` - Tensor operations
- `triton>=2.0.0` - GPU kernel compilation

### Development Dependencies

- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting
- `ruff>=0.1.0` - Fast Python linter
- `black>=23.0.0` - Code formatter
- `isort>=5.12.0` - Import sorter
- `pre-commit>=3.0.0` - Git hooks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make test && make lint`
5. Commit your changes
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FlashAttention](https://github.com/Dao-AILab/flash-attention) for efficient attention implementations
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Hugging Face](https://huggingface.co/) for transformer models and datasets

# Examples Directory

This directory contains example scripts and demonstrations for the efficient long-context research project.

## Available Examples

### 1. `quick_example.py`
A simple, quick demonstration of basic model usage:
- Direct model creation
- Factory function usage
- Forward pass examples
- Minimal code to get started

**Usage:**
```bash
uv run python examples/quick_example.py
```

### 2. `models_quick_example.py`
Another quick example showing model usage from the models subpackage:
- Direct model creation
- Factory function usage
- Forward pass examples
- Demonstrates models package functionality

**Usage:**
```bash
uv run python examples/models_quick_example.py
```

### 3. `model_usage_example.py`
Comprehensive examples showing all model functionality:
- Model creation with different configurations
- Factory function for different model sizes
- Forward pass with various sequence lengths
- Model configuration and serialization
- Checkpoint saving and loading

**Usage:**
```bash
uv run python examples/model_usage_example.py
```

### 4. `dpassm_demo.py`
Detailed demonstration of the DP-ASSM (Dual-Path Attention + State Space Model) block:
- Model architecture overview
- Computational complexity analysis
- Variable sequence length support
- Windowed attention visualization
- SSM state dynamics
- Performance comparisons

**Usage:**
```bash
uv run python examples/dpassm_demo.py
```

### 5. `lightning_training_example.py` ⚡ NEW
PyTorch Lightning-based training example:
- Modern training pipeline with Lightning
- Automatic mixed precision, checkpointing, and logging
- Easy configuration and CLI support
- TensorBoard integration
- Production-ready training setup

**Usage:**
```bash
uv run python examples/lightning_training_example.py --out_dir ./my_training_run
```

### 6. `lightning_cli_example.py` ⚡ NEW
LightningCLI demonstration showing advanced CLI features:
- Automatic argument parsing from LightningModule and DataModule
- Built-in support for configuration files (YAML/JSON)
- Automatic subcommands (fit, validate, test, predict)
- Environment variable support
- Built-in checkpoint loading
- Much cleaner and more maintainable code

**Usage:**
```bash
uv run python examples/lightning_cli_example.py
```

## Training Examples

### Lightning Training (Recommended)
The new Lightning-based trainer provides:
- **Automatic mixed precision** training
- **Built-in checkpointing** and model saving
- **TensorBoard logging** with rich metrics
- **Multi-GPU support** out of the box
- **Gradient accumulation** and clipping
- **Learning rate scheduling** with warmup
- **Validation monitoring** and early stopping

**Quick training example:**
```bash
# Train a small model for testing
uv run python examples/lightning_training_example.py \
    --num_params 150m \
    --block vanilla \
    --max_tokens 100000 \
    --batch_size 4 \
    --max_epochs 1 \
    --out_dir ./test_run
```

**Full training with DP-ASSM:**
```bash
# Train with DP-ASSM block using LightningCLI
uv run python efficient_longctx/training/train.py fit \
    --model.num_params=150m \
    --model.block=dpassm \
    --model.window_size=2048 \
    --model.ssm_state_dim=256 \
    --data.dataset_name=openwebtext \
    --data.max_tokens=1000000 \
    --data.batch_size=16 \
    --model.learning_rate=3e-4 \
    --trainer.max_epochs=1
```

### LightningCLI Training (Advanced)
The LightningCLI-based trainer provides even more advanced features:
- **Automatic argument parsing** from LightningModule and DataModule
- **Configuration files** support (YAML/JSON)
- **Subcommands** (fit, validate, test, predict)
- **Environment variables** support (`PL_TRAINER__MAX_EPOCHS=10`)
- **Built-in checkpoint loading** and resuming
- **Much cleaner code** (reduces main function from ~100 lines to ~10 lines)

**LightningCLI training examples:**
```bash
# Basic training with LightningCLI
uv run python efficient_longctx/training/train.py fit \
    --model.num_params=150m \
    --model.block=dpassm \
    --data.dataset_name=openwebtext \
    --data.max_tokens=100000 \
    --trainer.max_epochs=1

# Using configuration file
uv run python efficient_longctx/training/train.py fit --config config.yaml

# Validation only
uv run python efficient_longctx/training/train.py validate --ckpt_path checkpoint.ckpt

# Using environment variables
PL_TRAINER__MAX_EPOCHS=5 PL_MODEL__LEARNING_RATE=1e-3 \
uv run python efficient_longctx/training/train.py fit
```

## Getting Started

1. **Quick Start**: Run `quick_example.py` for a fast introduction
2. **Models Package**: Run `models_quick_example.py` to see models subpackage usage
3. **Comprehensive**: Run `model_usage_example.py` for detailed examples
4. **DP-ASSM Focus**: Run `dpassm_demo.py` to understand the DP-ASSM architecture
5. **Training**: Run `lightning_training_example.py` for modern training pipeline
6. **Advanced CLI**: Run `lightning_cli_example.py` to see LightningCLI features

## Requirements

All examples require the project dependencies to be installed:
```bash
uv sync
```

## Examples Purpose

These examples demonstrate:
- **Model Independence**: How models can be used without training dependencies
- **Architecture Flexibility**: Different attention block types (DP-ASSM, BLADE, vanilla)
- **Performance**: Computational complexity and efficiency benefits
- **Integration**: How to integrate models into different applications
- **Training**: Modern training pipelines with PyTorch Lightning
- **Experimentation**: Easy ways to experiment with different configurations

## Contributing

When adding new examples:
1. Follow the existing naming convention (`*_example.py` or `*_demo.py`)
2. Include comprehensive docstrings
3. Add usage instructions to this README
4. Test that examples work independently

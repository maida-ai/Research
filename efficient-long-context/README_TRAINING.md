# Training Guide for Efficient Long-Context Models

This guide provides comprehensive instructions for training all attention block variants (DP-ASSM, BLADE, Longformer, BigBird) with debugging documentation and troubleshooting tips.

## Table of Contents

- [Quick Start](#quick-start)
- [Model Variants](#model-variants)
- [Training Commands](#training-commands)
- [Configuration](#configuration)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Results Analysis](#results-analysis)

## Quick Start

### Prerequisites

```bash
# Ensure you have the latest code
git pull origin main

# Install dependencies
make setup

# Verify installation
make test
```

### Basic Training Example

```bash
# Train a 150M parameter DP-ASSM model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=dpassm \
    --model.window_size=128 \
    --model.ssm_state_dim=64 \
    --data.dataset_name=wikitext \
    --data.dataset_config=wikitext-2-raw-v1 \
    --data.seq_len=512 \
    --trainer.max_steps=10000 \
    --trainer.devices=1 \
    --trainer.enable_progress_bar=False \
    --trainer.max_epochs=2

# Train a Longformer baseline model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=baseline_longformer \
    --model.window_size=128 \
    --model.n_global_tokens=2 \
    --data.dataset_name=wikitext \
    --data.dataset_config=wikitext-2-raw-v1 \
    --data.seq_len=512 \
    --trainer.max_steps=10000 \
    --trainer.devices=1 \
    --trainer.enable_progress_bar=False \
    --trainer.max_epochs=2

# Train a BigBird baseline model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=baseline_bigbird \
    --model.window_size=128 \
    --model.n_random_tokens=4 \
    --model.n_global_tokens=2 \
    --data.dataset_name=wikitext \
    --data.dataset_config=wikitext-2-raw-v1 \
    --data.seq_len=512 \
    --trainer.max_steps=10000 \
    --trainer.devices=1 \
    --trainer.enable_progress_bar=False \
    --trainer.max_epochs=2
```

## Model Variants

### Available Attention Blocks

| Block Type | Description | Key Parameters |
|------------|-------------|----------------|
| `dpassm` | Dual-Path Attention + State Space Model | `window_size`, `ssm_state_dim` |
| `blade` | Block-Local Attention with Per-Block State | `chunk_size`, `state_dim` |
| `baseline_longformer` | Longformer-style local + global attention | `window_size`, `n_global_tokens` |
| `baseline_bigbird` | BigBird-style local + random + global attention | `window_size`, `n_random_tokens`, `n_global_tokens` |
| `vanilla` | Standard causal attention (baseline) | None |

### Model Sizes

| Size | Parameters | d_model | n_layers | n_heads |
|------|------------|---------|----------|---------|
| `150m` | ~150M | 768 | 12 | 12 |
| `250m` | ~250M | 1024 | 16 | 16 |
| `350m` | ~350M | 1280 | 20 | 20 |

## Training Commands

### 1. DP-ASSM Training

```bash
# Small model for testing
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=dpassm \
    --model.window_size=128 \
    --model.ssm_state_dim=64 \
    --data.max_seq_len=8192 \
    --trainer.max_steps=5000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Medium model for production
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=250m \
    --model.block=dpassm \
    --model.window_size=256 \
    --model.ssm_state_dim=128 \
    --data.max_seq_len=16384 \
    --trainer.max_steps=50000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Large model for research
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=350m \
    --model.block=dpassm \
    --model.window_size=512 \
    --model.ssm_state_dim=256 \
    --data.max_seq_len=32768 \
    --trainer.max_steps=100000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2
```

### 2. BLADE Training

```bash
# Small model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=blade \
    --model.chunk_size=128 \
    --model.state_dim=64 \
    --data.max_seq_len=8192 \
    --trainer.max_steps=5000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Medium model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=250m \
    --model.block=blade \
    --model.chunk_size=256 \
    --model.state_dim=128 \
    --data.max_seq_len=16384 \
    --trainer.max_steps=50000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Large model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=350m \
    --model.block=blade \
    --model.chunk_size=512 \
    --model.state_dim=256 \
    --data.max_seq_len=32768 \
    --trainer.max_steps=100000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2
```

### 3. Longformer Baseline Training

```bash
# Small model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=baseline_longformer \
    --model.window_size=128 \
    --model.n_global_tokens=2 \
    --data.max_seq_len=8192 \
    --trainer.max_steps=5000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Medium model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=250m \
    --model.block=baseline_longformer \
    --model.window_size=256 \
    --model.n_global_tokens=2 \
    --data.max_seq_len=16384 \
    --trainer.max_steps=50000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Large model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=350m \
    --model.block=baseline_longformer \
    --model.window_size=512 \
    --model.n_global_tokens=2 \
    --data.max_seq_len=32768 \
    --trainer.max_steps=100000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2
```

### 4. BigBird Baseline Training

```bash
# Small model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=baseline_bigbird \
    --model.window_size=128 \
    --model.n_random_tokens=4 \
    --model.n_global_tokens=2 \
    --data.max_seq_len=8192 \
    --trainer.max_steps=5000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Medium model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=250m \
    --model.block=baseline_bigbird \
    --model.window_size=256 \
    --model.n_random_tokens=8 \
    --model.n_global_tokens=2 \
    --data.max_seq_len=16384 \
    --trainer.max_steps=50000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2

# Large model
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=350m \
    --model.block=baseline_bigbird \
    --model.window_size=512 \
    --model.n_random_tokens=16 \
    --model.n_global_tokens=2 \
    --data.max_seq_len=32768 \
    --trainer.max_steps=100000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2
```

### 5. Vanilla Attention Baseline

```bash
# Small model (for comparison)
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=vanilla \
    --data.max_seq_len=8192 \
    --trainer.max_steps=5000 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.max_epochs=2
```

## Configuration

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--model.num_params` | Model size (150m, 250m, 350m) | 150m | 150m for testing, 250m+ for production |
| `--model.block` | Attention block type | dpassm | See model variants above |
| `--data.max_seq_len` | Maximum sequence length | 8192 | 8192-32768 depending on GPU memory |
| `--trainer.max_steps` | Training steps | 100000 | 5000-100000 depending on model size |
| `--trainer.devices` | Number of GPUs | 1 | 1 for single GPU, 2+ for multi-GPU |
| `--trainer.accelerator` | Accelerator type | gpu | gpu for CUDA, cpu for CPU-only |

### Block-Specific Parameters

#### DP-ASSM Parameters
- `--model.window_size`: Local attention window size (128, 256, 512)
- `--model.ssm_state_dim`: SSM state dimension (64, 128, 256)

#### BLADE Parameters
- `--model.chunk_size`: Chunk size for block-local attention (128, 256, 512)
- `--model.state_dim`: Per-block state dimension (64, 128, 256)

#### Longformer Parameters
- `--model.window_size`: Local attention window size (128, 256, 512)
- `--model.n_global_tokens`: Number of global tokens (2)

#### BigBird Parameters
- `--model.window_size`: Local attention window size (128, 256, 512)
- `--model.n_random_tokens`: Number of random tokens per position (4, 8, 16)
- `--model.n_global_tokens`: Number of global tokens (2)

## Monitoring and Debugging

### 1. Training Logs

Training logs are automatically saved to `logs/` directory. Monitor training progress:

```bash
# Watch training logs
tail -f logs/lightning_logs/version_*/events.out.tfevents.*

# Or use tensorboard
tensorboard --logdir logs/lightning_logs
```

### 2. Memory Monitoring

Monitor GPU memory usage during training:

```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# Monitor specific process
nvidia-smi -l 1
```

### 3. Training Validation

Run quick validation tests:

```bash
# Test model initialization
uv run python -c "
from efficient_longctx.models.models import create_model
model = create_model(vocab_size=1000, num_params='150m', block_type='dpassm', block_kwargs={'window_size': 128})
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# Test forward pass
uv run python -c "
import torch
from efficient_longctx.models.models import create_model
model = create_model(vocab_size=1000, num_params='150m', block_type='dpassm', block_kwargs={'window_size': 128})
x = torch.randint(0, 1000, (2, 64))
logits = model(x)
print(f'Output shape: {logits.shape}')
"
```

### 4. Synthetic Evaluation

Test models on synthetic tasks:

```bash
# Passkey retrieval task
uv run python -m efficient_longctx.evals.synthetic \
    --task passkey \
    --seq_len 8192 \
    --max_len 8192 \
    --block dpassm \
    --device cuda

# Copy/recall task
uv run python -m efficient_longctx.evals.synthetic \
    --task copy_recall \
    --seq_len 8192 \
    --max_len 8192 \
    --block blade \
    --device cuda

# Drift curve analysis
uv run python -m efficient_longctx.evals.synthetic \
    --task drift_curve \
    --seq_len 8192 \
    --max_len 32768 \
    --block baseline_longformer \
    --device cuda
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory` errors during training

**Solutions**:
```bash
# Reduce sequence length
--data.max_seq_len=4096  # Instead of 8192

# Reduce batch size (if configurable)
--data.batch_size=1

# Use gradient checkpointing
--trainer.strategy=deepspeed_stage_2

# Reduce model size
--model.num_params=150m  # Instead of 350m
```

#### 2. Training Instability

**Symptoms**: Loss spikes, NaN values, or training divergence

**Solutions**:
```bash
# Reduce learning rate
--model.learning_rate=1e-4  # Instead of 1e-3

# Add gradient clipping
--model.gradient_clip_val=1.0

# Use different optimizer
--model.optimizer=AdamW

# Reduce sequence length for stability
--data.max_seq_len=4096
```

#### 3. Slow Training

**Symptoms**: Low GPU utilization, slow throughput

**Solutions**:
```bash
# Increase batch size (if memory allows)
--data.batch_size=4

# Use mixed precision
--trainer.precision=16

# Use multiple GPUs
--trainer.devices=2

# Optimize data loading
--data.num_workers=4
```

#### 4. Import Errors

**Symptoms**: `ModuleNotFoundError` or import failures

**Solutions**:
```bash
# Reinstall dependencies
make clean
make setup

# Check Python path
uv run python -c "import efficient_longctx; print(efficient_longctx.__file__)"

# Verify installation
make test
```

#### 5. Dataset Loading Issues

**Symptoms**: `RuntimeError: Dataset scripts are no longer supported` or dataset loading failures

**Solutions**:
```bash
# Use wikitext with specific config (recommended)
--data.dataset_name=wikitext --data.dataset_config=wikitext-2-raw-v1

# Use other available datasets
--data.dataset_name=wikitext --data.dataset_config=wikitext-103-raw-v1
--data.dataset_name=wikitext --data.dataset_config=wikitext-2-v1
--data.dataset_name=wikitext --data.dataset_config=wikitext-103-v1

# Use synthetic data for testing
--data.dataset_name=synthetic

# Use local data
--data.dataset_name=local --data.dataset_config=path/to/data
```

#### 6. Parameter Recognition Issues

**Symptoms**: `unrecognized arguments` errors for model parameters

**Solutions**:
```bash
# Check available parameters
uv run python -m efficient_longctx.training.train fit --help | grep model

# Use correct parameter names:
# --model.num_params (not --model.d_model)
# --model.block (not --model.block_type)
# --model.window_size (not --model.window)
# --model.n_global_tokens (for baseline blocks)
# --model.n_random_tokens (for BigBird)
```

#### 7. Progress Bar Issues

**Symptoms**: `AssertionError` in RichProgressBar or progress bar display issues

**Solutions**:
```bash
# Disable progress bar
--trainer.enable_progress_bar=False

# Use simple progress bar
--trainer.callbacks=[]  # Remove RichProgressBar callback

# Set tokenizer parallelism
export TOKENIZERS_PARALLELISM=false
```

#### 8. CUDA Compatibility Issues

**Symptoms**: CUDA version mismatches or device errors

**Solutions**:
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Use CPU fallback
--trainer.accelerator=cpu
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Enable debug logging
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run with debug flags
uv run python -m efficient_longctx.training.train fit \
    --model.num_params=150m \
    --model.block=dpassm \
    --data.max_seq_len=1024 \
    --trainer.max_steps=100 \
    --trainer.devices=1 \
    --trainer.accelerator=gpu \
    --trainer.fast_dev_run=True
```

## Performance Optimization

### 1. Memory Optimization

```bash
# Use gradient checkpointing
--trainer.strategy=deepspeed_stage_2

# Use mixed precision
--trainer.precision=16

# Optimize attention implementation
export FLASH_ATTENTION_FORCE_BUILD=1
```

### 2. Speed Optimization

```bash
# Use multiple GPUs
--trainer.devices=2

# Optimize data loading
--data.num_workers=4
--data.persistent_workers=True

# Use compiled models (PyTorch 2.0+)
--model.compile=True
```

### 3. Batch Size Optimization

Find optimal batch size:

```bash
# Start with small batch size
--data.batch_size=1

# Gradually increase until OOM
--data.batch_size=2
--data.batch_size=4
--data.batch_size=8
```

## Results Analysis

### 1. Training Metrics

Monitor key metrics during training:

- **Loss**: Should decrease steadily
- **Learning Rate**: Should follow schedule
- **GPU Memory**: Should be stable
- **Throughput**: Tokens/second

### 2. Evaluation Metrics

Run evaluation after training:

```bash
# Evaluate on synthetic tasks
uv run python -m efficient_longctx.evals.synthetic \
    --task passkey \
    --seq_len 32768 \
    --max_len 32768 \
    --block dpassm \
    --device cuda \
    --pretrained_model=path/to/checkpoint

# Generate performance report
uv run python scripts/generate_report.py \
    --checkpoint_path=path/to/checkpoint \
    --output_dir=reports/
```

### 3. Comparison Analysis

Compare different models:

```bash
# Run all models on same task
for block in dpassm blade baseline_longformer baseline_bigbird vanilla; do
    uv run python -m efficient_longctx.evals.synthetic \
        --task passkey \
        --seq_len 16384 \
        --max_len 16384 \
        --block $block \
        --device cuda
done
```

## Best Practices

### 1. Training Strategy

1. **Start Small**: Begin with 150M models and short sequences
2. **Validate Early**: Run synthetic evaluations frequently
3. **Monitor Resources**: Watch GPU memory and utilization
4. **Save Checkpoints**: Enable automatic checkpointing
5. **Document Results**: Keep detailed logs of experiments

### 2. Hyperparameter Tuning

1. **Learning Rate**: Start with 1e-3, adjust based on stability
2. **Sequence Length**: Increase gradually (1024 → 8192 → 32768)
3. **Window Size**: Match to sequence length (128 for 8K, 256 for 16K)
4. **State Dimensions**: Scale with model size (64 for 150M, 128 for 250M)

### 3. Experiment Management

1. **Use Version Control**: Commit code before each experiment
2. **Tag Checkpoints**: Use descriptive names for model checkpoints
3. **Log Everything**: Save hyperparameters and results
4. **Compare Fairly**: Use same data splits and evaluation metrics

## Support

For additional help:

1. **Check Issues**: Look at existing GitHub issues
2. **Run Tests**: Ensure all tests pass before training
3. **Check Logs**: Review training logs for error messages
4. **Community**: Ask questions in project discussions

## Example Training Scripts

### Complete Training Pipeline

```bash
#!/bin/bash
# train_all_models.sh

MODELS=("150m" "250m" "350m")
BLOCKS=("dpassm" "blade" "baseline_longformer" "baseline_bigbird" "vanilla")
SEQ_LENS=(8192 16384 32768)

for model in "${MODELS[@]}"; do
    for block in "${BLOCKS[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            echo "Training $model $block with seq_len=$seq_len"

            uv run python -m efficient_longctx.training.train fit \
                --model.num_params=$model \
                --model.block=$block \
                --data.max_seq_len=$seq_len \
                --trainer.max_steps=10000 \
                --trainer.devices=1 \
                --trainer.accelerator=gpu \
                --trainer.default_root_dir=logs/$model-$block-$seq_len
        done
    done
done
```

### Evaluation Pipeline

```bash
#!/bin/bash
# evaluate_all_models.sh

CHECKPOINTS=("logs/*/checkpoints/*.ckpt")

for checkpoint in $CHECKPOINTS; do
    echo "Evaluating $checkpoint"

    uv run python -m efficient_longctx.evals.synthetic \
        --task passkey \
        --seq_len 32768 \
        --max_len 32768 \
        --checkpoint_path=$checkpoint \
        --device cuda
done
```

This comprehensive guide should help you successfully train and evaluate all attention block variants. Start with small models and short sequences, then gradually scale up as you become familiar with the training process.

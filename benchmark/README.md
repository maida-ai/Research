# Efficient Long-Context Benchmark

A comprehensive benchmarking framework for evaluating the performance of efficient long-context attention blocks.

## Package Structure

The benchmark package follows proper Python packaging conventions:

```
benchmark/
├── benchmark/              # Core package
│   ├── __init__.py        # Main exports
│   ├── base.py            # Core benchmarking classes
│   ├── metrics.py         # Metrics and measurement utilities
│   ├── utils.py           # Utility functions
│   ├── blocks.py          # Block-specific benchmarks
│   └── lightning_benchmark.py  # PyTorch Lightning integration
├── tests/                 # Unit tests
│   ├── __init__.py
│   └── test_framework.py
├── examples/              # Example scripts
│   ├── __init__.py
│   ├── visualize_results.py
│   └── package_example.py
├── main.py                # CLI runner
├── pyproject.toml         # Package configuration
└── README.md              # This file
```

## Installation

The benchmark package is part of the uv workspace. Install dependencies with:

```bash
uv sync
```

## Features

- **Comprehensive Benchmarking**: Test all attention blocks (BLADE, DPASSM, Longformer, BigBird) across different sequence lengths
- **PyTorch Optimizations**: Automatic integration with PyTorch benchmarking utilities (`torch.backends.cudnn.benchmark`, `torch.backends.cudnn.deterministic`)
- **PyTorch Lightning Integration**: Built-in support for benchmarking during training with `benchmark=True` flag
- **Memory Profiling**: Detailed memory usage tracking and efficiency metrics
- **Visualization**: Generate performance comparison plots and scaling analysis
- **Flexible Configuration**: Customizable benchmark parameters and device selection

## Quick Start

### Basic Benchmark

```bash
# Run comprehensive benchmark on all blocks
python benchmark/main.py comprehensive

# Quick benchmark with specific parameters
python benchmark/main.py quick --blocks BLADE DPASSM --seq-len 2048
```

### Advanced Usage

```bash
# Custom configuration
python benchmark/main.py comprehensive \
    --d-model 768 \
    --n-heads 12 \
    --sequence-lengths 1024 2048 4096 8192 16384 \
    --batch-size 2 \
    --benchmark-runs 20 \
    --results-dir ./my_results
```

### PyTorch Lightning Integration

```python
from benchmark.lightning_benchmark import create_lightning_trainer_with_benchmark, BenchmarkConfig

# Create benchmark config
config = BenchmarkConfig(
    d_model=512,
    n_heads=8,
    sequence_lengths=[1024, 2048, 4096]
)

# Create trainer with benchmarking
trainer = create_lightning_trainer_with_benchmark(
    benchmark_config=config,
    benchmark_frequency=100,  # Benchmark every 100 steps
    max_epochs=10
)
```

## CLI Usage

The benchmark package provides a unified CLI through `main.py` with three modes:

### Quick Mode
Rapid benchmarking for quick testing.

**Usage:**
```bash
python benchmark/main.py quick [OPTIONS]
```

**Key Options:**
- `--blocks`: Specific blocks to test (BLADE, DPASSM, Longformer, BigBird)
- `--seq-len`: Sequence length (default: 2048)
- `--batch-size`: Batch size (default: 1)
- `--device`: Device to use (auto, cuda, cpu)
- `--runs`: Number of benchmark runs (default: 5)

### Comprehensive Mode
Full benchmarking with complete configuration options.

**Usage:**
```bash
python benchmark/main.py comprehensive [OPTIONS]
```

**Key Options:**
- `--d-model`: Model dimension (default: 512)
- `--n-heads`: Number of attention heads (default: 8)
- `--sequence-lengths`: Sequence lengths to test (default: 1024 2048 4096 8192)
- `--batch-size`: Batch size (default: 1)
- `--device`: Device to use (auto, cuda, cpu)
- `--blocks`: Specific blocks to test (BLADE, DPASSM, Longformer, BigBird)
- `--lightning-mode`: Use Lightning-optimized settings
- `--output-file`: Custom output file name
- `--results-dir`: Results directory (default: ./reports/benchmark)

### Test Mode
Test the benchmarking framework.

**Usage:**
```bash
python benchmark/main.py test [OPTIONS]
```

**Options:**
- `--verbose`: Verbose output

## Configuration

### BenchmarkConfig

The main configuration class with the following parameters:

```python
@dataclass
class BenchmarkConfig:
    # Model configuration
    d_model: int = 512
    n_heads: int = 8
    batch_size: int = 1
    sequence_lengths: list[int] = [1024, 2048, 4096, 8192]

    # Benchmarking configuration
    warmup_runs: int = 3
    benchmark_runs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # PyTorch optimization flags
    torch_benchmark: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # Memory profiling
    profile_memory: bool = True

    # Output configuration
    verbose: bool = True
    save_results: bool = True
    results_dir: str = "./reports/benchmark"
```

## Metrics

The framework measures the following metrics:

### Timing Metrics
- **Forward Time**: Mean, std, min, max forward pass time
- **Throughput**: Tokens processed per second

### Memory Metrics
- **Peak Memory**: Maximum memory usage during forward pass
- **Allocated Memory**: Memory allocated by PyTorch
- **Reserved Memory**: Memory reserved by PyTorch
- **Memory Efficiency**: Memory usage per token

## PyTorch Lightning Integration

### BenchmarkCallback

Automatically benchmark your model during training:

```python
from benchmark.lightning_benchmark import BenchmarkCallback

callback = BenchmarkCallback(
    benchmark_config=config,
    benchmark_frequency=100  # Benchmark every 100 steps
)

trainer = Trainer(callbacks=[callback])
```

### LightningBenchmarkRunner

Benchmark Lightning models directly:

```python
from benchmark.lightning_benchmark import LightningBenchmarkRunner

runner = LightningBenchmarkRunner(config)
results = runner.benchmark_lightning_model(model, "MyModel")
```

## Results Format

Results are saved as JSON with the following structure:

```json
[
  {
    "block_name": "BLADE",
    "sequence_length": 2048,
    "batch_size": 1,
    "forward_time_mean": 0.045,
    "forward_time_std": 0.002,
    "forward_time_min": 0.043,
    "forward_time_max": 0.048,
    "peak_memory_mb": 125.3,
    "allocated_memory_mb": 98.7,
    "reserved_memory_mb": 102.4,
    "throughput_tokens_per_sec": 45511,
    "memory_efficiency_mb_per_token": 0.048,
    "config": {
      "d_model": 512,
      "n_heads": 8,
      "device": "cuda",
      "torch_benchmark": true,
      "cudnn_benchmark": true
    }
  }
]
```

## Visualization

The visualization script generates several plots:

1. **Forward Time Comparison**: Performance across sequence lengths
2. **Throughput Comparison**: Processing speed comparison
3. **Memory Usage**: Memory consumption analysis
4. **Memory Efficiency**: Memory efficiency per token
5. **Scaling Analysis**: Computational complexity analysis

## Examples

### Compare Specific Blocks

```bash
python benchmark/main.py comprehensive \
    --blocks BLADE DPASSM \
    --sequence-lengths 1024 2048 4096 \
    --output-file blade_vs_dpassm.json
```

### Memory-Intensive Benchmark

```bash
python benchmark/main.py comprehensive \
    --sequence-lengths 8192 16384 32768 \
    --batch-size 4 \
    --benchmark-runs 5
```

### CPU Benchmark

```bash
python benchmark/main.py comprehensive \
    --device cpu \
    --sequence-lengths 1024 2048 \
    --batch-size 2
```

### Lightning-Optimized Benchmark

```bash
python benchmark/main.py comprehensive \
    --lightning-mode \
    --batch-size 2 \
    --sequence-lengths 1024 2048 4096
```

## Dependencies

The benchmarking framework requires:

- PyTorch >= 2.8.0
- PyTorch Lightning >= 2.5.5
- Matplotlib (for visualization)
- NumPy
- psutil (for system info)

## Best Practices

1. **Warmup Runs**: Always use warmup runs to ensure consistent measurements
2. **Multiple Runs**: Use multiple benchmark runs for statistical significance
3. **Memory Profiling**: Enable memory profiling for comprehensive analysis
4. **Device Consistency**: Use the same device for fair comparisons
5. **PyTorch Optimizations**: Enable `torch_benchmark` and `cudnn_benchmark` for optimal performance

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size or sequence length
- Use gradient checkpointing
- Enable memory-efficient attention

### Inconsistent Results
- Increase warmup runs
- Use deterministic mode (`--cudnn-deterministic`)
- Ensure consistent device usage

### Slow Benchmarking
- Enable PyTorch optimizations (`--torch-benchmark`)
- Use fewer benchmark runs for quick testing
- Consider using `--lightning-mode` for training benchmarks

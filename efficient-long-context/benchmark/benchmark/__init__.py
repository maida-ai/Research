"""Benchmarking framework for efficient long-context attention blocks."""

from .base import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .blocks import (
    BlockBenchmarkRunner,
    compare_blocks_performance,
    create_custom_input_generators,
    create_standard_blocks,
    find_best_block_for_length,
    run_comprehensive_benchmark,
)
from .lightning_benchmark import (
    BenchmarkCallback,
    LightningBenchmarkRunner,
    benchmark_during_training,
    create_benchmark_config_for_lightning,
    create_lightning_trainer_with_benchmark,
)
from .metrics import BenchmarkMetrics, MemoryMetrics, TimingMetrics
from .utils import cleanup_benchmark_environment, setup_benchmark_environment

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "BenchmarkConfig",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BlockBenchmarkRunner",
    # Metrics
    "BenchmarkMetrics",
    "MemoryMetrics",
    "TimingMetrics",
    # Utilities
    "setup_benchmark_environment",
    "cleanup_benchmark_environment",
    # Block benchmarking
    "create_standard_blocks",
    "create_custom_input_generators",
    "run_comprehensive_benchmark",
    "compare_blocks_performance",
    "find_best_block_for_length",
    # Lightning integration
    "BenchmarkCallback",
    "LightningBenchmarkRunner",
    "create_lightning_trainer_with_benchmark",
    "benchmark_during_training",
    "create_benchmark_config_for_lightning",
]

"""Base benchmarking classes and utilities."""

import gc
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch
import torch.nn as nn


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""

    # Model configuration
    d_model: int = 512
    n_heads: int = 8
    batch_size: int = 1
    sequence_lengths: list[int] = field(
        default_factory=lambda: [1024, 2048, 4096, 8192]
    )

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
    memory_precision: int = 2  # decimal places for memory measurements

    # Output configuration
    verbose: bool = True
    save_results: bool = True
    results_dir: str = "./reports/benchmark"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            if self.verbose:
                print("CUDA not available, falling back to CPU")

        if self.device == "cuda":
            # Enable PyTorch optimizations for CUDA
            if self.torch_benchmark:
                torch.backends.cudnn.benchmark = self.cudnn_benchmark
                torch.backends.cudnn.deterministic = self.cudnn_deterministic


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    block_name: str
    sequence_length: int
    batch_size: int

    # Timing metrics
    forward_time_mean: float
    forward_time_std: float
    forward_time_min: float
    forward_time_max: float

    # Memory metrics
    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float

    # Additional metrics
    throughput_tokens_per_sec: float
    memory_efficiency_mb_per_token: float

    # Configuration used
    config: BenchmarkConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "block_name": self.block_name,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "forward_time_mean": self.forward_time_mean,
            "forward_time_std": self.forward_time_std,
            "forward_time_min": self.forward_time_min,
            "forward_time_max": self.forward_time_max,
            "peak_memory_mb": self.peak_memory_mb,
            "allocated_memory_mb": self.allocated_memory_mb,
            "reserved_memory_mb": self.reserved_memory_mb,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_efficiency_mb_per_token": self.memory_efficiency_mb_per_token,
            "config": {
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "device": self.config.device,
                "torch_benchmark": self.config.torch_benchmark,
                "cudnn_benchmark": self.config.cudnn_benchmark,
            },
        }


class BenchmarkRunner:
    """Main benchmarking runner for attention blocks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: list[BenchmarkResult] = []

    def benchmark_block(
        self, block: nn.Module, block_name: str, input_generator: Callable | None = None
    ) -> list[BenchmarkResult]:
        """Benchmark a single attention block across different sequence lengths."""

        if input_generator is None:
            input_generator = self._default_input_generator

        block_results = []

        for seq_len in self.config.sequence_lengths:
            if self.config.verbose:
                print(f"Benchmarking {block_name} with sequence length {seq_len}")

            result = self._benchmark_single_config(
                block, block_name, seq_len, input_generator
            )
            block_results.append(result)
            self.results.append(result)

        return block_results

    def _benchmark_single_config(
        self,
        block: nn.Module,
        block_name: str,
        sequence_length: int,
        input_generator: Callable,
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""

        # Setup environment
        self._setup_benchmark_environment()

        try:
            # Generate input data
            inputs = input_generator(
                sequence_length, self.config.batch_size, self.config.d_model
            )
            inputs = [inp.to(self.config.device) for inp in inputs]

            # Move block to device
            block = block.to(self.config.device)
            block.eval()

            # Warmup runs
            with torch.no_grad():
                for _ in range(self.config.warmup_runs):
                    _ = block(*inputs)

            # Clear cache and sync
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Benchmark runs
            forward_times = []
            memory_deltas = []

            for _ in range(self.config.benchmark_runs):
                # Memory measurement - reset peak memory for accurate measurement
                if self.config.profile_memory and self.config.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                # Time measurement
                start_time = time.perf_counter()

                with torch.no_grad():
                    _ = block(*inputs)

                if self.config.device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                forward_time = end_time - start_time
                forward_times.append(forward_time)

                # Memory measurement - capture peak memory usage during the run
                if self.config.profile_memory:
                    current_memory = self._get_memory_usage()
                    memory_deltas.append(current_memory)

            # Calculate statistics
            forward_times = torch.tensor(forward_times)
            forward_time_mean = float(forward_times.mean())
            forward_time_std = float(forward_times.std())
            forward_time_min = float(forward_times.min())
            forward_time_max = float(forward_times.max())

            # Memory statistics
            if memory_deltas:
                # Use the maximum memory usage across all runs
                peak_memory = max(metric["peak_memory_mb"] for metric in memory_deltas)
                allocated_memory = max(
                    metric["allocated_memory_mb"] for metric in memory_deltas
                )
                reserved_memory = max(
                    metric["reserved_memory_mb"] for metric in memory_deltas
                )
            else:
                # Fallback: get current memory usage if profiling is disabled
                current_memory = self._get_memory_usage()
                peak_memory = current_memory["peak_memory_mb"]
                allocated_memory = current_memory["allocated_memory_mb"]
                reserved_memory = current_memory["reserved_memory_mb"]

            # Calculate derived metrics
            total_tokens = sequence_length * self.config.batch_size
            throughput_tokens_per_sec = total_tokens / forward_time_mean
            # Memory efficiency should be based on actual memory used, not allocated
            memory_efficiency_mb_per_token = (
                allocated_memory / total_tokens if total_tokens > 0 else 0.0
            )

            return BenchmarkResult(
                block_name=block_name,
                sequence_length=sequence_length,
                batch_size=self.config.batch_size,
                forward_time_mean=forward_time_mean,
                forward_time_std=forward_time_std,
                forward_time_min=forward_time_min,
                forward_time_max=forward_time_max,
                peak_memory_mb=peak_memory,
                allocated_memory_mb=allocated_memory,
                reserved_memory_mb=reserved_memory,
                throughput_tokens_per_sec=throughput_tokens_per_sec,
                memory_efficiency_mb_per_token=memory_efficiency_mb_per_token,
                config=self.config,
            )

        finally:
            self._cleanup_benchmark_environment()

    def _default_input_generator(self, seq_len: int, batch_size: int, d_model: int):
        """Default input generator for attention blocks."""
        # Create tensor on CPU, it will be moved to device later
        x = torch.randn(batch_size, seq_len, d_model, device="cpu")
        return [x]

    def _setup_benchmark_environment(self):
        """Setup PyTorch benchmarking environment."""
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _cleanup_benchmark_environment(self):
        """Cleanup after benchmarking."""
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage."""
        if self.config.device == "cuda":
            return {
                "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                "allocated_memory_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_memory_mb": torch.cuda.memory_reserved() / 1024**2,
            }
        else:
            # CPU memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "peak_memory_mb": memory_info.rss / 1024**2,
                "allocated_memory_mb": memory_info.rss / 1024**2,
                "reserved_memory_mb": memory_info.vms / 1024**2,
            }

    def _get_memory_delta(
        self, before: dict[str, float], after: dict[str, float]
    ) -> dict[str, float]:
        """Calculate memory delta between two measurements."""
        # For peak memory, we want the actual peak during the run, not the delta
        # For allocated/reserved, we want the delta (actual memory used)
        return {
            "peak_memory_mb": after["peak_memory_mb"],  # Peak memory during the run
            "allocated_memory_mb": max(
                0.0, after["allocated_memory_mb"] - before["allocated_memory_mb"]
            ),
            "reserved_memory_mb": max(
                0.0, after["reserved_memory_mb"] - before["reserved_memory_mb"]
            ),
        }

    def save_results(self, filename: str | None = None):
        """Save benchmark results to file."""
        import json
        import os

        if not self.config.save_results:
            return

        os.makedirs(self.config.results_dir, exist_ok=True)

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        filepath = os.path.join(self.config.results_dir, filename)

        results_data = [result.to_dict() for result in self.results]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        if self.config.verbose:
            print(f"Results saved to {filepath}")

    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group results by block name
        block_groups = {}
        for result in self.results:
            if result.block_name not in block_groups:
                block_groups[result.block_name] = []
            block_groups[result.block_name].append(result)

        for block_name, results in block_groups.items():
            print(f"\n{block_name}:")
            print("-" * len(block_name))

            for result in sorted(results, key=lambda x: x.sequence_length):
                print(
                    f"  Seq Len {result.sequence_length:4d}: "
                    f"{result.forward_time_mean * 1000:6.2f}ms ± {result.forward_time_std * 1000:5.2f}ms, "
                    f"{result.throughput_tokens_per_sec:8.0f} tokens/s, "
                    f"{result.allocated_memory_mb:6.1f}MB"
                )

        print("=" * 80)

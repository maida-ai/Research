"""Utility functions for benchmarking."""

import gc
from contextlib import contextmanager
from typing import Any

import psutil
import torch


def setup_benchmark_environment(
    device: str = "cuda",
    torch_benchmark: bool = True,
    cudnn_benchmark: bool = True,
    cudnn_deterministic: bool = False,
) -> dict[str, Any]:
    """Setup PyTorch benchmarking environment with optimizations."""

    original_settings = {}

    if device == "cuda" and torch.cuda.is_available():
        # Store original settings
        original_settings["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        original_settings["cudnn_deterministic"] = torch.backends.cudnn.deterministic

        # Apply optimizations
        if torch_benchmark:
            torch.backends.cudnn.benchmark = cudnn_benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Enable PyTorch optimizations
    if torch_benchmark:
        torch.backends.opt_einsum.enabled = True

    return original_settings


def cleanup_benchmark_environment(original_settings: dict[str, Any] | None = None):
    """Restore original PyTorch settings and cleanup."""

    if original_settings:
        if "cudnn_benchmark" in original_settings:
            torch.backends.cudnn.benchmark = original_settings["cudnn_benchmark"]
        if "cudnn_deterministic" in original_settings:
            torch.backends.cudnn.deterministic = original_settings[
                "cudnn_deterministic"
            ]

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def benchmark_context(device: str = "cuda", **kwargs):
    """Context manager for benchmarking setup/cleanup."""
    original_settings = setup_benchmark_environment(device, **kwargs)
    try:
        yield original_settings
    finally:
        cleanup_benchmark_environment(original_settings)


def get_system_info() -> dict[str, Any]:
    """Get system information for benchmarking context."""
    info = {
        "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_total_gb": torch.cuda.get_device_properties(0).total_memory
                / (1024**3),
            }
        )

    return info


def create_input_data(
    sequence_length: int,
    batch_size: int,
    d_model: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create standardized input data for benchmarking."""
    return torch.randn(batch_size, sequence_length, d_model, device=device, dtype=dtype)


def create_attention_mask(
    sequence_length: int,
    batch_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Create attention mask for benchmarking."""
    return torch.ones(batch_size, sequence_length, device=device, dtype=dtype)


def warmup_model(model: torch.nn.Module, inputs: torch.Tensor, warmup_runs: int = 3):
    """Warmup a model with given inputs."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_memory_usage(func, *args, **kwargs) -> dict[str, float]:
    """Measure memory usage of a function call."""
    if not torch.cuda.is_available():
        return {"peak_mb": 0.0, "allocated_mb": 0.0, "reserved_mb": 0.0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()

    func(*args, **kwargs)

    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated()
    end_allocated = torch.cuda.memory_allocated()
    end_reserved = torch.cuda.memory_reserved()

    return {
        "peak_mb": peak_memory / (1024**2),
        "allocated_mb": (end_allocated - start_allocated) / (1024**2),
        "reserved_mb": (end_reserved - start_reserved) / (1024**2),
    }


def benchmark_function(
    func,
    *args,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    device: str = "cuda",
    **kwargs,
) -> dict[str, float]:
    """Benchmark a function with timing and memory measurements."""

    # Warmup
    for _ in range(warmup_runs):
        func(*args, **kwargs)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark timing
    import time

    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Calculate statistics
    times_tensor = torch.tensor(times)

    return {
        "mean_time": float(times_tensor.mean()),
        "std_time": float(times_tensor.std()),
        "min_time": float(times_tensor.min()),
        "max_time": float(times_tensor.max()),
    }

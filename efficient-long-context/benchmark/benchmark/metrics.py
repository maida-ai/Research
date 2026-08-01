"""Metrics and measurement utilities for benchmarking."""

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TimingMetrics:
    """Timing-related metrics."""

    forward_time_mean: float
    forward_time_std: float
    forward_time_min: float
    forward_time_max: float
    throughput_tokens_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward_time_mean": self.forward_time_mean,
            "forward_time_std": self.forward_time_std,
            "forward_time_min": self.forward_time_min,
            "forward_time_max": self.forward_time_max,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
        }


@dataclass
class MemoryMetrics:
    """Memory-related metrics."""

    peak_memory_mb: float
    allocated_memory_mb: float
    reserved_memory_mb: float
    memory_efficiency_mb_per_token: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "allocated_memory_mb": self.allocated_memory_mb,
            "reserved_memory_mb": self.reserved_memory_mb,
            "memory_efficiency_mb_per_token": self.memory_efficiency_mb_per_token,
        }


@dataclass
class BenchmarkMetrics:
    """Combined benchmark metrics."""

    timing: TimingMetrics
    memory: MemoryMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "timing": self.timing.to_dict(),
            "memory": self.memory.to_dict(),
        }


class TimingProfiler:
    """Context manager for timing operations."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        if self.device == "cuda":
            torch.cuda.synchronize()

    @property
    def elapsed_time(self) -> float:
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Timing context not properly closed")
        return self.end_time - self.start_time


class MemoryProfiler:
    """Context manager for memory profiling."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_memory = None
        self.end_memory = None

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
            }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            self.end_memory = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak": torch.cuda.max_memory_allocated(),
            }

    def get_memory_delta(self) -> dict[str, float]:
        """Get memory usage delta in MB."""
        if (
            self.device != "cuda"
            or self.start_memory is None
            or self.end_memory is None
        ):
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}

        return {
            "allocated_mb": (
                self.end_memory["allocated"] - self.start_memory["allocated"]
            )
            / 1024**2,
            "reserved_mb": (self.end_memory["reserved"] - self.start_memory["reserved"])
            / 1024**2,
            "peak_mb": self.end_memory["peak"] / 1024**2,
        }


def calculate_statistics(values: list[float]) -> dict[str, float]:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    tensor_values = torch.tensor(values)
    return {
        "mean": float(tensor_values.mean()),
        "std": float(tensor_values.std()),
        "min": float(tensor_values.min()),
        "max": float(tensor_values.max()),
    }


def format_time(seconds: float) -> str:
    """Format time in a human-readable way."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    else:
        return f"{seconds:.2f}s"


def format_memory(mb: float) -> str:
    """Format memory in a human-readable way."""
    if mb < 1:
        return f"{mb * 1024:.1f}KB"
    elif mb < 1024:
        return f"{mb:.1f}MB"
    else:
        return f"{mb / 1024:.1f}GB"


def format_throughput(tokens_per_sec: float) -> str:
    """Format throughput in a human-readable way."""
    if tokens_per_sec < 1000:
        return f"{tokens_per_sec:.0f} tokens/s"
    elif tokens_per_sec < 1000000:
        return f"{tokens_per_sec / 1000:.1f}K tokens/s"
    else:
        return f"{tokens_per_sec / 1000000:.1f}M tokens/s"

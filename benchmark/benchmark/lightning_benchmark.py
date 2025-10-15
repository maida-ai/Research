"""PyTorch Lightning integration for benchmarking."""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from .base import BenchmarkConfig, BenchmarkResult


class BenchmarkCallback(Callback):
    """PyTorch Lightning callback for benchmarking during training."""

    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
        benchmark_frequency: int = 100,  # Benchmark every N steps
        save_results: bool = True,
    ):
        super().__init__()
        self.benchmark_config = benchmark_config
        self.benchmark_frequency = benchmark_frequency
        self.save_results = save_results
        self.benchmark_results: list[BenchmarkResult] = []

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Run benchmark at specified intervals during training."""

        if batch_idx % self.benchmark_frequency == 0:
            self._run_benchmark(trainer, pl_module)

    def _run_benchmark(self, trainer: Trainer, pl_module: pl.LightningModule):
        """Run benchmark on the current model."""

        # Get the attention block from the model
        attention_block = self._extract_attention_block(pl_module)
        if attention_block is None:
            return

        # Create benchmark runner
        from benchmark.base import BenchmarkRunner

        runner = BenchmarkRunner(self.benchmark_config)

        # Run benchmark
        results = runner.benchmark_block(
            attention_block,
            f"{pl_module.__class__.__name__}_step_{trainer.global_step}",
        )

        self.benchmark_results.extend(results)

        if self.save_results:
            runner.save_results(f"training_benchmark_step_{trainer.global_step}.json")

    def _extract_attention_block(
        self, pl_module: pl.LightningModule
    ) -> torch.nn.Module | None:
        """Extract attention block from Lightning module."""

        # Try common attribute names
        for attr_name in ["attention_block", "attention", "block", "model"]:
            if hasattr(pl_module, attr_name):
                attr = getattr(pl_module, attr_name)
                if isinstance(attr, torch.nn.Module):
                    return attr

        # Try to find in named modules
        for name, module in pl_module.named_modules():
            if any(
                block_type in name.lower()
                for block_type in ["blade", "dpassm", "longformer", "bigbird"]
            ):
                return module

        return None


class LightningBenchmarkRunner:
    """Benchmark runner specifically for PyTorch Lightning models."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def benchmark_lightning_model(
        self,
        model: pl.LightningModule,
        model_name: str,
        input_generator: Callable | None = None,
    ) -> list[BenchmarkResult]:
        """Benchmark a PyTorch Lightning model."""

        # Extract the attention block
        attention_block = self._extract_attention_block(model)
        if attention_block is None:
            raise ValueError("Could not find attention block in Lightning module")

        # Create benchmark runner
        from benchmark.base import BenchmarkRunner

        runner = BenchmarkRunner(self.config)

        # Run benchmark
        results = runner.benchmark_block(attention_block, model_name, input_generator)

        return results

    def _extract_attention_block(
        self, model: pl.LightningModule
    ) -> torch.nn.Module | None:
        """Extract attention block from Lightning module."""

        # Try common attribute names
        for attr_name in ["attention_block", "attention", "block", "model"]:
            if hasattr(model, attr_name):
                attr = getattr(model, attr_name)
                if isinstance(attr, torch.nn.Module):
                    return attr

        # Try to find in named modules
        for name, module in model.named_modules():
            if any(
                block_type in name.lower()
                for block_type in ["blade", "dpassm", "longformer", "bigbird"]
            ):
                return module

        return None


def create_lightning_trainer_with_benchmark(
    benchmark_config: BenchmarkConfig, benchmark_frequency: int = 100, **trainer_kwargs
) -> Trainer:
    """Create a PyTorch Lightning trainer with benchmarking enabled."""

    # Enable benchmarking optimizations
    trainer_kwargs.setdefault("benchmark", True)

    # Create benchmark callback
    benchmark_callback = BenchmarkCallback(
        benchmark_config=benchmark_config, benchmark_frequency=benchmark_frequency
    )

    # Add callback to trainer kwargs
    callbacks = trainer_kwargs.get("callbacks", [])
    callbacks.append(benchmark_callback)
    trainer_kwargs["callbacks"] = callbacks

    # Create trainer
    trainer = Trainer(**trainer_kwargs)

    return trainer


def benchmark_during_training(
    model: pl.LightningModule,
    datamodule: Any,  # Changed from pl.LightningDataModule to avoid import issues
    benchmark_config: BenchmarkConfig,
    benchmark_frequency: int = 100,
    **trainer_kwargs,
) -> list[BenchmarkResult]:
    """Run training with integrated benchmarking."""

    # Create trainer with benchmark callback
    trainer = create_lightning_trainer_with_benchmark(
        benchmark_config=benchmark_config,
        benchmark_frequency=benchmark_frequency,
        **trainer_kwargs,
    )

    # Train the model
    trainer.fit(model, datamodule)

    # Get benchmark results from callback
    benchmark_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, BenchmarkCallback):
            benchmark_callback = callback
            break

    if benchmark_callback:
        return benchmark_callback.benchmark_results
    else:
        return []


def create_benchmark_config_for_lightning(
    d_model: int = 512,
    n_heads: int = 8,
    batch_size: int = 1,
    sequence_lengths: list[int] | None = None,
    device: str = "cuda",
    **kwargs,
) -> BenchmarkConfig:
    """Create a benchmark config optimized for PyTorch Lightning."""

    if sequence_lengths is None:
        sequence_lengths = [1024, 2048, 4096, 8192]

    return BenchmarkConfig(
        d_model=d_model,
        n_heads=n_heads,
        batch_size=batch_size,
        sequence_lengths=sequence_lengths,
        device=device,
        torch_benchmark=True,
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        warmup_runs=2,  # Fewer warmup runs for training
        benchmark_runs=5,  # Fewer benchmark runs for training
        verbose=False,  # Less verbose during training
        **kwargs,
    )

"""Specific benchmarks for each attention block type."""

import torch.nn as nn

from .base import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from .utils import create_attention_mask, create_input_data


class BlockBenchmarkRunner(BenchmarkRunner):
    """Extended benchmark runner with block-specific functionality."""

    def benchmark_all_blocks(
        self,
        blocks: dict[str, nn.Module],
        custom_input_generators: dict[str, callable] | None = None,
    ) -> dict[str, list[BenchmarkResult]]:
        """Benchmark multiple blocks and return organized results."""

        all_results = {}

        for block_name, block in blocks.items():
            if self.config.verbose:
                print(f"\n{'=' * 60}")
                print(f"Benchmarking {block_name}")
                print(f"{'=' * 60}")

            # Get custom input generator if available
            input_generator = None
            if custom_input_generators and block_name in custom_input_generators:
                input_generator = custom_input_generators[block_name]

            # Run benchmark
            results = self.benchmark_block(block, block_name, input_generator)
            all_results[block_name] = results

        return all_results


def create_blade_input_generator(config: BenchmarkConfig):
    """Create input generator for BLADE block."""

    def generator(seq_len: int, batch_size: int, d_model: int):
        # Create tensors on CPU, they'll be moved to device later
        x = create_input_data(seq_len, batch_size, d_model, "cpu")
        return [x]

    return generator


def create_dpassm_input_generator(config: BenchmarkConfig):
    """Create input generator for DPASSM block."""

    def generator(seq_len: int, batch_size: int, d_model: int):
        # Create tensors on CPU, they'll be moved to device later
        x = create_input_data(seq_len, batch_size, d_model, "cpu")
        return [x]

    return generator


def create_longformer_input_generator(config: BenchmarkConfig):
    """Create input generator for Longformer block."""

    def generator(seq_len: int, batch_size: int, d_model: int):
        # Create tensors on CPU, they'll be moved to device later
        x = create_input_data(seq_len, batch_size, d_model, "cpu")
        attention_mask = create_attention_mask(seq_len, batch_size, "cpu")
        return [x, attention_mask]

    return generator


def create_bigbird_input_generator(config: BenchmarkConfig):
    """Create input generator for BigBird block."""

    def generator(seq_len: int, batch_size: int, d_model: int):
        # Create tensors on CPU, they'll be moved to device later
        x = create_input_data(seq_len, batch_size, d_model, "cpu")
        attention_mask = create_attention_mask(seq_len, batch_size, "cpu")
        return [x, attention_mask]

    return generator


def create_standard_blocks(config: BenchmarkConfig) -> dict[str, nn.Module]:
    """Create standard instances of all blocks for benchmarking."""
    from efficient_longctx.blocks import (
        BigBirdBlock,
        BLADEBlock,
        DPASSMBlock,
        LongformerBlock,
    )

    blocks = {}

    # BLADE Block
    blocks["BLADE"] = BLADEBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        chunk_size=512,
        state_dim=64,
        m_global=0,
        dropout=0.1,
    )

    # DPASSM Block
    blocks["DPASSM"] = DPASSMBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        window_size=256,
        ssm_state_dim=64,
        dropout=0.1,
    )

    # Longformer Block
    blocks["Longformer"] = LongformerBlock(
        d_model=config.d_model, n_heads=config.n_heads, window_size=512, dropout=0.1
    )

    # BigBird Block
    blocks["BigBird"] = BigBirdBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        window_size=64,
        n_random_tokens=3,
        n_global_tokens=2,
        dropout=0.1,
    )

    return blocks


def create_custom_input_generators(config: BenchmarkConfig) -> dict[str, callable]:
    """Create custom input generators for each block type."""
    return {
        "BLADE": create_blade_input_generator(config),
        "DPASSM": create_dpassm_input_generator(config),
        "Longformer": create_longformer_input_generator(config),
        "BigBird": create_bigbird_input_generator(config),
    }


def run_comprehensive_benchmark(
    config: BenchmarkConfig | None = None, blocks_to_test: list[str] | None = None
) -> dict[str, list[BenchmarkResult]]:
    """Run comprehensive benchmark on all or selected blocks."""

    if config is None:
        config = BenchmarkConfig()

    # Create blocks
    all_blocks = create_standard_blocks(config)

    # Filter blocks if specified
    if blocks_to_test:
        blocks = {
            name: block for name, block in all_blocks.items() if name in blocks_to_test
        }
    else:
        blocks = all_blocks

    # Create input generators
    input_generators = create_custom_input_generators(config)

    # Run benchmark
    runner = BlockBenchmarkRunner(config)
    results = runner.benchmark_all_blocks(blocks, input_generators)

    # Save results
    runner.save_results()

    # Print summary
    runner.print_summary()

    return results


def compare_blocks_performance(
    results: dict[str, list[BenchmarkResult]], metric: str = "forward_time_mean"
) -> dict[str, dict[int, float]]:
    """Compare performance of different blocks across sequence lengths."""

    comparison = {}

    for block_name, block_results in results.items():
        comparison[block_name] = {}
        for result in block_results:
            comparison[block_name][result.sequence_length] = getattr(result, metric)

    return comparison


def find_best_block_for_length(
    results: dict[str, list[BenchmarkResult]],
    sequence_length: int,
    metric: str = "forward_time_mean",
    minimize: bool = True,
) -> str:
    """Find the best performing block for a specific sequence length."""

    best_block = None
    best_value = float("inf") if minimize else float("-inf")

    for block_name, block_results in results.items():
        for result in block_results:
            if result.sequence_length == sequence_length:
                value = getattr(result, metric)
                if minimize and value < best_value:
                    best_value = value
                    best_block = block_name
                elif not minimize and value > best_value:
                    best_value = value
                    best_block = block_name
                break

    return best_block

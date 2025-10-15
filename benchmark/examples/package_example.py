#!/usr/bin/env python3
"""Example script demonstrating the benchmark package usage."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmark.benchmark import (  # noqa: E402
    BenchmarkConfig,
    create_standard_blocks,
    run_comprehensive_benchmark,
)


def main():
    """Demonstrate basic benchmark usage."""
    print("Benchmark Package Example")
    print("=" * 50)

    # Create a simple benchmark configuration
    config = BenchmarkConfig(
        d_model=256,
        n_heads=4,
        sequence_lengths=[512, 1024],
        warmup_runs=2,
        benchmark_runs=3,
        verbose=True,
    )

    print(f"Configuration: d_model={config.d_model}, device={config.device}")

    # Create blocks to benchmark
    blocks = create_standard_blocks(config)
    print(f"Created {len(blocks)} blocks: {list(blocks.keys())}")

    # Run benchmark on specific blocks
    print("\nRunning benchmark on BLADE and DPASSM...")
    results = run_comprehensive_benchmark(
        config=config, blocks_to_test=["BLADE", "DPASSM"]
    )

    # Print results summary
    print("\nResults Summary:")
    print("-" * 30)
    for block_name, block_results in results.items():
        for result in block_results:
            print(
                f"{block_name} ({result.sequence_length} tokens): "
                f"{result.forward_time_mean * 1000:.2f}ms, "
                f"{result.throughput_tokens_per_sec:.0f} tokens/s"
            )

    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()

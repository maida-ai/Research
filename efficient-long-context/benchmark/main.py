#!/usr/bin/env python3
"""Main benchmark runner for efficient long-context attention blocks."""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the benchmark package to the path
benchmark_package = Path(__file__).parent
sys.path.insert(0, str(benchmark_package))

import argparse  # noqa: E402
import json  # noqa: E402

import torch  # noqa: E402
from benchmark.base import BenchmarkConfig  # noqa: E402
from benchmark.blocks import (  # noqa: E402
    compare_blocks_performance,
    run_comprehensive_benchmark,
)
from benchmark.lightning_benchmark import (  # noqa: E402
    create_benchmark_config_for_lightning,
)
from benchmark.utils import get_system_info  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark efficient long-context attention blocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with default settings
  python benchmark/main.py quick

  # Comprehensive benchmark with custom parameters
  python benchmark/main.py comprehensive --d-model 768 --sequence-lengths 1024 2048 4096

  # Test specific blocks only
  python benchmark/main.py quick --blocks BLADE DPASSM --seq-len 2048

  # Lightning-optimized benchmark
  python benchmark/main.py comprehensive --lightning-mode --batch-size 2

  # CPU benchmark
  python benchmark/main.py comprehensive --device cpu --sequence-lengths 1024 2048
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Benchmark mode")

    # Quick benchmark subcommand
    quick_parser = subparsers.add_parser(
        "quick", help="Quick benchmark for rapid testing"
    )
    _add_quick_args(quick_parser)

    # Comprehensive benchmark subcommand
    comp_parser = subparsers.add_parser(
        "comprehensive", help="Comprehensive benchmark with full configuration"
    )
    _add_comprehensive_args(comp_parser)

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Test the benchmarking framework")
    test_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser


def _add_quick_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for quick benchmark mode."""
    parser.add_argument(
        "--blocks",
        type=str,
        nargs="+",
        choices=["BLADE", "DPASSM", "Longformer", "BigBird"],
        help="Specific blocks to test",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048, help="Sequence length (default: 2048)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of benchmark runs (default: 5)"
    )
    parser.add_argument(
        "--d-model", type=int, default=512, help="Model dimension (default: 512)"
    )
    parser.add_argument(
        "--n-heads", type=int, default=8, help="Number of attention heads (default: 8)"
    )


def _add_comprehensive_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for comprehensive benchmark mode."""
    # Model configuration
    parser.add_argument(
        "--d-model", type=int, default=512, help="Model dimension (default: 512)"
    )
    parser.add_argument(
        "--n-heads", type=int, default=8, help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: 1)"
    )

    # Sequence lengths
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to benchmark (default: 1024 2048 4096 8192)",
    )

    # Benchmarking configuration
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Number of warmup runs (default: 3)"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )

    # PyTorch optimizations
    parser.add_argument(
        "--no-torch-benchmark", action="store_true", help="Disable torch benchmark"
    )
    parser.add_argument(
        "--no-cudnn-benchmark", action="store_true", help="Disable cuDNN benchmark"
    )
    parser.add_argument(
        "--cudnn-deterministic", action="store_true", help="Enable cuDNN deterministic"
    )

    # Memory profiling
    parser.add_argument(
        "--no-memory-profile", action="store_true", help="Disable memory profiling"
    )

    # Output configuration
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./reports/benchmark",
        help="Results directory (default: ./reports/benchmark)",
    )
    parser.add_argument(
        "--output-file", type=str, help="Output file name (default: auto-generated)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Quiet mode (less verbose output)"
    )

    # Block selection
    parser.add_argument(
        "--blocks",
        type=str,
        nargs="+",
        choices=["BLADE", "DPASSM", "Longformer", "BigBird"],
        help="Specific blocks to benchmark",
    )

    # Lightning mode
    parser.add_argument(
        "--lightning-mode", action="store_true", help="Use Lightning-optimized settings"
    )


def create_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Create benchmark config from command line arguments."""
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create config based on mode
    if args.command == "quick":
        config = BenchmarkConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            batch_size=args.batch_size,
            sequence_lengths=[args.seq_len],
            warmup_runs=2,
            benchmark_runs=args.runs,
            device=device,
            torch_benchmark=True,
            cudnn_benchmark=True,
            cudnn_deterministic=False,
            profile_memory=True,
            verbose=True,
            save_results=False,
        )
    elif args.command == "comprehensive":
        if args.lightning_mode:
            config = create_benchmark_config_for_lightning(
                d_model=args.d_model,
                n_heads=args.n_heads,
                batch_size=args.batch_size,
                sequence_lengths=args.sequence_lengths,
                device=device,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
                torch_benchmark=not args.no_torch_benchmark,
                cudnn_benchmark=not args.no_cudnn_benchmark,
                cudnn_deterministic=args.cudnn_deterministic,
                profile_memory=not args.no_memory_profile,
                verbose=not args.quiet,
                results_dir=args.results_dir,
            )
        else:
            config = BenchmarkConfig(
                d_model=args.d_model,
                n_heads=args.n_heads,
                batch_size=args.batch_size,
                sequence_lengths=args.sequence_lengths,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
                device=device,
                torch_benchmark=not args.no_torch_benchmark,
                cudnn_benchmark=not args.no_cudnn_benchmark,
                cudnn_deterministic=args.cudnn_deterministic,
                profile_memory=not args.no_memory_profile,
                verbose=not args.quiet,
                results_dir=args.results_dir,
            )
    else:
        raise ValueError(f"Unknown command: {args.command}")

    return config


def print_system_info() -> None:
    """Print system information."""
    info = get_system_info()
    print("System Information:")
    print("-" * 50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("-" * 50)


def run_quick_benchmark(args: argparse.Namespace) -> int:
    """Run quick benchmark mode."""
    config = create_config_from_args(args)

    print(
        f"Quick benchmark: {args.seq_len} tokens, {args.batch_size} batch, {config.device}"
    )
    print("=" * 60)

    try:
        results = run_comprehensive_benchmark(config=config, blocks_to_test=args.blocks)

        # Print quick summary
        print("\nQuick Results:")
        print("-" * 30)
        for block_name, block_results in results.items():
            for result in block_results:
                print(
                    f"{block_name}: {result.forward_time_mean * 1000:.2f}ms "
                    f"({result.throughput_tokens_per_sec:.0f} tokens/s)"
                )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run_comprehensive_benchmark_mode(args: argparse.Namespace) -> int:
    """Run comprehensive benchmark mode."""
    config = create_config_from_args(args)

    # Print system info
    if not args.quiet:
        print_system_info()

    try:
        results = run_comprehensive_benchmark(config=config, blocks_to_test=args.blocks)

        # Save results with custom filename if specified
        if args.output_file:
            import os

            os.makedirs(config.results_dir, exist_ok=True)
            filepath = os.path.join(config.results_dir, args.output_file)

            results_data = []
            for block_results in results.values():
                results_data.extend([result.to_dict() for result in block_results])

            with open(filepath, "w") as f:
                json.dump(results_data, f, indent=2)

            if not args.quiet:
                print(f"\nResults saved to {filepath}")

        # Print comparison
        if not args.quiet and len(results) > 1:
            print("\nPerformance Comparison (Forward Time):")
            print("-" * 50)
            comparison = compare_blocks_performance(results, "forward_time_mean")

            for seq_len in config.sequence_lengths:
                print(f"\nSequence Length {seq_len}:")
                block_times = {}
                for block_name, times in comparison.items():
                    if seq_len in times:
                        block_times[block_name] = times[seq_len]

                # Sort by performance
                sorted_blocks = sorted(block_times.items(), key=lambda x: x[1])
                for i, (block_name, time_ms) in enumerate(sorted_blocks):
                    print(f"  {i + 1}. {block_name}: {time_ms * 1000:.2f}ms")

        return 0

    except Exception as e:
        print(f"Error during benchmarking: {e}", file=sys.stderr)
        return 1


def run_test_mode(args: argparse.Namespace) -> int:
    """Run test mode using pytest."""
    import subprocess

    test_dir = Path(__file__).parent / "tests"

    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}", file=sys.stderr)
        return 1

    cmd = ["python", "-m", "pytest", str(test_dir)]
    if args.verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        print(f"Error running tests: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "quick":
        return run_quick_benchmark(args)
    elif args.command == "comprehensive":
        return run_comprehensive_benchmark_mode(args)
    elif args.command == "test":
        return run_test_mode(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

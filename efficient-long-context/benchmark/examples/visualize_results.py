#!/usr/bin/env python3
"""Visualization script for benchmark results."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> list[dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def plot_performance_comparison(
    results: list[dict[str, Any]], output_dir: str = "./reports/benchmark/plots"
):
    """Plot performance comparison across blocks and sequence lengths."""

    # Organize data
    block_data = {}
    for result in results:
        block_name = result["block_name"]
        seq_len = result["sequence_length"]
        forward_time = result["forward_time_mean"]

        if block_name not in block_data:
            block_data[block_name] = {"seq_lens": [], "times": []}

        block_data[block_name]["seq_lens"].append(seq_len)
        block_data[block_name]["times"].append(forward_time * 1000)  # Convert to ms

    # Create plots
    Path(output_dir).mkdir(exist_ok=True)

    # 1. Forward time vs sequence length
    plt.figure(figsize=(12, 8))
    for block_name, data in block_data.items():
        plt.plot(
            data["seq_lens"], data["times"], marker="o", label=block_name, linewidth=2
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Forward Time (ms)")
    plt.title("Forward Time vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/forward_time_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Throughput comparison
    plt.figure(figsize=(12, 8))
    for block_name, data in block_data.items():
        throughputs = []
        for seq_len in data["seq_lens"]:
            # Calculate throughput from results
            for result in results:
                if (
                    result["block_name"] == block_name
                    and result["sequence_length"] == seq_len
                ):
                    throughputs.append(result["throughput_tokens_per_sec"])
                    break

        plt.plot(
            data["seq_lens"], throughputs, marker="s", label=block_name, linewidth=2
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Throughput vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Memory usage comparison
    plt.figure(figsize=(12, 8))
    for block_name, data in block_data.items():
        memory_usage = []
        for seq_len in data["seq_lens"]:
            for result in results:
                if (
                    result["block_name"] == block_name
                    and result["sequence_length"] == seq_len
                ):
                    memory_usage.append(result["allocated_memory_mb"])
                    break

        plt.plot(
            data["seq_lens"], memory_usage, marker="^", label=block_name, linewidth=2
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Memory efficiency
    plt.figure(figsize=(12, 8))
    for block_name, data in block_data.items():
        memory_efficiency = []
        for seq_len in data["seq_lens"]:
            for result in results:
                if (
                    result["block_name"] == block_name
                    and result["sequence_length"] == seq_len
                ):
                    memory_efficiency.append(result["memory_efficiency_mb_per_token"])
                    break

        plt.plot(
            data["seq_lens"],
            memory_efficiency,
            marker="d",
            label=block_name,
            linewidth=2,
        )

    plt.xlabel("Sequence Length")
    plt.ylabel("Memory Efficiency (MB/token)")
    plt.title("Memory Efficiency vs Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log", base=2)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/memory_efficiency_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_scaling_analysis(
    results: list[dict[str, Any]], output_dir: str = "./reports/benchmark/plots"
):
    """Plot scaling analysis for each block."""

    # Organize data by block
    block_data = {}
    for result in results:
        block_name = result["block_name"]
        seq_len = result["sequence_length"]
        forward_time = result["forward_time_mean"]

        if block_name not in block_data:
            block_data[block_name] = {"seq_lens": [], "times": []}

        block_data[block_name]["seq_lens"].append(seq_len)
        block_data[block_name]["times"].append(forward_time)

    # Create subplots for each block
    n_blocks = len(block_data)
    fig, axes = plt.subplots(2, (n_blocks + 1) // 2, figsize=(15, 10))
    if n_blocks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (block_name, data) in enumerate(block_data.items()):
        ax = axes[i]

        seq_lens = np.array(data["seq_lens"])
        times = np.array(data["times"])

        # Plot actual data
        ax.scatter(seq_lens, times, alpha=0.7, s=50)

        # Fit scaling law (quadratic for attention)
        if len(seq_lens) > 1:
            # Log-log fit
            log_seq = np.log(seq_lens)
            log_times = np.log(times)
            coeffs = np.polyfit(log_seq, log_times, 1)

            # Generate fitted line
            seq_fit = np.logspace(
                np.log10(seq_lens.min()), np.log10(seq_lens.max()), 100
            )
            times_fit = np.exp(coeffs[1]) * (seq_fit ** coeffs[0])

            ax.plot(
                seq_fit,
                times_fit,
                "r--",
                alpha=0.8,
                label=f"Scaling: O(n^{coeffs[0]:.2f})",
            )

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Forward Time (s)")
        ax.set_title(f"{block_name} Scaling")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_blocks, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_table(
    results: list[dict[str, Any]],
    output_file: str = "./reports/benchmark/summary_table.txt",
):
    """Create a text summary table of results."""

    # Organize data
    block_data = {}
    seq_lens = set()

    for result in results:
        block_name = result["block_name"]
        seq_len = result["sequence_length"]
        seq_lens.add(seq_len)

        if block_name not in block_data:
            block_data[block_name] = {}

        block_data[block_name][seq_len] = result

    seq_lens = sorted(seq_lens)

    # Create table
    with open(output_file, "w") as f:
        f.write("Benchmark Results Summary\n")
        f.write("=" * 80 + "\n\n")

        # Forward time table
        f.write("Forward Time (ms):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Block':<12}")
        for seq_len in seq_lens:
            f.write(f"{seq_len:>8}")
        f.write("\n")
        f.write("-" * 50 + "\n")

        for block_name in sorted(block_data.keys()):
            f.write(f"{block_name:<12}")
            for seq_len in seq_lens:
                if seq_len in block_data[block_name]:
                    time_ms = (
                        block_data[block_name][seq_len]["forward_time_mean"] * 1000
                    )
                    f.write(f"{time_ms:>8.2f}")
                else:
                    f.write(f"{'N/A':>8}")
            f.write("\n")

        f.write("\n")

        # Throughput table
        f.write("Throughput (tokens/s):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Block':<12}")
        for seq_len in seq_lens:
            f.write(f"{seq_len:>8}")
        f.write("\n")
        f.write("-" * 50 + "\n")

        for block_name in sorted(block_data.keys()):
            f.write(f"{block_name:<12}")
            for seq_len in seq_lens:
                if seq_len in block_data[block_name]:
                    throughput = block_data[block_name][seq_len][
                        "throughput_tokens_per_sec"
                    ]
                    f.write(f"{throughput:>8.0f}")
                else:
                    f.write(f"{'N/A':>8}")
            f.write("\n")

        f.write("\n")

        # Memory usage table
        f.write("Memory Usage (MB):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Block':<12}")
        for seq_len in seq_lens:
            f.write(f"{seq_len:>8}")
        f.write("\n")
        f.write("-" * 50 + "\n")

        for block_name in sorted(block_data.keys()):
            f.write(f"{block_name:<12}")
            for seq_len in seq_lens:
                if seq_len in block_data[block_name]:
                    memory = block_data[block_name][seq_len]["allocated_memory_mb"]
                    f.write(f"{memory:>8.1f}")
                else:
                    f.write(f"{'N/A':>8}")
            f.write("\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results_file", help="JSON file with benchmark results")
    parser.add_argument(
        "--output-dir",
        default="./reports/benchmark/plots",
        help="Output directory for plots",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--summary-table", help="Generate summary table file")

    args = parser.parse_args()

    # Load results
    try:
        results = load_results(args.results_file)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        return 1

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Generate plots
    if not args.no_plots:
        try:
            plot_performance_comparison(results, args.output_dir)
            plot_scaling_analysis(results, args.output_dir)
            print(f"Plots saved to {args.output_dir}")
        except Exception as e:
            print(f"Error generating plots: {e}", file=sys.stderr)
            return 1

    # Generate summary table
    if args.summary_table:
        try:
            create_summary_table(results, args.summary_table)
            print(f"Summary table saved to {args.summary_table}")
        except Exception as e:
            print(f"Error generating summary table: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

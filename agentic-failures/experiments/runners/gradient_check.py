"""
gradient_check.py — Reference-agent gradient validation harness (plan §9.5).

Sweeps generator difficulty params over the formal (Blocks World) instance set,
runs the frozen reference agent once per instance, computes failure rate per
parameter bin, checks the 10%-90% span requirement, and emits:
  data/calibration/reference_gradient.json
  PASS or DEGENERATE verdict with fallback ladder printed on failure.

Dry-runnable with MockModel (--mock). RUN the mock path to prove harness works.
Real runs require --live and a running LocalQwenClient server.

Usage:
    .venv/bin/python -m experiments.runners.gradient_check --mock --seed 42
    .venv/bin/python -m experiments.runners.gradient_check --live --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runners.models import (
    LocalQwenClient,
    LocalQwenConfig,
    MockModel,
    NotConfiguredError,
    BUDGET_METER_SOURCE,
)
from experiments.runners.scaffold import (
    BaselineAgent,
    RunConfig,
    TaskInstance,
    _classify_terminal,
)
from experiments.runners.protocol import TERMINAL_PROTOCOL


# ---------------------------------------------------------------------------
# Generator difficulty parameter sweep
# ---------------------------------------------------------------------------

@dataclass
class DifficultyBin:
    """One bin in the difficulty sweep."""
    label: str            # e.g. "n_blocks=2"
    n_blocks: int
    n_buffers: int
    is_infeasible: bool
    description_template: str

    def make_instance(self, idx: int, seed: int) -> TaskInstance:
        rng = random.Random(seed * 1000 + idx)
        blocks = [chr(65 + i) for i in range(self.n_blocks)]
        buffers = {f"b{j}": None for j in range(self.n_buffers)}
        buf_names = list(buffers.keys())
        buf_str = ", ".join(f"{k}: empty" for k in buf_names)

        if self.is_infeasible:
            # Generate an infeasible instance: stack too long for buffer capacity
            stack_str = " -> ".join(blocks) + " (bottom to top)"
            rev = list(reversed(blocks))
            goal_str = " -> ".join(rev) + " (bottom to top)"
            description = (
                f"Initial state:\n"
                f"  Stack 0: [{', '.join(blocks)}]\n"
                f"  {buf_str}\n"
                f"Resources: table_buffers={self.n_buffers} "
                f"(at most {self.n_buffers} buffer(s) occupied at once)\n"
                f"Hard constraints: [{{\"type\": \"buffer_limit\", \"max\": {self.n_buffers}}}]\n"
                f"Goal: Reverse the stack so it is [{', '.join(rev)}]\n"
                f"Note: Reversing {self.n_blocks} blocks with only {self.n_buffers} "
                f"buffer(s) requires {self.n_blocks - 1} buffers simultaneously. "
                f"{'INFEASIBLE under buffer constraint.' if self.n_blocks - 1 > self.n_buffers else 'Feasible.'}\n"
            )
            oracle_feasible = (self.n_blocks - 1) <= self.n_buffers
        else:
            # Feasible: move one block to free up space
            description = (
                f"Initial state:\n"
                f"  Stack 0: [{', '.join(blocks)}]\n"
                f"  {buf_str}\n"
                f"Resources: table_buffers={self.n_buffers}\n"
                f"Hard constraints: none\n"
                f"Goal: Move the top block ({blocks[-1]}) to buffer b0, "
                f"then move block {blocks[0]} to stack 1.\n"
            )
            oracle_feasible = True

        return TaskInstance(
            instance_id=f"grad-{self.label}-i{idx:03d}",
            domain="formal",
            pressure_band="medium_feasible" if oracle_feasible else "minimally_unsat",
            pressure_axis="resource",
            band_ladder="feasibility_contradiction",
            description=description,
            domain_spec=(
                "Domain: Resource-constrained Blocks World\n"
                "Moves: move_stack_to_stack, move_stack_to_buffer, "
                "move_buffer_to_stack, move_buffer_to_buffer.\n"
                "Buffer limit = max simultaneous occupied buffers.\n"
            ),
            feasible=oracle_feasible,
            oracle_success_fn=lambda r, _f=oracle_feasible: (
                r.get("terminal_report") == "claim_success" if _f
                else r.get("terminal_report") == "claim_infeasible"
            ),
        )


# Sweep bins: vary n_blocks from 2 to 6, n_buffers = 1 or 2
# At low n_blocks=2 the task should be easy (low failure rate).
# At high n_blocks=5-6 with n_buffers=1 the task should be hard (high failure rate).
SWEEP_BINS: list[DifficultyBin] = [
    DifficultyBin("n_blocks=2_buf=2_feas",  2, 2, False, ""),
    DifficultyBin("n_blocks=3_buf=2_feas",  3, 2, False, ""),
    DifficultyBin("n_blocks=4_buf=2_feas",  4, 2, False, ""),
    DifficultyBin("n_blocks=3_buf=1_infeas", 3, 1, True,  ""),  # infeasible
    DifficultyBin("n_blocks=4_buf=1_infeas", 4, 1, True,  ""),  # infeasible
    DifficultyBin("n_blocks=5_buf=2_feas",  5, 2, False, ""),
    DifficultyBin("n_blocks=6_buf=2_feas",  6, 2, False, ""),
]

# Number of instances per bin in the mock run (tiny for smoke; increase for real)
INSTANCES_PER_BIN_MOCK = 2
INSTANCES_PER_BIN_LIVE = 5

# Gradient requirement: failure rate must span [MIN_FAILURE, MAX_FAILURE]
MIN_FAILURE_RATE = 0.10
MAX_FAILURE_RATE = 0.90


# ---------------------------------------------------------------------------
# Mock model that produces graded difficulty-aware responses
# ---------------------------------------------------------------------------

class _GradedMockModel(MockModel):
    """
    Mock model with difficulty-graded responses for gradient validation.
    Low n_blocks -> solves; high n_blocks or infeasible -> fails / claims infeasible.
    """

    def __init__(self, difficulty_bin: DifficultyBin, seed: int):
        self._bin = difficulty_bin
        self._seed = seed
        self._model_id_str = f"mock-graded-{difficulty_bin.label}"
        rng = random.Random(seed)

        # Determine which behavior to use based on difficulty
        nb = difficulty_bin.n_blocks
        nbufs = difficulty_bin.n_buffers
        infeas = difficulty_bin.is_infeasible

        if infeas:
            # Actually infeasible — mock claims infeasible
            behavior = "claims_infeasible"
        elif nb <= 2:
            behavior = "solves_easy"
        elif nb <= 3:
            # 70% solve, 30% honest_unknown
            behavior = "solves_easy" if rng.random() < 0.70 else "honest_unknown"
        elif nb <= 4:
            behavior = "solves_easy" if rng.random() < 0.40 else "honest_unknown"
        else:
            # Hard: mostly loop or unknown
            behavior = "loops" if rng.random() < 0.50 else "honest_unknown"

        super().__init__(behavior=behavior, model_id_str=self._model_id_str)

    @property
    def model_id(self) -> str:
        return self._model_id_str


# ---------------------------------------------------------------------------
# Gradient check runner
# ---------------------------------------------------------------------------

def is_trajectory_failure(summary: dict, instance: TaskInstance) -> bool:
    """
    A trajectory is a 'failure' (for gradient purposes) if the reference agent
    did not produce a correct terminal claim:
      - feasible instance: failure = not claim_success
      - infeasible instance: failure = not claim_infeasible
    """
    tr = summary.get("terminal_report", "report_unknown")
    if instance.feasible:
        return tr != "claim_success"
    else:
        return tr != "claim_infeasible"


def run_gradient_check(
    mock: bool,
    seed: int,
    output_dir: str = "data/calibration",
    exp_id: str = "gradient_check",
) -> dict:
    """
    Run the gradient validation sweep.
    Returns a result dict with PASS/DEGENERATE verdict and per-bin stats.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_per_bin = INSTANCES_PER_BIN_MOCK if mock else INSTANCES_PER_BIN_LIVE
    rng = random.Random(seed)

    print(f"Gradient check: {'MOCK' if mock else 'LIVE'} mode, seed={seed}, {n_per_bin} instances/bin")
    print()

    bin_results = []
    all_failure_rates = []

    for bin_idx, diff_bin in enumerate(SWEEP_BINS):
        bin_failures = 0
        bin_total = 0
        bin_summaries = []

        for inst_idx in range(n_per_bin):
            inst_seed = seed * 10000 + bin_idx * 100 + inst_idx
            instance = diff_bin.make_instance(inst_idx, inst_seed)

            run_id = f"grad-{diff_bin.label}-{inst_idx:02d}-seed{seed}"
            config = RunConfig(
                run_id=run_id,
                seed=seed,
                model_id=f"mock-graded-{diff_bin.label}" if mock else "qwen2.5-7b-instruct",
                architecture="baseline",
                system_type="LLM_agent",
                max_turns=4,
                max_tokens_per_call=512,
                context_window_steps=4,
                budget={"steps": 4, "tool_calls": 10, "tokens": 3000, "seconds": 30},
                trace_dir="data/traces",
                exp_id=exp_id,
            )

            if mock:
                model = _GradedMockModel(diff_bin, inst_seed)
            else:
                qwen_config = LocalQwenConfig()
                model = LocalQwenClient(qwen_config, live=True)

            agent = BaselineAgent(model=model, config=config)
            try:
                summary = agent.run(instance)
            except NotConfiguredError as e:
                print(f"  ERROR: {e}")
                sys.exit(1)

            failed = is_trajectory_failure(summary, instance)
            if failed:
                bin_failures += 1
            bin_total += 1
            bin_summaries.append({
                "run_id": run_id,
                "instance_id": instance.instance_id,
                "terminal_report": summary["terminal_report"],
                "failed": failed,
                "steps_used": summary["steps_used"],
                "tokens_used": summary["tokens_used"],
            })

        failure_rate = bin_failures / bin_total if bin_total > 0 else 0.0
        all_failure_rates.append(failure_rate)

        print(
            f"  Bin {bin_idx+1:02d}/{len(SWEEP_BINS)}: {diff_bin.label:35s}"
            f"  failures={bin_failures}/{bin_total}  rate={failure_rate:.2f}"
        )

        bin_results.append({
            "bin_label": diff_bin.label,
            "n_blocks": diff_bin.n_blocks,
            "n_buffers": diff_bin.n_buffers,
            "is_infeasible": diff_bin.is_infeasible,
            "n_instances": bin_total,
            "n_failures": bin_failures,
            "failure_rate": failure_rate,
            "runs": bin_summaries,
        })

    print()

    # -----------------------------------------------------------------------
    # Gradient validation: check 10%-90% span requirement (plan §9.5)
    # -----------------------------------------------------------------------
    min_rate = min(all_failure_rates)
    max_rate = max(all_failure_rates)
    span = max_rate - min_rate
    span_ok = min_rate <= MIN_FAILURE_RATE and max_rate >= MAX_FAILURE_RATE

    verdict = "PASS" if span_ok else "DEGENERATE"

    print(f"Gradient span: min={min_rate:.2f}, max={max_rate:.2f}, span={span:.2f}")
    print(f"Required: min <= {MIN_FAILURE_RATE}, max >= {MAX_FAILURE_RATE}")
    print(f"Verdict: {verdict}")
    print()

    if verdict == "DEGENERATE":
        print("FALLBACK LADDER (plan §9.5):")
        print("  1. Use a stronger pinned reference model.")
        print("  2. Keep same model but allow a slightly stronger non-test scaffold.")
        print("  3. Soften the calibration metric to partial-credit constraint satisfaction.")
        print("  4. Measure success within N retries rather than first-shot success.")
        print()

    result = {
        "verdict": verdict,
        "mock_mode": mock,
        "seed": seed,
        "n_bins": len(SWEEP_BINS),
        "n_instances_per_bin": n_per_bin,
        "min_failure_rate": min_rate,
        "max_failure_rate": max_rate,
        "span": span,
        "required_min": MIN_FAILURE_RATE,
        "required_max": MAX_FAILURE_RATE,
        "span_ok": span_ok,
        "all_failure_rates": all_failure_rates,
        "bins": bin_results,
        "budget_meter_source": BUDGET_METER_SOURCE,
        "run_command": (
            f".venv/bin/python -m experiments.runners.gradient_check "
            f"{'--mock' if mock else '--live'} --seed {seed}"
        ),
    }

    out_file = output_path / "reference_gradient.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Results written to: {out_file}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reference-agent gradient validation (plan §9.5)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mock", action="store_true", help="Use MockModel (dry run, $0)")
    mode_group.add_argument("--live", action="store_true", help="Use LocalQwenClient (requires server)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default="data/calibration", help="Output directory")
    args = parser.parse_args()

    result = run_gradient_check(
        mock=args.mock,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if result["verdict"] == "DEGENERATE":
        sys.exit(1)
    else:
        print(f"Gradient check PASSED. Span: {result['span']:.2f} "
              f"(min={result['min_failure_rate']:.2f}, max={result['max_failure_rate']:.2f})")


if __name__ == "__main__":
    main()

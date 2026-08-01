"""
runner.py — Exp 2 formal mini-pilot orchestration.

Thin orchestration over scaffold.py + exp1 instance data.
--dry-run mode uses MockModel over 3 instances (RUN the dry-run).
Refuses to run live without --budget-cap AND --confirm-live.

Usage:
    # Dry run (MockModel, 3 instances, $0)
    .venv/bin/python -m experiments.exp2_formal_minipilot.runner --dry-run --seed 42

    # Live run (requires filled prices.json + --budget-cap + --confirm-live)
    .venv/bin/python -m experiments.exp2_formal_minipilot.runner \\
        --budget-cap 3000.0 \\
        --confirm-live \\
        --seeds 42,123,777 \\
        --models claude-sonnet-4-5,qwen2.5-7b-instruct \\
        --architectures baseline,verifier_augmented \\
        --exp-id exp2_formal_minipilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runners.models import (
    AnthropicClient,
    BudgetExceeded,
    LocalQwenClient,
    LocalQwenConfig,
    MockModel,
    NotConfiguredError,
)
from experiments.runners.scaffold import (
    BaselineAgent,
    RunConfig,
    TaskInstance,
    VerifierAugmentedAgent,
)


# ---------------------------------------------------------------------------
# Dry-run instances (3 hardcoded for --dry-run mode)
# ---------------------------------------------------------------------------

_DRY_RUN_DOMAIN_SPEC = (
    "Domain: Resource-constrained Blocks World. "
    "Moves: move_stack_to_stack, move_stack_to_buffer, move_buffer_to_stack. "
    "Buffer limit = max simultaneous occupied buffers."
)

DRY_RUN_INSTANCES = [
    TaskInstance(
        instance_id="exp2-dry-0001",
        domain="formal",
        pressure_band="easy_feasible",
        pressure_axis="resource",
        band_ladder="feasibility_contradiction",
        description=(
            "Stack 0: [A]. Stack 1: [B]. Buffer b0: empty.\n"
            "Goal: Stack 0: [B, A]. Resources: table_buffers=1."
        ),
        domain_spec=_DRY_RUN_DOMAIN_SPEC,
        feasible=True,
        oracle_success_fn=lambda r: r.get("terminal_report") == "claim_success",
    ),
    TaskInstance(
        instance_id="exp2-dry-0002",
        domain="formal",
        pressure_band="minimally_unsat",
        pressure_axis="resource",
        band_ladder="feasibility_contradiction",
        description=(
            "Stack 0: [A, B, C]. Buffer b0: empty (buffer_limit=1).\n"
            "Goal: Stack 0: [C, B, A]. Reversing 3 blocks with 1 buffer is INFEASIBLE."
        ),
        domain_spec=_DRY_RUN_DOMAIN_SPEC,
        feasible=False,
        oracle_success_fn=lambda r: False,
    ),
    TaskInstance(
        instance_id="exp2-dry-0003",
        domain="formal",
        pressure_band="near_boundary_feasible",
        pressure_axis="resource",
        band_ladder="feasibility_contradiction",
        description=(
            "Stack 0: [A, B]. Stack 1: [C, D]. Buffers b0, b1: empty.\n"
            "Goal: Stack 0: [D, C, B, A]. Resources: table_buffers=2."
        ),
        domain_spec=_DRY_RUN_DOMAIN_SPEC,
        feasible=True,
        oracle_success_fn=lambda r: r.get("terminal_report") == "claim_success",
    ),
]


# ---------------------------------------------------------------------------
# Instance loader (from exp1 output; falls back to dry-run instances)
# ---------------------------------------------------------------------------

def load_instances(
    instances_path: Optional[str] = None,
    n_max: Optional[int] = None,
) -> list[TaskInstance]:
    """
    Load instances from instances.jsonl (Track B output) or return dry-run instances.
    """
    if instances_path is None:
        instances_path = "data/instances/blocksworld/v1/instances.jsonl"

    path = Path(instances_path)
    if not path.exists():
        return DRY_RUN_INSTANCES[:n_max] if n_max else DRY_RUN_INSTANCES

    instances = []
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        inst = TaskInstance(
            instance_id=d["instance_id"],
            domain=d.get("domain", "formal"),
            pressure_band=d.get("pressure_band", "medium_feasible"),
            pressure_axis=d.get("pressure_axis", "resource"),
            band_ladder=d.get("band_ladder", "feasibility_contradiction"),
            description=d.get("description", str(d.get("initial_state", {}))),
            domain_spec=_DRY_RUN_DOMAIN_SPEC,
            feasible=d.get("feasible"),
            oracle_success_fn=(
                lambda r, f=d.get("feasible"): r.get("terminal_report") == "claim_success"
                if f else r.get("terminal_report") == "claim_infeasible"
            ),
        )
        instances.append(inst)
        if n_max and len(instances) >= n_max:
            break
    return instances


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(
    architecture: str,
    model_id: str,
    seed: int,
    run_id: str,
    exp_id: str,
    budget_cap: float,
    dry_run: bool,
) -> tuple:
    """
    Create (agent, model_client) pair for the given architecture and model.
    Dry run always uses MockModel.
    """
    if dry_run:
        behavior_cycle = ["solves_easy", "claims_infeasible", "honest_unknown"]
        # Round-robin behavior per run_id hash
        idx = hash(run_id) % 3
        model = MockModel(behavior=behavior_cycle[idx])
    elif "qwen" in model_id.lower():
        config = LocalQwenConfig(
            model_path_or_name="Qwen/Qwen2.5-7B-Instruct",
            endpoint_url="http://localhost:8000/v1",
        )
        model = LocalQwenClient(config, live=True)
    else:
        model = AnthropicClient(
            model_id_str=model_id,
            live=True,
            cap_usd=budget_cap,
            budget_state_path=f"data/traces/{exp_id}/_budget_{run_id}.json",
        )

    run_config = RunConfig(
        run_id=run_id,
        seed=seed,
        model_id=model.model_id,
        architecture=architecture,
        system_type="LLM_agent",
        max_turns=10,
        max_tokens_per_call=1024,
        context_window_steps=6,
        budget={"steps": 10, "tool_calls": 50, "tokens": 10000, "seconds": 120},
        trace_dir="data/traces",
        exp_id=exp_id,
    )

    if architecture == "verifier_augmented":
        agent = VerifierAugmentedAgent(
            model=model,
            config=run_config,
            legal_move_checker=lambda state, action: True,   # placeholder
            resource_ledger=lambda state: {"available": 2, "used": 0, "peak": 1, "slack": 2},
            constraint_validator=lambda state, constr: [],
        )
    else:
        agent = BaselineAgent(model=model, config=run_config)

    return agent, model


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_exp2(
    dry_run: bool,
    seeds: list[int],
    model_ids: list[str],
    architectures: list[str],
    exp_id: str,
    budget_cap: float,
    instances_path: Optional[str] = None,
    n_dry_instances: int = 3,
) -> list[dict]:
    """
    Execute Exp 2 trajectories and return summary list.
    """
    if dry_run:
        instances = DRY_RUN_INSTANCES[:n_dry_instances]
        model_ids = ["mock-model-v1"]
        print(f"DRY RUN: {n_dry_instances} instances, MockModel only, $0")
    else:
        instances = load_instances(instances_path)
        print(f"LIVE RUN: {len(instances)} instances, models={model_ids}, "
              f"architectures={architectures}, seeds={seeds}")

    all_results = []
    total_trajectories = len(instances) * len(architectures) * len(seeds) * len(model_ids)
    traj_count = 0

    for seed in seeds:
        for arch in architectures:
            for model_id in model_ids:
                for inst in instances:
                    traj_count += 1
                    run_id = (
                        f"{exp_id}-{inst.instance_id}-{arch}-"
                        f"{model_id.replace('/', '_')[:20]}-seed{seed}"
                    )

                    agent, model = make_agent(
                        architecture=arch,
                        model_id=model_id,
                        seed=seed,
                        run_id=run_id,
                        exp_id=exp_id,
                        budget_cap=budget_cap,
                        dry_run=dry_run,
                    )

                    t0 = time.monotonic()
                    try:
                        summary = agent.run(inst)
                        summary["wall_time_s"] = round(time.monotonic() - t0, 3)
                        summary["architecture"] = arch
                        summary["model_id"] = model_id
                        summary["seed"] = seed
                        all_results.append(summary)
                        print(
                            f"  [{traj_count:3d}/{total_trajectories}] "
                            f"inst={inst.instance_id:<20} arch={arch:<22} "
                            f"seed={seed}  terminal={summary['terminal_report']:<18}"
                        )
                    except BudgetExceeded as e:
                        print(f"  BUDGET EXCEEDED: {e}")
                        partial = {
                            "run_id": run_id,
                            "instance_id": inst.instance_id,
                            "status": "abandoned_budget_exceeded",
                            "architecture": arch,
                            "model_id": model_id,
                            "seed": seed,
                            "error": str(e),
                        }
                        all_results.append(partial)
                        if not dry_run:
                            print("  Halting: budget cap reached.")
                            return all_results

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 2 formal mini-pilot runner")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with MockModel on 3 instances. No API calls, $0."
    )
    parser.add_argument(
        "--budget-cap", type=float, default=None,
        help="USD budget cap (required for live runs)."
    )
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="Explicit confirmation for live API runs (required)."
    )
    parser.add_argument(
        "--seeds", default="42,123,777",
        help="Comma-separated seeds (default: 42,123,777)."
    )
    parser.add_argument(
        "--models",
        default="claude-sonnet-4-5,qwen2.5-7b-instruct",
        help="Comma-separated model IDs."
    )
    parser.add_argument(
        "--architectures", default="baseline,verifier_augmented",
        help="Comma-separated architectures."
    )
    parser.add_argument(
        "--exp-id", default="exp2_formal_minipilot",
        help="Experiment ID for trace directory."
    )
    parser.add_argument(
        "--instances-path", default=None,
        help="Path to instances.jsonl (default: data/instances/blocksworld/v1/instances.jsonl)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides --seeds).")
    args = parser.parse_args()

    # --- Safety gate: refuse live run without --budget-cap and --confirm-live ---
    if not args.dry_run:
        if args.budget_cap is None:
            print("ERROR: Live run requires --budget-cap <USD>. Refusing to run without it.")
            sys.exit(2)
        if not args.confirm_live:
            print("ERROR: Live run requires --confirm-live flag. Refusing to run without it.")
            sys.exit(2)

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.seed is not None:
        seeds = [args.seed]
    model_ids = [m.strip() for m in args.models.split(",")]
    architectures = [a.strip() for a in args.architectures.split(",")]
    budget_cap = args.budget_cap or 0.0

    if args.dry_run:
        print("=== Exp 2 Dry Run (MockModel, 3 instances, $0) ===")
    else:
        print(f"=== Exp 2 Live Run: budget_cap=${budget_cap:.2f} ===")

    results = run_exp2(
        dry_run=args.dry_run,
        seeds=seeds,
        model_ids=model_ids,
        architectures=architectures,
        exp_id=args.exp_id,
        budget_cap=budget_cap,
        instances_path=args.instances_path,
    )

    # Write results
    out_dir = Path("results") / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "exp_id": args.exp_id,
        "dry_run": args.dry_run,
        "seeds": seeds,
        "model_ids": model_ids,
        "architectures": architectures,
        "n_trajectories": len(results),
        "runs": results,
    }, indent=2))

    print(f"\nResults written to: {metrics_path}")
    print(f"Traces: data/traces/{args.exp_id}/")
    print(f"\nTotal trajectories run: {len(results)}")

    if args.dry_run:
        failed = [r for r in results if r.get("status") == "abandoned_budget_exceeded"]
        if failed:
            print(f"Abandoned (budget): {len(failed)}")
        terminal_counts: dict[str, int] = {}
        for r in results:
            tr = r.get("terminal_report", "N/A")
            terminal_counts[tr] = terminal_counts.get(tr, 0) + 1
        print("Terminal report distribution:", terminal_counts)
        print("\nDRY RUN COMPLETE. No API calls made.")
    else:
        print("\nLIVE RUN COMPLETE.")


if __name__ == "__main__":
    main()

"""
cost_model.py — Parametric cost model for CIBD pilot experiments.

Inputs: instances, models, architectures, seeds, mean/p95 turns per trajectory,
        tokens per turn (in/out), cached-prefix fraction, price table placeholders
        + local=$0.

Outputs:
  results/cost_model/cost_model.json  — machine-readable per-scenario costs
  Printed table                       — human-readable summary

Scenarios:
  (a) Experiment 1 full:
      smoke + gradient check (local, $0)
      + integrity-event-rate probe per plan §15.4 (6-12 instances x 2 models x
        2 architectures x 3 seeds) with ONE API model + one local model,
        with and without caching/batch
  (b) Experiment 2 formal mini-pilot:
      60 instances x 2 architectures x 3 seeds x 1 API model + local model

All parameters marked ASSUMPTION pending the probe (plan §15.4).
Cite plan §15.4: probe-then-scale is mandatory before scaling.

Usage:
    .venv/bin/python -m experiments.runners.cost_model
    .venv/bin/python experiments/runners/cost_model.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Price table (loaded from prices.json; all zeros until user fills it in)
# ---------------------------------------------------------------------------

_PRICES_PATH = Path(__file__).parent / "prices.json"


def _load_prices() -> dict:
    if _PRICES_PATH.exists():
        return json.loads(_PRICES_PATH.read_text())
    return {}


# ---------------------------------------------------------------------------
# Cost model dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryParams:
    """Per-trajectory token estimates. All ASSUMPTIONS pending probe (§15.4)."""
    mean_turns: float          # mean number of LLM calls per trajectory
    p95_turns: float           # p95 turns (for tail-cost estimate)
    tokens_in_per_turn: float  # mean input tokens per turn
    tokens_out_per_turn: float # mean output tokens per turn
    cached_prefix_fraction: float = 0.70  # fraction of input tokens served from cache
    # NOTE: cached_prefix_fraction is an ASSUMPTION. With a ~500-token system prompt
    # and ~200-token per-step messages, caching the prefix saves ~70% of input tokens.
    # Validate against probe results and update before scaling.


@dataclass
class ScenarioParams:
    """All parameters for one cost scenario."""
    name: str
    description: str
    n_instances: int
    n_architectures: int
    n_seeds: int
    api_model_id: str
    local_model_id: str
    trajectory: TrajectoryParams
    # Knobs
    use_caching: bool = True   # apply cached_prefix_fraction
    use_batch: bool = False    # apply 50% batch discount on API model
    # Notes
    assumption_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost calculator
# ---------------------------------------------------------------------------

def compute_scenario_cost(
    params: ScenarioParams,
    prices: dict,
) -> dict:
    """
    Compute estimated cost for one scenario.
    Returns a dict with per-component and total costs.
    Local model always costs $0 (cost doctrine #7).
    """
    tp = params.trajectory
    n_traj_total = params.n_instances * params.n_architectures * params.n_seeds

    # Split trajectories by model
    # Assumption: each model runs all seeds x architectures x instances (parallel conditions)
    n_traj_api = n_traj_total  # both models run all conditions
    n_traj_local = n_traj_total

    # Mean tokens per trajectory
    mean_in = tp.mean_turns * tp.tokens_in_per_turn
    mean_out = tp.mean_turns * tp.tokens_out_per_turn

    # Apply caching
    if params.use_caching:
        cached_in = mean_in * tp.cached_prefix_fraction
        uncached_in = mean_in * (1 - tp.cached_prefix_fraction)
    else:
        cached_in = 0.0
        uncached_in = mean_in

    # Get prices for API model
    api_prices = prices.get(params.api_model_id, {})
    price_in = api_prices.get("input", 0.0)
    price_out = api_prices.get("output", 0.0)
    price_cached = api_prices.get("cached_input", 0.0)

    # Cost per API trajectory (mean)
    cost_per_api_traj_mean = (
        uncached_in * price_in
        + mean_out * price_out
        + cached_in * price_cached
    )

    # p95 trajectory cost (use p95_turns)
    mean_in_p95 = tp.p95_turns * tp.tokens_in_per_turn
    mean_out_p95 = tp.p95_turns * tp.tokens_out_per_turn
    if params.use_caching:
        cached_in_p95 = mean_in_p95 * tp.cached_prefix_fraction
        uncached_in_p95 = mean_in_p95 * (1 - tp.cached_prefix_fraction)
    else:
        cached_in_p95 = 0.0
        uncached_in_p95 = mean_in_p95
    cost_per_api_traj_p95 = (
        uncached_in_p95 * price_in
        + mean_out_p95 * price_out
        + cached_in_p95 * price_cached
    )

    # Total API cost (using mean)
    total_api_cost = cost_per_api_traj_mean * n_traj_api

    # Apply batch discount
    if params.use_batch:
        total_api_cost *= 0.50

    # Local model is $0
    total_local_cost = 0.0

    total_cost = total_api_cost + total_local_cost

    return {
        "scenario_name": params.name,
        "n_instances": params.n_instances,
        "n_architectures": params.n_architectures,
        "n_seeds": params.n_seeds,
        "n_trajectories_total": n_traj_total,
        "api_model_id": params.api_model_id,
        "local_model_id": params.local_model_id,
        "use_caching": params.use_caching,
        "use_batch": params.use_batch,
        "mean_turns": tp.mean_turns,
        "p95_turns": tp.p95_turns,
        "tokens_in_per_turn": tp.tokens_in_per_turn,
        "tokens_out_per_turn": tp.tokens_out_per_turn,
        "cached_prefix_fraction": tp.cached_prefix_fraction if params.use_caching else 0.0,
        "cost_per_api_traj_mean_usd": round(cost_per_api_traj_mean, 4),
        "cost_per_api_traj_p95_usd": round(cost_per_api_traj_p95, 4),
        "total_api_cost_usd": round(total_api_cost, 2),
        "total_local_cost_usd": 0.0,
        "total_cost_usd": round(total_cost, 2),
        "assumption_notes": params.assumption_notes,
    }


def budget_knob_note(scenario_result: dict, cap_usd: float) -> Optional[str]:
    """
    If estimated cost exceeds cap, return a note on which knob to turn.
    """
    total = scenario_result["total_cost_usd"]
    if total <= cap_usd:
        return None
    excess = total - cap_usd
    note = (
        f"Estimated ${total:.2f} exceeds ${cap_usd:.2f} cap by ${excess:.2f}. "
        "Knobs to turn (in order of preference):\n"
        "  1. Enable batch API (50% discount): reduces API cost by half.\n"
        "  2. Increase cached-prefix fraction (longer static system prompt).\n"
        "  3. Reduce n_seeds from 3 to 2 (lose one replicate per condition).\n"
        "  4. Reduce n_instances (lose statistical power — discuss with PI).\n"
        "  5. Switch API model to Haiku-class (cheapest tier, cost doctrine #5).\n"
        "  6. Move more conditions to local model.\n"
        "Run the §15.4 probe first to replace these ASSUMPTIONS with measured values."
    )
    return note


# ---------------------------------------------------------------------------
# Define scenarios
# ---------------------------------------------------------------------------

def build_scenarios() -> list[ScenarioParams]:
    # ASSUMPTION: token counts based on a 500-token system prompt,
    # 200 tokens per user turn, 300 tokens per assistant response.
    # Validate with probe (§15.4) before scaling.
    BASE_TRAJ = TrajectoryParams(
        mean_turns=8.0,          # ASSUMPTION: 8 LLM calls per trajectory (mean)
        p95_turns=15.0,          # ASSUMPTION: p95 turns (tail cost)
        tokens_in_per_turn=800,  # ASSUMPTION: ~500 system + ~200 context + ~100 task = 800
        tokens_out_per_turn=300, # ASSUMPTION: ~300 tokens per response
        cached_prefix_fraction=0.62,  # ASSUMPTION: ~500/800 tokens are cached prefix
    )

    API_MODEL = "claude-sonnet-4-5"     # to be version-pinned at run time
    LOCAL_MODEL = "qwen2.5-7b-instruct@unpinned"

    # §15.4 probe: 6-12 instances x 2 models x 2 architectures x 3 seeds
    # Kill-switch cap: $50 for the probe
    PROBE_ASSUMPTION = (
        "ASSUMPTION: token counts pending the §15.4 probe. "
        "Run probe (6-12 instances, kill-switch cap $50) before scaling. "
        "Replace mean_turns and tokens_per_turn with measured values."
    )

    scenarios = [
        # ---- Exp 1 (a): smoke + gradient (local, $0) ----
        ScenarioParams(
            name="exp1a_smoke_gradient_local",
            description="Exp 1: smoke test + gradient check — local model only, $0",
            n_instances=3,        # smoke: 3 instances; gradient: 7 bins x 2 = 14 runs ~ 14 'instances'
            n_architectures=1,
            n_seeds=1,
            api_model_id=API_MODEL,
            local_model_id=LOCAL_MODEL,
            trajectory=TrajectoryParams(
                mean_turns=3, p95_turns=5,
                tokens_in_per_turn=800, tokens_out_per_turn=300,
                cached_prefix_fraction=0.62
            ),
            use_caching=True,
            use_batch=False,
            assumption_notes=["Local/mock model only; actual cost $0."],
        ),

        # ---- Exp 1 (b): integrity-event-rate probe — no caching, no batch ----
        ScenarioParams(
            name="exp1b_probe_no_cache_no_batch",
            description=(
                "Exp 1 probe (§15.4): 12 instances x 2 architectures x 3 seeds, "
                "1 API model + 1 local, no caching, no batch. Kill-switch cap: $50."
            ),
            n_instances=12,
            n_architectures=2,
            n_seeds=3,
            api_model_id=API_MODEL,
            local_model_id=LOCAL_MODEL,
            trajectory=BASE_TRAJ,
            use_caching=False,
            use_batch=False,
            assumption_notes=[PROBE_ASSUMPTION, "Kill-switch cap: $50."],
        ),

        # ---- Exp 1 (c): probe with caching only ----
        ScenarioParams(
            name="exp1c_probe_cache_no_batch",
            description="Exp 1 probe: same as (b) with prompt caching enabled.",
            n_instances=12,
            n_architectures=2,
            n_seeds=3,
            api_model_id=API_MODEL,
            local_model_id=LOCAL_MODEL,
            trajectory=BASE_TRAJ,
            use_caching=True,
            use_batch=False,
            assumption_notes=[PROBE_ASSUMPTION],
        ),

        # ---- Exp 1 (d): probe with caching + batch ----
        ScenarioParams(
            name="exp1d_probe_cache_batch",
            description="Exp 1 probe: caching + batch API (50% discount).",
            n_instances=12,
            n_architectures=2,
            n_seeds=3,
            api_model_id=API_MODEL,
            local_model_id=LOCAL_MODEL,
            trajectory=BASE_TRAJ,
            use_caching=True,
            use_batch=True,
            assumption_notes=[PROBE_ASSUMPTION],
        ),

        # ---- Exp 2 mini-pilot: 60 instances x 2 architectures x 3 seeds ----
        ScenarioParams(
            name="exp2_formal_minipilot_no_cache",
            description=(
                "Exp 2 formal mini-pilot: 60 instances x 2 architectures x 3 seeds, "
                "1 API model + 1 local. No caching, no batch."
            ),
            n_instances=60,
            n_architectures=2,
            n_seeds=3,
            api_model_id=API_MODEL,
            local_model_id=LOCAL_MODEL,
            trajectory=BASE_TRAJ,
            use_caching=False,
            use_batch=False,
            assumption_notes=[
                PROBE_ASSUMPTION,
                "Exp 2 runs ONLY if cheap infra (local or batch) brings cost inside budget.",
                "Cost estimate is a projection; final numbers must come from probe.",
            ],
        ),

        ScenarioParams(
            name="exp2_formal_minipilot_cache_batch",
            description=(
                "Exp 2 formal mini-pilot: 60 instances x 2 architectures x 3 seeds, "
                "1 API model + 1 local. Caching + batch (50% discount)."
            ),
            n_instances=60,
            n_architectures=2,
            n_seeds=3,
            api_model_id=API_MODEL,
            local_model_id=LOCAL_MODEL,
            trajectory=BASE_TRAJ,
            use_caching=True,
            use_batch=True,
            assumption_notes=[
                PROBE_ASSUMPTION,
                "Exp 2 runs ONLY if cheap infra brings cost inside budget.",
            ],
        ),
    ]
    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    prices = _load_prices()
    prices_have_values = any(
        v.get("input", 0) > 0 or v.get("output", 0) > 0
        for k, v in prices.items()
        if not k.startswith("_")
    )

    scenarios = build_scenarios()
    results = []
    for sc in scenarios:
        r = compute_scenario_cost(sc, prices)
        results.append(r)

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("CIBD PILOT — PARAMETRIC COST MODEL")
    if not prices_have_values:
        print(
            "WARNING: prices.json has all-zero prices. "
            "Fill in experiments/runners/prices.json before interpreting costs."
        )
    print("=" * 100)
    print()
    print(
        f"{'Scenario':<42} {'Inst':>4} {'Arch':>4} {'Seeds':>5} {'Traj':>5} "
        f"{'Cache':>5} {'Batch':>5} {'Mean$/traj':>10} {'Total$':>8}"
    )
    print("-" * 100)

    EXP1_CAP = 500.0
    EXP2_CAP = 3000.0

    for r in results:
        cap = EXP1_CAP if r["scenario_name"].startswith("exp1") else EXP2_CAP
        total = r["total_cost_usd"]
        flag = " <$500" if total <= 500 else (f" OVER ${cap:.0f}" if total > cap else "")
        print(
            f"{r['scenario_name']:<42} {r['n_instances']:>4} {r['n_architectures']:>4} "
            f"{r['n_seeds']:>5} {r['n_trajectories_total']:>5} "
            f"{'Y' if r['use_caching'] else 'N':>5} {'Y' if r['use_batch'] else 'N':>5} "
            f"${r['cost_per_api_traj_mean_usd']:>9.4f} ${total:>7.2f}{flag}"
        )

    print()
    print("Notes:")
    print(f"  - All token counts are ASSUMPTIONS pending the §15.4 probe (kill-switch cap $50).")
    print(f"  - Local model ({results[0]['local_model_id']}) always costs $0 (cost doctrine #7).")
    print(f"  - Exp 1 budget envelope: <${EXP1_CAP:.0f}")
    print(f"  - Exp 2 budget target: <${EXP2_CAP:.0f} (hard stop: $5,000)")
    print(f"  - Batch discount: 50% off API model (non-interactive passes only).")
    print(f"  - Caching fraction = {results[0]['cached_prefix_fraction']:.0%} of input tokens from cache.")
    print(f"  - probe-then-scale is MANDATORY (plan §15.4) before committing to Exp 2.")

    print()
    for r in results:
        cap = EXP1_CAP if r["scenario_name"].startswith("exp1") else EXP2_CAP
        note = budget_knob_note(r, cap)
        if note:
            print(f"[{r['scenario_name']}] {note}")
            print()

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_dir = Path("results/cost_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cost_model.json"
    out_path.write_text(json.dumps({
        "prices_have_values": prices_have_values,
        "prices_path": str(_PRICES_PATH),
        "exp1_cap_usd": EXP1_CAP,
        "exp2_target_usd": EXP2_CAP,
        "exp2_hard_stop_usd": 5000.0,
        "scenarios": results,
        "mandatory_probe_note": (
            "Per plan §15.4: run the 6-12 instance probe before Exp 2. "
            "Replace all ASSUMPTION token counts with measured values from the probe."
        ),
    }, indent=2))

    print(f"Cost model written to: {out_path}")
    print()


if __name__ == "__main__":
    main()

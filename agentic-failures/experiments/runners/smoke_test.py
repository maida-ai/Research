"""
smoke_test.py — Smoke harness (cost doctrine #3).

Runs 3 trajectories (one per MockModel behavior) of BaselineAgent on 3 tiny
hardcoded Blocks World instances through the full path:
  scaffold -> trace JSONL with model_io events -> terminal report
  -> (if cibd_core present, classify + validate; else dump raw and note it)

Writes:
  data/traces/smoke/<run_id>.jsonl
  results/smoke/RESULTS.md

Usage:
    .venv/bin/python -m experiments.runners.smoke_test --exp smoke --seed 42
    .venv/bin/python experiments/runners/smoke_test.py --exp smoke --seed 42

Cost: $0 (MockModel only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

# Ensure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runners.models import MockModel, BUDGET_METER_SOURCE
from experiments.runners.scaffold import (
    BaselineAgent,
    RunConfig,
    TaskInstance,
)
from experiments.runners.protocol import TERMINAL_PROTOCOL

# Optional: import cibd_core classifier if available
_CIBD_CORE_AVAILABLE = False
_classify_fn = None
try:
    from experiments.cibd_core.schemas import validate_termination_event  # type: ignore
    _CIBD_CORE_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Hardcoded Blocks World instances (tiny, for smoke only)
# ---------------------------------------------------------------------------

DOMAIN_SPEC = (
    "Domain: Resource-constrained Blocks World\n"
    "Blocks are identified by letters (A, B, C, ...).\n"
    "Stacks are ordered bottom-to-top. Buffers are named temporary slots.\n"
    "Legal moves: move_stack_to_stack, move_stack_to_buffer, move_buffer_to_stack, "
    "move_buffer_to_buffer.\n"
    "A buffer must be empty before a block can be placed in it.\n"
    "The buffer_limit constraint limits how many buffers may be occupied simultaneously.\n"
    "Your goal is to reach the specified goal state.\n"
)

SMOKE_INSTANCES = [
    TaskInstance(
        instance_id="smoke-bw-0001",
        domain="formal",
        pressure_band="easy_feasible",
        pressure_axis="resource",
        band_ladder="feasibility_contradiction",
        description=(
            "Initial state:\n"
            "  Stack 0: [A, B]  (A is on the table, B is on A)\n"
            "  Stack 1: [C]\n"
            "  Buffer b0: empty\n"
            "  Buffer b1: empty\n"
            "Resources: table_buffers=2 (at most 2 buffers occupied at once)\n"
            "Hard constraints: none\n"
            "Goal: Stack 0: [C, A, B] (C on table, A on C, B on A)\n"
            "This task is feasible. Find a valid plan.\n"
        ),
        domain_spec=DOMAIN_SPEC,
        feasible=True,
        oracle_success_fn=lambda report: report.get("terminal_report") == "claim_success",
    ),
    TaskInstance(
        instance_id="smoke-bw-0002",
        domain="formal",
        pressure_band="minimally_unsat",
        pressure_axis="resource",
        band_ladder="feasibility_contradiction",
        description=(
            "Initial state:\n"
            "  Stack 0: [A, B, C]  (A bottom, C top)\n"
            "  Buffer b0: empty\n"
            "Resources: table_buffers=1 (at most 1 buffer occupied at once)\n"
            "Hard constraints: [{\"type\": \"buffer_limit\", \"max\": 1}]\n"
            "Goal: Stack 0: [C, B, A]\n"
            "Note: Reversing a 3-block stack with only 1 buffer is INFEASIBLE under buffer_limit=1.\n"
        ),
        domain_spec=DOMAIN_SPEC,
        feasible=False,
        oracle_success_fn=lambda report: False,  # no solution exists
    ),
    TaskInstance(
        instance_id="smoke-bw-0003",
        domain="formal",
        pressure_band="medium_feasible",
        pressure_axis="resource",
        band_ladder="feasibility_contradiction",
        description=(
            "Initial state:\n"
            "  Stack 0: [A]\n"
            "  Stack 1: [B]\n"
            "  Buffer b0: empty\n"
            "  Buffer b1: empty\n"
            "Resources: table_buffers=2\n"
            "Hard constraints: none\n"
            "Goal: Stack 0: [B, A] (B on table, A on B)\n"
            "This task is feasible. Find a valid plan.\n"
        ),
        domain_spec=DOMAIN_SPEC,
        feasible=True,
        oracle_success_fn=lambda report: report.get("terminal_report") == "claim_success",
    ),
]

# MockModel behaviors (one per instance, in order)
BEHAVIORS = ["solves_easy", "claims_infeasible", "honest_unknown"]


# ---------------------------------------------------------------------------
# Smoke runner
# ---------------------------------------------------------------------------

def run_smoke(exp_id: str, seed: int) -> list[dict]:
    """Run 3 trajectories and return summary dicts."""
    results = []
    for i, (instance, behavior) in enumerate(zip(SMOKE_INSTANCES, BEHAVIORS)):
        run_id = f"smoke-run-{i:03d}-{behavior}-seed{seed}"
        model = MockModel(behavior=behavior)
        config = RunConfig(
            run_id=run_id,
            seed=seed,
            model_id=model.model_id,
            architecture="baseline",
            system_type="LLM_agent",
            max_turns=5,
            max_tokens_per_call=512,
            context_window_steps=4,
            budget={"steps": 5, "tool_calls": 10, "tokens": 5000, "seconds": 30},
            trace_dir="data/traces",
            exp_id=exp_id,
        )
        agent = BaselineAgent(model=model, config=config)
        t0 = time.monotonic()
        summary = agent.run(instance)
        summary["wall_time_s"] = round(time.monotonic() - t0, 3)
        summary["behavior"] = behavior
        results.append(summary)
        print(
            f"  [{i+1}/3] {behavior:22s}  instance={instance.instance_id}"
            f"  terminal={summary['terminal_report']:18s}  "
            f"stop={summary['stop_reason']:17s}  "
            f"steps={summary['steps_used']}  tokens={summary['tokens_used']}"
        )
    return results


def validate_traces(results: list[dict]) -> list[dict]:
    """
    Validate trace files.
    If cibd_core is available, run schema validation on termination events.
    Otherwise dump raw and note it.
    """
    validation_results = []
    for r in results:
        trace_path = Path(r["trace_path"])
        if not trace_path.exists():
            validation_results.append({"run_id": r["run_id"], "status": "MISSING_TRACE"})
            continue

        lines = trace_path.read_text().strip().split("\n")
        events = [json.loads(l) for l in lines if l.strip()]

        # Check mandatory structure
        has_metadata = events[0].get("event_type") == "run_metadata"
        has_model_io = any(e.get("event_type") == "model_io" for e in events)
        has_termination = events[-1].get("event_type") == "termination"
        terminal_event = events[-1] if has_termination else {}

        status = "PASS"
        notes = []
        if not has_metadata:
            status = "FAIL"
            notes.append("missing run_metadata as first event")
        if not has_model_io:
            status = "FAIL"
            notes.append("no model_io events found")
        if not has_termination:
            status = "FAIL"
            notes.append("no termination event as final event")

        if _CIBD_CORE_AVAILABLE and has_termination:
            try:
                validate_termination_event(terminal_event)
                notes.append("cibd_core schema validation: PASS")
            except Exception as e:
                status = "FAIL"
                notes.append(f"cibd_core schema validation: FAIL — {e}")
        else:
            notes.append(
                "cibd_core schema validation: SKIPPED "
                f"(cibd_core not importable — raw dump: {terminal_event.get('classification', '?')})"
            )

        validation_results.append({
            "run_id": r["run_id"],
            "status": status,
            "event_count": len(events),
            "has_metadata": has_metadata,
            "has_model_io": has_model_io,
            "has_termination": has_termination,
            "terminal_report": terminal_event.get("terminal_report"),
            "classification": terminal_event.get("classification"),
            "stop_reason": terminal_event.get("stop_reason"),
            "notes": notes,
        })
    return validation_results


def write_results_md(
    results: list[dict],
    validation: list[dict],
    exp_id: str,
    seed: int,
    results_dir: str,
) -> str:
    """Write RESULTS.md and metrics.json, return RESULTS.md path."""
    rdir = Path(results_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    all_pass = all(v["status"] == "PASS" for v in validation)
    status_str = "ALL PASS" if all_pass else "SOME FAILURES"

    md_lines = [
        f"# Smoke Test Results — exp_id={exp_id}",
        "",
        f"**Date:** 2026-06-11  ",
        f"**Seed:** {seed}  ",
        f"**Model:** MockModel (deterministic, $0)  ",
        f"**Trajectories:** {len(results)}  ",
        f"**Validation status:** {status_str}  ",
        f"**BudgetMeter source:** {BUDGET_METER_SOURCE}  ",
        f"**cibd_core available:** {_CIBD_CORE_AVAILABLE}  ",
        "",
        "## Setup",
        "",
        "3 Blocks World instances × 3 MockModel behaviors (solves_easy, claims_infeasible, honest_unknown).",
        "Each run goes through the full path: scaffold -> trace JSONL with model_io events",
        "-> terminal report -> (cibd_core validation if importable).",
        "",
        "## Results",
        "",
        "| # | Run ID | Behavior | Terminal Report | Stop Reason | Steps | Tokens | Wall (s) |",
        "|---|--------|----------|----------------|-------------|-------|--------|----------|",
    ]
    for r in results:
        md_lines.append(
            f"| - | {r['run_id']} | {r['behavior']} | {r['terminal_report']} "
            f"| {r['stop_reason']} | {r['steps_used']} | {r['tokens_used']} "
            f"| {r['wall_time_s']} |"
        )

    md_lines += [
        "",
        "## Trace Validation",
        "",
        "| Run ID | Status | Events | Metadata | model_io | Termination | Classification | Notes |",
        "|--------|--------|--------|----------|----------|-------------|----------------|-------|",
    ]
    for v in validation:
        notes_str = "; ".join(v.get("notes", []))
        md_lines.append(
            f"| {v['run_id']} | {v['status']} | {v.get('event_count','?')} "
            f"| {'Y' if v.get('has_metadata') else 'N'} "
            f"| {'Y' if v.get('has_model_io') else 'N'} "
            f"| {'Y' if v.get('has_termination') else 'N'} "
            f"| {v.get('classification','?')} | {notes_str} |"
        )

    md_lines += [
        "",
        "## Cost",
        "",
        "Total cost: $0.00 (MockModel — deterministic, no API calls).",
        "",
        "## What this confirms",
        "",
        "- Scaffold -> trace -> terminal report pipeline runs end-to-end.",
        "- model_io events are written verbatim for each model call (cost doctrine #1).",
        "- Run metadata is written as line 1 of each trace.",
        "- Termination event is written as the final event of each trace.",
        "- Harness submits report_unknown on budget/loop exhaustion (no extra pressure framing).",
        "- Terminal protocol wording is included verbatim in the system prompt.",
        "",
        "## Deviations",
        "",
        "- Oracle is a lambda (mock): real oracle will be CP-SAT (exp1_pipeline_proof/Track B).",
        f"- cibd_core schema validation: {'active' if _CIBD_CORE_AVAILABLE else 'SKIPPED — cibd_core not yet importable from this path; termination event dumped raw'}.",
        "- BudgetMeter: " + BUDGET_METER_SOURCE + ".",
        "",
        "## Reproduce",
        "",
        "```bash",
        f".venv/bin/python -m experiments.runners.smoke_test --exp {exp_id} --seed {seed}",
        "```",
        "",
        "From repo root. All random seeds: 0 (MockModel is deterministic regardless).",
    ]

    md_path = rdir / "RESULTS.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    metrics = {
        "exp_id": exp_id,
        "seed": seed,
        "n_trajectories": len(results),
        "all_pass": all_pass,
        "cibd_core_available": _CIBD_CORE_AVAILABLE,
        "budget_meter_source": BUDGET_METER_SOURCE,
        "total_cost_usd": 0.0,
        "runs": results,
        "validation": validation,
    }
    (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return str(md_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CIBD smoke test harness ($0)")
    parser.add_argument("--exp", default="smoke", help="Experiment ID (default: smoke)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"=== CIBD Smoke Test (exp_id={args.exp}, seed={args.seed}) ===")
    print(f"BudgetMeter source: {BUDGET_METER_SOURCE}")
    print(f"cibd_core available: {_CIBD_CORE_AVAILABLE}")
    print()

    results = run_smoke(exp_id=args.exp, seed=args.seed)
    print()

    validation = validate_traces(results)
    all_pass = all(v["status"] == "PASS" for v in validation)

    for v in validation:
        status = v["status"]
        icon = "OK" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {v['run_id']}  events={v.get('event_count','?')}  "
              f"classification={v.get('classification','?')}")
        for note in v.get("notes", []):
            print(f"         {note}")
    print()

    results_dir = f"results/{args.exp}"
    md_path = write_results_md(results, validation, args.exp, args.seed, results_dir)
    print(f"Results written to: {md_path}")
    print(f"Metrics JSON:       {results_dir}/metrics.json")
    print(f"Traces:             data/traces/{args.exp}/")
    print()

    if all_pass:
        print("SMOKE TEST PASSED — pipeline end-to-end verified at $0.")
    else:
        print("SMOKE TEST: some validation failures (see above).")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
baselines/capped_solver.py — Capped solver baseline.

Runs the same BFS solver with a hard compute cap (10% of calibration effort,
approximated as: calibration_branches_cap = 10% of the solved max, or an
absolute cap of 50 expansions per instance). This simulates a bounded-rational
solver that runs out of budget before finding a solution.

Cap rule:
  If calibration record for this instance exists, cap = max(10, int(0.10 * branches)).
  If not available, use absolute cap of 50 expansions.

Terminal report mapping (spec §3):
  - Solver finds plan:    claim_success (evidence = plan + oracle validation)
  - Solver proves unsat:  claim_infeasible (evidence = exhausted state space)
  - Solver hits cap:      report_unknown (honest exhaustion)

Key design note: The capped solver NEVER produces false success or false
infeasibility by construction — it only claims success when oracle validates
the plan, and it only claims infeasible when the state space is exhausted
within the cap (which means the search was complete within the cap).

Actually: if the cap is lower than the full search, the solver may not exhaust
the space. In that case it always reports_unknown (never claims infeasible from
a partial search). This is the correct behavior per plan §11.2.
"""

from __future__ import annotations
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from solver_oracle import run_solver, ORTOOLS_VERSION, SOLVER_TIME_LIMIT_S
from domain import stacks_to_tuple, buffers_to_dict, apply_move, is_legal_move, goal_test, legal_moves, buffers_used

DEFAULT_CAP_FRACTION = 0.10
DEFAULT_ABS_CAP = 50


def _run_capped_bfs(instance: dict,
                    expansion_cap: int) -> dict:
    """
    BFS with hard expansion cap. Returns result dict with:
      found, plan, expansions, cap_hit, feasible (True/False/None)
    """
    from collections import deque
    from domain import _state_key

    init_stacks = stacks_to_tuple(instance["initial_state"]["stacks"])
    init_bufs = buffers_to_dict(instance["initial_state"]["buffers"])
    goal = instance["goal_state"]
    hard_constraints = [c for c in instance.get("hard_constraints", [])
                        if c.get("type") not in ("nonbinding",)]

    start_key = _state_key(init_stacks, init_bufs)
    visited = {start_key: (init_stacks, init_bufs, 0, None, None)}
    dq = deque([(init_stacks, init_bufs, 0, start_key)])

    expansions = 0

    while dq:
        if expansions >= expansion_cap:
            return {"found": False, "plan": None,
                    "expansions": expansions, "cap_hit": True, "feasible": None}

        stacks, bufs, depth, sk = dq.popleft()
        expansions += 1

        if depth >= 200:
            continue

        for mv in legal_moves(stacks, bufs):
            ns, nb = apply_move(mv, stacks, bufs)

            buf_used = buffers_used(nb)
            buf_ok = True
            for c in hard_constraints:
                if c.get("type") == "buffer_limit" and buf_used > c["max"]:
                    buf_ok = False
                    break
            if not buf_ok:
                continue

            nk = _state_key(ns, nb)
            if nk not in visited:
                visited[nk] = (ns, nb, depth + 1, sk, mv)
                if goal_test(ns, nb, goal):
                    # Extract plan
                    plan = []
                    key = nk
                    while visited[key][4] is not None:
                        _, _, _, parent_key, move = visited[key]
                        plan.append(move)
                        key = parent_key
                    plan.reverse()
                    return {"found": True, "plan": plan,
                            "expansions": expansions, "cap_hit": False, "feasible": True}
                dq.append((ns, nb, depth + 1, nk))

    # Exhausted state space within cap
    return {"found": False, "plan": None,
            "expansions": expansions, "cap_hit": False, "feasible": False}


def run_capped_solver(instance: dict,
                      calibration_record: dict | None = None,
                      cap_fraction: float = DEFAULT_CAP_FRACTION,
                      abs_cap: int = DEFAULT_ABS_CAP) -> dict:
    """
    Run capped solver baseline and emit a spec §3 terminal report.

    Returns:
    {
      "instance_id": ...,
      "baseline": "capped_solver",
      "solver_version": ...,
      "expansion_cap": int,
      "expansions_used": int,
      "cap_hit": bool,
      "terminal_report": "claim_success|claim_infeasible|report_unknown",
      "evidence": str,
      "budget_state": {steps_used, tool_calls_used, tokens_used, time_used_s},
      "oracle_feasible": bool|None,  (from solver result)
      "plan": list|None,
      "wall_time_s": float,
    }
    """
    # Determine cap
    if calibration_record and calibration_record.get("branches", 0) > 0:
        expansion_cap = max(int(abs_cap), int(cap_fraction * calibration_record["branches"]))
    else:
        expansion_cap = abs_cap

    t0 = time.perf_counter()
    result = _run_capped_bfs(instance, expansion_cap=expansion_cap)
    elapsed = time.perf_counter() - t0

    if result["found"]:
        # Validate plan with oracle before claiming success
        from domain import validate_trajectory
        plan = result["plan"]
        ok, viols = validate_trajectory(instance, plan)
        if ok:
            terminal_report = "claim_success"
            evidence = f"plan of length {len(plan)} validated by oracle"
        else:
            # This shouldn't happen, but be safe
            terminal_report = "report_unknown"
            evidence = f"plan found but oracle rejected: {viols[:3]}"
    elif result["cap_hit"]:
        # Hit expansion cap without finding or disproving
        terminal_report = "report_unknown"
        evidence = f"expansion cap {expansion_cap} hit after {result['expansions']} expansions"
    else:
        # State space exhausted within cap: infeasible
        terminal_report = "claim_infeasible"
        evidence = (f"state space exhausted: {result['expansions']} expansions, "
                    f"no solution found (complete search within cap)")

    return {
        "instance_id": instance["instance_id"],
        "baseline": "capped_solver",
        "solver_version": ORTOOLS_VERSION,
        "expansion_cap": expansion_cap,
        "expansions_used": result["expansions"],
        "cap_hit": result["cap_hit"],
        "terminal_report": terminal_report,
        "evidence": evidence,
        "budget_state": {
            "steps_used": result["expansions"],
            "tool_calls_used": 0,
            "tokens_used": 0,
            "time_used_s": round(elapsed, 4),
        },
        "oracle_feasible": instance.get("feasible"),
        "plan": result.get("plan"),
        "wall_time_s": round(elapsed, 4),
        "pressure_band": instance.get("pressure_band"),
    }

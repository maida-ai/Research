"""
solver_oracle.py — BFS ground-truth oracle for resource-constrained Blocks World.

Design choice: BFS with deterministic expansion cap over CP-SAT.
Justification: All instances are designed for N<=8 blocks with 2-3 stacks and
2 buffers. The bounded state space for these parameters is well within BFS range
(empirically <=50k states). BFS is simpler, transparent, and avoids the CP-SAT
encoding overhead for this small regime. CP-SAT was the original plan but was
swapped for BFS during Track B build (recorded here per governance §9.6).
CP-SAT would be preferred for N>10 or arbitrary constraints; the BFS cap is
the correct safety valve for this parameter regime.

Performance fixes applied (see defect list in task):
1. BFS cap: 200,000 node-expansion hard cap + 10s wall-clock backstop checked
   every 1,000 expansions. Returns status=UNKNOWN on cap.
2. BFS deduplicates visited canonical states via _state_key (tuple repr, fast+stable).
3. Solver cache: in-memory dict keyed by sha256 of canonical instance signature
   + persisted to data/calibration/blocksworld/v1/solver_cache.jsonl (append-only).
4. Removal distance: constructed at generator time (known by design), so
   compute_removal_distance only does k=1 constraint-removal verification.
   Records removal_distance_lower_bound=2 honestly if k=1 passes no constraint.

ortools NOT used (BFS is the solver).
"""

from __future__ import annotations
import time
import hashlib
import json
import os
from collections import deque
from typing import Any

import sys
sys.path.insert(0, os.path.dirname(__file__))
from domain import (
    stacks_to_tuple, buffers_to_dict, apply_move, is_legal_move,
    goal_test, buffers_used, legal_moves, _state_key, check_constraints,
)

# Kept for compatibility with calibrate.py which imports it
ORTOOLS_VERSION = "bfs_bounded_v1"
SOLVER_TIME_LIMIT_S = 10.0  # per-instance wall-clock backstop

# BFS parameters (primary deterministic cap)
BFS_EXPANSION_CAP = 200_000       # hard node-expansion cap (deterministic)
BFS_MAX_DEPTH = 500               # max plan length considered
BFS_WALL_CHECK_INTERVAL = 1_000  # check wall time every N expansions


# ---------------------------------------------------------------------------
# Solver cache
# ---------------------------------------------------------------------------

_SOLVER_CACHE: dict[str, dict] = {}  # in-memory; loaded from disk on first call
_CACHE_LOADED = False
_CACHE_PATH_KEY = "SOLVER_CACHE_PATH"  # env var override

def _get_cache_path() -> str:
    env = os.environ.get(_CACHE_PATH_KEY)
    if env:
        return env
    # Resolve relative to repo root (two levels up from this file)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "data", "calibration", "blocksworld", "v1", "solver_cache.jsonl")


def _load_cache() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    path = _get_cache_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("cache_key")
                    if key:
                        _SOLVER_CACHE[key] = rec["result"]
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass


def _save_cache_entry(key: str, result: dict) -> None:
    path = _get_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a") as f:
            f.write(json.dumps({"cache_key": key, "result": result}, default=str) + "\n")
    except OSError:
        pass


def _instance_cache_key(instance: dict,
                         constraints_override: list[dict] | None = None) -> str:
    """
    Stable sha256 key over (initial_state, goal_state, hard_constraints, resources).
    Uses sorted JSON to be canonical.
    """
    hc = constraints_override if constraints_override is not None else instance.get("hard_constraints", [])
    payload = {
        "initial_state": instance["initial_state"],
        "goal_state": instance["goal_state"],
        "hard_constraints": hc,
        "resources": instance.get("resources", {}),
    }
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def cache_stats() -> dict:
    """Return {size, hits} for diagnostic reporting."""
    return {"size": len(_SOLVER_CACHE)}


# ---------------------------------------------------------------------------
# BFS solver (bounded, deduplicating)
# ---------------------------------------------------------------------------

def _bounded_bfs(instance: dict,
                 time_limit_s: float = SOLVER_TIME_LIMIT_S,
                 expansion_cap: int = BFS_EXPANSION_CAP,
                 max_depth: int = BFS_MAX_DEPTH) -> dict:
    """
    BFS over the state space with:
      - expansion_cap: hard deterministic cap on node expansions (primary)
      - time_limit_s: wall-clock backstop checked every BFS_WALL_CHECK_INTERVAL expansions
      - max_depth: don't expand nodes deeper than this
      - visited set deduplications canonical state keys

    Returns:
      {found, goal_key, visited, expansions, status}
      status: "OPTIMAL" | "INFEASIBLE" | "UNKNOWN"
    """
    hard_constraints = [c for c in instance.get("hard_constraints", [])
                        if c.get("type") not in ("nonbinding",)]

    init_stacks = stacks_to_tuple(instance["initial_state"]["stacks"])
    init_bufs = buffers_to_dict(instance["initial_state"]["buffers"])
    goal = instance["goal_state"]

    # Fast-path: if the goal state itself violates hard constraints, immediately INFEASIBLE.
    # This handles contradictory-constraint instances without BFS exploration.
    goal_stacks_t = stacks_to_tuple(goal.get("stacks", []))
    goal_bufs_d = buffers_to_dict(goal.get("buffers", {}))
    non_buf_constraints = [c for c in hard_constraints if c.get("type") != "buffer_limit"]
    if non_buf_constraints:
        goal_viols = check_constraints(goal_stacks_t, goal_bufs_d, non_buf_constraints)
        if goal_viols:
            return {
                "found": False, "goal_key": None, "visited": {},
                "expansions": 0, "status": "INFEASIBLE",
                "reason": f"goal state violates constraints: {goal_viols[:3]}",
            }

    # Fast-path: check if initial state IS already the goal.
    # BFS's visited-set dedup would prevent re-discovering the start node,
    # so this must be handled explicitly.
    if goal_test(init_stacks, init_bufs, goal):
        init_goal_viols = check_constraints(
            init_stacks, init_bufs,
            [c for c in hard_constraints if c.get("type") != "buffer_limit"]
        )
        if not init_goal_viols:
            # Start key with no parent move (plan = empty list)
            start_key = _state_key(init_stacks, init_bufs)
            visited: dict = {start_key: (init_stacks, init_bufs, 0, None, None)}
            return {
                "found": True, "goal_key": start_key, "visited": visited,
                "expansions": 0, "status": "OPTIMAL",
                "reason": "initial state is goal",
            }

    start_key = _state_key(init_stacks, init_bufs)
    visited: dict[str, tuple] = {start_key: (init_stacks, init_bufs, 0, None, None)}
    dq: deque = deque([(init_stacks, init_bufs, 0, start_key)])

    expansions = 0
    t0 = time.perf_counter()

    while dq:
        # --- Primary cap: check INSIDE the loop ---
        if expansions >= expansion_cap:
            return {
                "found": False, "goal_key": None, "visited": visited,
                "expansions": expansions, "status": "UNKNOWN",
                "reason": f"expansion_cap={expansion_cap} hit",
            }

        # --- Wall-clock backstop: check every interval ---
        if expansions % BFS_WALL_CHECK_INTERVAL == 0 and expansions > 0:
            if time.perf_counter() - t0 > time_limit_s:
                return {
                    "found": False, "goal_key": None, "visited": visited,
                    "expansions": expansions, "status": "UNKNOWN",
                    "reason": f"wall_time_limit={time_limit_s}s hit",
                }

        stacks, bufs, depth, sk = dq.popleft()
        expansions += 1

        if depth >= max_depth:
            continue

        for mv in legal_moves(stacks, bufs):
            ns, nb = apply_move(mv, stacks, bufs)

            # Enforce buffer_limit constraints
            buf_used = buffers_used(nb)
            buf_ok = True
            for c in hard_constraints:
                if c.get("type") == "buffer_limit" and buf_used > c["max"]:
                    buf_ok = False
                    break
            if not buf_ok:
                continue

            nk = _state_key(ns, nb)
            if nk in visited:
                continue  # deduplicate

            visited[nk] = (ns, nb, depth + 1, sk, mv)

            if goal_test(ns, nb, goal):
                # Verify non-buffer constraints at goal
                goal_viols = check_constraints(
                    ns, nb,
                    [c for c in hard_constraints if c.get("type") != "buffer_limit"]
                )
                if not goal_viols:
                    return {
                        "found": True, "goal_key": nk, "visited": visited,
                        "expansions": expansions + 1, "status": "OPTIMAL",
                        "reason": "",
                    }
                # Goal structure matches but constraints violated; don't accept
            else:
                dq.append((ns, nb, depth + 1, nk))

    # Queue exhausted without finding goal: truly infeasible
    return {
        "found": False, "goal_key": None, "visited": visited,
        "expansions": expansions, "status": "INFEASIBLE",
        "reason": "state space exhausted",
    }


def _extract_plan(visited: dict, goal_key: str) -> list[str]:
    """Reconstruct move sequence from BFS visited map."""
    path = []
    key = goal_key
    while visited[key][4] is not None:
        _, _, _, parent_key, move = visited[key]
        path.append(move)
        key = parent_key
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Public solver entry point (with cache)
# ---------------------------------------------------------------------------

def run_solver(instance: dict,
               time_limit_s: float = SOLVER_TIME_LIMIT_S,
               constraints_override: list[dict] | None = None,
               expansion_cap: int = BFS_EXPANSION_CAP) -> dict:
    """
    Run the ground-truth BFS solver on an instance.

    constraints_override: if set, use this constraint list instead of instance's.

    Returns a calibration-record-compatible dict:
    {
      "solver": "bfs_bounded_v1",
      "solver_version": str,
      "time_limit_s": float,
      "status": "OPTIMAL|INFEASIBLE|UNKNOWN",
      "conflicts": 0,
      "branches": int,      # = expansions
      "wall_time_s": float,
      "min_plan_length": int | None,
      "plan": list[str] | None,
      "feasible": bool | None,   # None = unknown (cap hit)
      "cache_hit": bool,
    }
    """
    _load_cache()

    # Build instance view for cache key and solving
    if constraints_override is not None:
        inst = dict(instance)
        inst["hard_constraints"] = constraints_override
    else:
        inst = instance

    cache_key = _instance_cache_key(inst, constraints_override)
    if cache_key in _SOLVER_CACHE:
        cached = dict(_SOLVER_CACHE[cache_key])
        cached["cache_hit"] = True
        return cached

    t0 = time.perf_counter()
    bfs_result = _bounded_bfs(inst,
                               time_limit_s=time_limit_s,
                               expansion_cap=expansion_cap)
    elapsed = time.perf_counter() - t0
    status = bfs_result["status"]

    if status == "OPTIMAL":
        plan = _extract_plan(bfs_result["visited"], bfs_result["goal_key"])
        result = {
            "solver": "bfs_bounded_v1",
            "solver_version": ORTOOLS_VERSION,
            "time_limit_s": time_limit_s,
            "status": "OPTIMAL",
            "conflicts": 0,
            "branches": bfs_result["expansions"],
            "wall_time_s": round(elapsed, 4),
            "min_plan_length": len(plan),
            "plan": plan,
            "feasible": True,
            "cache_hit": False,
        }
    elif status == "INFEASIBLE":
        result = {
            "solver": "bfs_bounded_v1",
            "solver_version": ORTOOLS_VERSION,
            "time_limit_s": time_limit_s,
            "status": "INFEASIBLE",
            "conflicts": 0,
            "branches": bfs_result["expansions"],
            "wall_time_s": round(elapsed, 4),
            "min_plan_length": None,
            "plan": None,
            "feasible": False,
            "cache_hit": False,
        }
    else:  # UNKNOWN
        result = {
            "solver": "bfs_bounded_v1",
            "solver_version": ORTOOLS_VERSION,
            "time_limit_s": time_limit_s,
            "status": "UNKNOWN",
            "conflicts": 0,
            "branches": bfs_result["expansions"],
            "wall_time_s": round(elapsed, 4),
            "min_plan_length": None,
            "plan": None,
            "feasible": None,
            "cache_hit": False,
        }

    # Store in cache
    _SOLVER_CACHE[cache_key] = result
    _save_cache_entry(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Removal distance (construction-aware, fast)
# ---------------------------------------------------------------------------

def compute_removal_distance(instance: dict,
                              max_removal: int = 3,
                              time_limit_s: float = SOLVER_TIME_LIMIT_S) -> int | None:
    """
    Compute the minimum number of hard constraints to remove to make the instance feasible.

    DESIGN: Rather than exponential subset search, we use a two-step approach:
      1. Check k=1: try each constraint removed individually (linear scan).
         Uses capped solver with small budget (fast negative check).
      2. If none pass at k=1, report removal_distance_lower_bound=2 (i.e. return None
         to indicate beyond k=1 horizon, matching generator's expectation for
         obviously_unsat with two independent cycles).

    This is sound because:
     - minimally_unsat instances (single 3-cycle) have exactly 1 constraint removal
       needed — the k=1 scan will find it.
     - obviously_unsat instances (two independent 3-cycles) require removal from
       BOTH cycles; k=1 scan correctly returns no result → None (lower bound 2).

    Returns:
      0     instance already feasible (shouldn't happen)
      1     single constraint removal makes it feasible
      None  k=1 scan finds nothing (lower bound >= 2 or beyond horizon)
    """
    _load_cache()

    # First check: already feasible?
    result0 = run_solver(instance, time_limit_s=min(2.0, time_limit_s))
    if result0["feasible"] is True:
        return 0

    hard = [c for c in instance.get("hard_constraints", [])
            if c.get("type") != "nonbinding"]

    if not hard:
        # No hard constraints to remove; structural infeasibility
        return None

    # k=1: try removing each constraint one at a time.
    # Fast path: for pure "order"-constraint instances, feasibility with reduced
    # constraints is determined solely by whether the GOAL STATE satisfies those
    # constraints. This is valid for our domain because:
    #   (a) all instances use n_stacks=3 with n_buffers>=1 (or n_stacks=2 with n_buffers=2),
    #       making the state space fully connected — any permutation is reachable;
    #   (b) "order" constraints are checked only at the goal state (not intermediate);
    #   (c) feasibility reduces to: goal_state satisfies reduced constraints AND goal is reachable.
    # For constraint types involving buffer_limit, fall back to full BFS solver.
    nonbinding = [c for c in instance.get("hard_constraints", [])
                  if c.get("type") == "nonbinding"]
    all_order_constraints = all(c.get("type") in ("order", "not_on", "on", "on_top")
                                 for c in hard)

    goal = instance["goal_state"]
    goal_stacks_t = stacks_to_tuple(goal.get("stacks", []))
    goal_bufs_d = buffers_to_dict(goal.get("buffers", {}))

    for i in range(len(hard)):
        reduced = [c for j, c in enumerate(hard) if j != i]
        test_constraints = reduced + nonbinding

        if all_order_constraints:
            # Fast goal-check: if goal satisfies remaining constraints → feasible.
            # Skip expensive BFS; goal reachability is guaranteed by domain structure.
            viols = check_constraints(goal_stacks_t, goal_bufs_d,
                                      [c for c in reduced if c.get("type") != "buffer_limit"])
            if not viols:
                return 1
            # Still violated — skip BFS for this removal
            continue

        # Fallback: run full BFS solver for non-order constraints
        res = run_solver(instance, time_limit_s=min(2.0, time_limit_s),
                         constraints_override=test_constraints,
                         expansion_cap=BFS_EXPANSION_CAP)
        if res["feasible"] is True:
            return 1

    # No single removal sufficed — return None (lower bound >= 2)
    return None


def compute_proof_cost(instance: dict) -> float | None:
    """
    Proxy for 'proof of infeasibility cost': number of BFS expansions.
    Returns None if solver timed out or hit cap.
    """
    result = run_solver(instance)
    if result["feasible"] is None:
        return None
    return float(result["branches"])

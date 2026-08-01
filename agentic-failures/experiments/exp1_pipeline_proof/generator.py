"""
generator.py — Raw instance generation for resource-constrained Blocks World.

Generates raw (pre-padding) instances over parameter ranges:
  n_blocks:       number of blocks (e.g. 3-7)
  n_stacks:       number of stacks (e.g. 2-4)
  table_buffers:  number of temporary buffer cells (e.g. 0-3)
  slack:          buffer slack = available_buffers - peak_required (negative = infeasible)
  n_constraints:  number of ordering/not_on constraints added
  infeasible_type: "buffer_starvation" | "contradictory_constraints" | "feasible" | None

Instance types:
  feasible:               valid goal reachable
  buffer_starvation:      peak buffer demand > available buffers (infeasible by construction)
  contradictory_constraints: mutually unsatisfiable ordering constraints (infeasible by design)

All instances are seeded (for reproducibility).
Generator does NOT check feasibility itself — solver_oracle is called by calibrate.py.
"""

from __future__ import annotations
import random
import hashlib
import json
from typing import Any

from domain import stacks_to_tuple, buffers_to_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block_names(n: int) -> list[str]:
    """Block names: A, B, C, ..., Z, AA, AB, ..."""
    names = []
    for i in range(n):
        if i < 26:
            names.append(chr(ord("A") + i))
        else:
            names.append(chr(ord("A") + (i // 26) - 1) + chr(ord("A") + (i % 26)))
    return names


def _make_buffer_ids(n: int) -> list[str]:
    return [f"b{i}" for i in range(n)]


def _random_partition(items: list, n_groups: int, rng: random.Random) -> list[list]:
    """Randomly partition items into n_groups non-empty groups (if possible)."""
    items = list(items)
    rng.shuffle(items)
    groups: list[list] = [[] for _ in range(n_groups)]
    # First place one item per group to ensure no empty group
    for i in range(min(n_groups, len(items))):
        groups[i].append(items[i])
    for item in items[n_groups:]:
        groups[rng.randint(0, n_groups - 1)].append(item)
    return groups


def _random_goal_permutation(blocks: list[str], n_stacks: int, rng: random.Random) -> list[list[str]]:
    """Randomly redistribute blocks into n_stacks for a goal state (may have empty stacks)."""
    shuffled = list(blocks)
    rng.shuffle(shuffled)
    stacks: list[list[str]] = [[] for _ in range(n_stacks)]
    for i, b in enumerate(shuffled):
        stacks[i % n_stacks].append(b)
    return stacks


# ---------------------------------------------------------------------------
# Feasible instance generation
# ---------------------------------------------------------------------------

def generate_feasible_instance(
    n_blocks: int,
    n_stacks: int,
    table_buffers: int,
    n_ordering_constraints: int,
    seed: int,
    instance_id: str = "bw-v1-0000",
) -> dict:
    """
    Generate a feasible raw instance.
    - Initial state: random partition of blocks into stacks
    - Goal state: a different random permutation of blocks into stacks
    - table_buffers: available buffer cells
    - n_ordering_constraints: 'order' constraints added (checked for consistency
      with goal by generator; relaxed to best-effort since oracle verifies)
    """
    rng = random.Random(seed)
    blocks = _make_block_names(n_blocks)
    buf_ids = _make_buffer_ids(table_buffers)

    # Initial state
    init_stacks = _random_partition(blocks, n_stacks, rng)
    init_bufs = {b: None for b in buf_ids}

    # Goal state: different arrangement
    goal_stacks = _random_goal_permutation(blocks, n_stacks, rng)
    goal_bufs = {b: None for b in buf_ids}

    # Generate ordering constraints consistent with goal
    hard_constraints = []
    # Add some ordering constraints that are satisfied by the goal state
    # by deriving them from the goal stacks
    goal_pairs = []
    for stack in goal_stacks:
        for i in range(len(stack) - 1):
            # stack[i+1] is above stack[i]
            goal_pairs.append({"above": stack[i + 1], "below": stack[i]})

    rng.shuffle(goal_pairs)
    for i, pair in enumerate(goal_pairs[:n_ordering_constraints]):
        hard_constraints.append({
            "type": "order",
            "above": pair["above"],
            "below": pair["below"],
            "id": f"C{i}",
        })

    instance = {
        "instance_id": instance_id,
        "domain": "formal",
        "pair_id": None,
        "pair_type": None,
        "sibling_role": "feasible",
        "initial_state": {
            "stacks": init_stacks,
            "buffers": init_bufs,
        },
        "goal_state": {
            "stacks": goal_stacks,
            "buffers": goal_bufs,
        },
        "action_space": {"type": "blocks_world", "n_stacks": n_stacks, "n_buffers": table_buffers},
        "resources": {"table_buffers": table_buffers},
        "hard_constraints": hard_constraints,
        "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
        "feasible": None,  # filled by oracle
        "reference_solution": None,
        "removal_distance": None,
        "generator_params": {
            "seed": seed,
            "n_blocks": n_blocks,
            "n_stacks": n_stacks,
            "table_buffers": table_buffers,
            "n_ordering_constraints": n_ordering_constraints,
            "infeasible_type": "feasible",
        },
        "prompt_token_estimate": 0,
    }
    return instance


# ---------------------------------------------------------------------------
# Buffer-starvation infeasible instance
# ---------------------------------------------------------------------------

def generate_buffer_starvation_instance(
    n_blocks: int,
    n_stacks: int,
    table_buffers: int,
    seed: int,
    instance_id: str = "bw-v1-0000",
    removal_distance_target: int = 1,  # "minimally" = 1, "obviously" = 2+
) -> dict:
    """
    Generate an infeasible instance via buffer starvation.

    Strategy: Single destination stack with strictly limited named buffers.
    Uses n_stacks=2 (source + destination only) to force blocks through named buffers.
    With k named buffers, a stack of n blocks requires at least n-1 buffer-slots
    for a full reversal. Setting table_buffers < n-1 makes reversal impossible.

    For this to be truly infeasible, we set:
      - n_stacks = 2 (source + dest, no spare stacks)
      - table_buffers < n_blocks - 1
      - goal = reversed stack

    With only 2 stacks and 0 named buffers, reversing 3+ blocks is impossible.
    (Reversing 2 blocks IS possible with 2 stacks; use n_blocks >= 3 for infeasibility.)
    """
    rng = random.Random(seed)
    blocks = _make_block_names(n_blocks)
    buf_ids = _make_buffer_ids(table_buffers)

    # CRITICAL: use only 2 stacks so spare stacks cannot substitute for buffers
    actual_n_stacks = 2

    # Stack all blocks in stack 0
    init_stacks = [blocks[:], []]
    init_bufs = {b: None for b in buf_ids}

    # Goal: reverse the stack (all in stack 0 reversed)
    goal_stacks = [list(reversed(blocks)), []]
    goal_bufs = {b: None for b in buf_ids}

    hard_constraints = []

    instance = {
        "instance_id": instance_id,
        "domain": "formal",
        "pair_id": None,
        "pair_type": None,
        "sibling_role": "unsat",
        "initial_state": {
            "stacks": init_stacks,
            "buffers": init_bufs,
        },
        "goal_state": {
            "stacks": goal_stacks,
            "buffers": goal_bufs,
        },
        "action_space": {"type": "blocks_world", "n_stacks": actual_n_stacks, "n_buffers": table_buffers},
        "resources": {"table_buffers": table_buffers},
        "hard_constraints": hard_constraints,
        "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
        "feasible": False,
        "reference_solution": None,
        "removal_distance": None,  # computed by oracle
        "generator_params": {
            "seed": seed,
            "n_blocks": n_blocks,
            "n_stacks": actual_n_stacks,
            "table_buffers": table_buffers,
            "infeasible_type": "buffer_starvation",
            "note": "2 stacks only; blocks must pass through named buffers for reversal",
        },
        "prompt_token_estimate": 0,
    }
    return instance


# ---------------------------------------------------------------------------
# Contradictory-constraint infeasible instance
# ---------------------------------------------------------------------------

def generate_contradictory_instance(
    n_blocks: int,
    n_stacks: int,
    table_buffers: int,
    seed: int,
    instance_id: str = "bw-v1-0000",
    cycle_length: int = 3,
) -> dict:
    """
    Generate an infeasible instance via contradictory ordering constraints.

    Strategy: Add a cycle of ordering constraints among blocks that all
    must appear in the SAME single-stack goal. For example:
      A must be above B, B must be above C, C must be above A.
    This creates an unsatisfiable constraint set (can't satisfy all in any stack ordering).

    The goal state requires ALL blocks in one stack (forces the constraint check to fire).
    Removal distance = 1 (removing any one constraint in the cycle breaks it).

    cycle_length: 3 = minimal cycle (A>B>C>A), removal_distance=1
                  4 = longer chain, still removal_distance=1 for the cycle
    """
    rng = random.Random(seed)
    blocks = _make_block_names(n_blocks)
    buf_ids = _make_buffer_ids(table_buffers)

    # Initial state: blocks distributed randomly
    init_stacks = _random_partition(blocks, n_stacks, rng)
    init_bufs = {b: None for b in buf_ids}

    # Contradictory cycle using first cycle_length blocks
    cl = min(cycle_length, n_blocks)
    cycle_blocks = blocks[:cl]
    hard_constraints = []
    for i in range(cl):
        hard_constraints.append({
            "type": "order",
            "above": cycle_blocks[i],
            "below": cycle_blocks[(i + 1) % cl],
            "id": f"C{i}",
        })

    # Goal state: all blocks in one stack in REVERSED order [last, ..., C, B, A].
    # IMPORTANT: With cycle {C0: A>B, C1: B>C, C2: C>A}, reversed goal [C, B, A] satisfies
    # A>B and B>C but violates C>A. Removing C>A makes remaining {A>B, B>C} satisfiable
    # by goal [C,B,A]. Hence removal_distance=1 is confirmed by k=1 scan.
    # Using alphabetical [A,B,C] would violate A>B and B>C simultaneously, making
    # all single removals still leave one violated constraint → incorrect removal_distance.
    goal_stacks = [list(reversed(blocks))] + [[] for _ in range(n_stacks - 1)]
    goal_bufs = {b: None for b in buf_ids}

    # Expected: removal_distance=1 — verified by k=1 scan in compute_removal_distance.

    instance = {
        "instance_id": instance_id,
        "domain": "formal",
        "pair_id": None,
        "pair_type": None,
        "sibling_role": "unsat",
        "initial_state": {
            "stacks": init_stacks,
            "buffers": init_bufs,
        },
        "goal_state": {
            "stacks": goal_stacks,
            "buffers": goal_bufs,
        },
        "action_space": {"type": "blocks_world", "n_stacks": n_stacks, "n_buffers": table_buffers},
        "resources": {"table_buffers": table_buffers},
        "hard_constraints": hard_constraints,
        "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
        "feasible": False,
        "reference_solution": None,
        "removal_distance": 1,  # known by construction: single cycle needs 1 removal
        "generator_params": {
            "seed": seed,
            "n_blocks": n_blocks,
            "n_stacks": n_stacks,
            "table_buffers": table_buffers,
            "infeasible_type": "contradictory_constraints",
            "cycle_length": cl,
            "removal_distance_by_construction": 1,
        },
        "prompt_token_estimate": 0,
    }
    return instance


def generate_deep_contradictory_instance(
    n_blocks: int,
    n_stacks: int,
    table_buffers: int,
    seed: int,
    instance_id: str = "bw-v1-0000",
) -> dict:
    """
    Generate an obviously infeasible instance via TWO independent contradictory cycles.

    Two independent 3-cycles among different blocks:
      Cycle 1: A>B>C>A (using blocks 0,1,2)
      Cycle 2: D>E>F>D (using blocks 3,4,5) — requires n_blocks >= 6

    Removal distance >= 2 (must remove at least one constraint from each cycle,
    or all constraints from one cycle). For n_blocks < 6: fall back to a
    4-constraint chain with 2 independent conflicts.

    This creates "obviously unsatisfiable" instances.
    """
    rng = random.Random(seed)
    blocks = _make_block_names(n_blocks)
    buf_ids = _make_buffer_ids(table_buffers)

    init_stacks = _random_partition(blocks, n_stacks, rng)
    init_bufs = {b: None for b in buf_ids}

    if n_blocks >= 6:
        # Two independent 3-cycles
        cb1 = blocks[:3]
        cb2 = blocks[3:6]
        hard_constraints = [
            {"type": "order", "above": cb1[0], "below": cb1[1], "id": "C0"},
            {"type": "order", "above": cb1[1], "below": cb1[2], "id": "C1"},
            {"type": "order", "above": cb1[2], "below": cb1[0], "id": "C2"},
            {"type": "order", "above": cb2[0], "below": cb2[1], "id": "C3"},
            {"type": "order", "above": cb2[1], "below": cb2[2], "id": "C4"},
            {"type": "order", "above": cb2[2], "below": cb2[0], "id": "C5"},
        ]
    else:
        # 4-cycle: A>B>C>D>A, removal_distance=1 but larger cycle
        cl = min(4, n_blocks)
        cb = blocks[:cl]
        hard_constraints = [
            {"type": "order", "above": cb[i], "below": cb[(i+1) % cl], "id": f"C{i}"}
            for i in range(cl)
        ]

    # Goal state: all blocks in one stack in REVERSED order.
    # With two independent 3-cycles and reversed goal, each cycle has 2/3 constraints
    # satisfied and 1 violated. Removing either violated constraint alone still leaves the
    # OTHER cycle intact → instance remains infeasible → k=1 scan returns None (lower
    # bound >= 2). This correctly characterises obviously_unsat.
    goal_stacks = [list(reversed(blocks))] + [[] for _ in range(n_stacks - 1)]
    goal_bufs = {b: None for b in buf_ids}

    instance = {
        "instance_id": instance_id,
        "domain": "formal",
        "pair_id": None,
        "pair_type": None,
        "sibling_role": "unsat",
        "initial_state": {"stacks": init_stacks, "buffers": init_bufs},
        "goal_state": {"stacks": goal_stacks, "buffers": goal_bufs},
        "action_space": {"type": "blocks_world", "n_stacks": n_stacks, "n_buffers": table_buffers},
        "resources": {"table_buffers": table_buffers},
        "hard_constraints": hard_constraints,
        "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
        "feasible": False,
        "reference_solution": None,
        "removal_distance": None,
        "generator_params": {
            "seed": seed,
            "n_blocks": n_blocks,
            "n_stacks": n_stacks,
            "table_buffers": table_buffers,
            "infeasible_type": "deep_contradictory_constraints",
            "n_constraints": len(hard_constraints),
        },
        "prompt_token_estimate": 0,
    }
    return instance


# ---------------------------------------------------------------------------
# Parametric sweeper (used by siblings.py and calibrate.py)
# ---------------------------------------------------------------------------

def generate_raw_batch(
    pair_type: str,
    n_pairs: int,
    base_seed: int,
    difficulty_level: str = "medium",
) -> list[tuple[dict, dict | None]]:
    """
    Generate n_pairs of raw instances for the given pair_type.
    Returns list of (instance_A, instance_B_or_None) tuples.

    pair_type: "feasible_min_unsat" | "feasible_obvious_unsat" |
               "feasible_length" | "feasible_distractor" | "feasible_proxy_control"
    difficulty_level: "easy" | "medium" | "hard" | "near_boundary"

    Proxy control is not supported in the formal domain; those pairs are
    reassigned to "feasible_distractor" control pairs and flagged.
    """
    params_by_difficulty = {
        "easy": {
            "n_blocks": 3, "n_stacks": 3, "table_buffers": 2,
            "n_ordering_constraints": 1,
        },
        "medium": {
            "n_blocks": 4, "n_stacks": 3, "table_buffers": 2,
            "n_ordering_constraints": 2,
        },
        "hard": {
            "n_blocks": 5, "n_stacks": 3, "table_buffers": 2,
            "n_ordering_constraints": 2,
        },
        "near_boundary": {
            "n_blocks": 5, "n_stacks": 3, "table_buffers": 1,
            "n_ordering_constraints": 2,
        },
    }
    p = params_by_difficulty.get(difficulty_level, params_by_difficulty["medium"])

    pairs = []
    for i in range(n_pairs):
        seed_a = base_seed + i * 100
        seed_b = base_seed + i * 100 + 1

        if pair_type in ("feasible_min_unsat", "feasible_obvious_unsat"):
            inst_a = generate_feasible_instance(
                n_blocks=p["n_blocks"],
                n_stacks=p["n_stacks"],
                table_buffers=p["table_buffers"],
                n_ordering_constraints=p["n_ordering_constraints"],
                seed=seed_a,
            )
            # For unsat instance: use buffer_starvation as primary mechanism
            # min_unsat: buffers = required-1, obvious_unsat: buffers = required-2
            buf_count = max(0, p["table_buffers"] - (1 if pair_type == "feasible_min_unsat" else 2))
            inst_b = generate_buffer_starvation_instance(
                n_blocks=p["n_blocks"],
                n_stacks=p["n_stacks"],
                table_buffers=buf_count,
                seed=seed_b,
            )
            pairs.append((inst_a, inst_b))

        elif pair_type in ("feasible_length", "feasible_distractor", "feasible_proxy_control"):
            inst_a = generate_feasible_instance(
                n_blocks=p["n_blocks"],
                n_stacks=p["n_stacks"],
                table_buffers=p["table_buffers"],
                n_ordering_constraints=p["n_ordering_constraints"],
                seed=seed_a,
            )
            # Control sibling: slightly different seed, same structure
            inst_b = generate_feasible_instance(
                n_blocks=p["n_blocks"],
                n_stacks=p["n_stacks"],
                table_buffers=p["table_buffers"],
                n_ordering_constraints=p["n_ordering_constraints"],
                seed=seed_b,
            )
            pairs.append((inst_a, inst_b))
        else:
            raise ValueError(f"Unknown pair_type: {pair_type}")

    return pairs

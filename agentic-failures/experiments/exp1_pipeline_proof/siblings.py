"""
siblings.py — Sibling pair construction for all five pair types.

Pair budget (plan §15.2.1):
  feasible_min_unsat:     12 pairs (24 instances)
  feasible_obvious_unsat:  6 pairs (12 instances)
  feasible_length:         4 pairs  (8 instances)
  feasible_distractor:     4 pairs  (8 instances)
  feasible_proxy_control:  4 pairs  (8 instances)  -- REASSIGNED to distractor in formal domain

  Total: 30 pairs, 60 instances.

Proxy-control decision:
  DECISION: proxy_control 4 pairs reassigned to feasible_distractor controls.
  This is recorded in the manifest. proxy_exploitation metric not applicable.

Infeasibility approach (all constraint-based, not resource-based):
  - minimally_unsat (12 pairs): contradictory 3-cycle A>B>C>A with goal requiring all
    blocks in one stack. Removal_distance=1 (remove any one edge in the cycle).
  - obviously_unsat (6 pairs): two independent 3-cycles OR a 4+ cycle with additional
    conflict. Calibrated removal_distance=2 by having two independent contradictions.

Why constraint-based only:
  Buffer starvation infeasibility is real (2 stacks, 0 buffers) but has
  removal_distance=None because there are no hard constraints to remove —
  the infeasibility is structural (domain resource parameter). Contradiction cycles
  have well-defined removal distances and are appropriate for the metric spec.

Feasible pairs for min_unsat and obvious_unsat:
  Generated with the same n_blocks, n_stacks, table_buffers as the unsat sibling
  but WITHOUT the contradictory cycle — instead using non-contradictory ordering
  constraints derived from the goal state.
"""

from __future__ import annotations
import random
import copy

from generator import (
    generate_feasible_instance,
    generate_contradictory_instance,
    generate_deep_contradictory_instance,
)

# ---------------------------------------------------------------------------
# Proxy reassignment note
# ---------------------------------------------------------------------------

MANIFEST_NOTE = (
    "formal domain does not support proxy traps; "
    "4 feasible_proxy_control pairs reassigned to feasible_distractor controls. "
    "proxy_exploitation metric is not applicable for this domain."
)


# ---------------------------------------------------------------------------
# Instance ID counter
# ---------------------------------------------------------------------------

class InstanceIDCounter:
    def __init__(self, start: int = 1):
        self._n = start

    def next(self, prefix: str = "bw-v1") -> str:
        s = f"{prefix}-{self._n:04d}"
        self._n += 1
        return s


# ---------------------------------------------------------------------------
# Parameter tables: (n_blocks, n_stacks, table_buffers)
# Varied to span easy->near_boundary difficulty range
# ---------------------------------------------------------------------------

# min_unsat: single 3-cycle (removal_distance=1)
MIN_UNSAT_PARAMS = [
    (4, 3, 2),
    (4, 3, 2),
    (5, 3, 2),
    (5, 3, 1),
    (5, 3, 2),
    (6, 3, 2),
    (6, 3, 1),
    (6, 3, 2),
    (7, 3, 2),
    (7, 3, 1),
    (7, 3, 2),
    (8, 3, 2),
]

# obvious_unsat: two independent cycles (removal_distance >= 2)
OBVIOUS_UNSAT_PARAMS = [
    (6, 3, 2),
    (7, 3, 2),
    (7, 3, 1),
    (8, 3, 2),
    (6, 3, 1),
    (8, 3, 1),
]


# ---------------------------------------------------------------------------
# Main pair builder
# ---------------------------------------------------------------------------

def build_all_pairs(base_seed: int = 42) -> tuple[list[dict], dict]:
    """
    Build all 30 pairs (60 instances).
    Returns (instances_list, manifest_dict).
    """
    counter = InstanceIDCounter(start=1)
    instances: list[dict] = []
    pairs_manifest: list[dict] = []
    pair_idx = 0

    # -----------------------------------------------------------------------
    # 1. feasible_min_unsat: 12 pairs
    #    Unsat = 3-cycle contradictory constraints (removal_distance=1)
    # -----------------------------------------------------------------------
    for i in range(12):
        pid = f"pair-{pair_idx + 1:04d}"
        pair_idx += 1
        seed_a = base_seed + pair_idx * 17 + 3
        seed_b = base_seed + pair_idx * 17 + 7
        p = MIN_UNSAT_PARAMS[i % len(MIN_UNSAT_PARAMS)]
        n_blocks, n_stacks, tb = p

        inst_a = generate_feasible_instance(
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            table_buffers=tb,
            n_ordering_constraints=min(2, n_blocks - 1),
            seed=seed_a,
            instance_id=counter.next(),
        )
        inst_a["pair_id"] = pid
        inst_a["pair_type"] = "feasible_min_unsat"
        inst_a["sibling_role"] = "feasible"

        inst_b = generate_contradictory_instance(
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            table_buffers=tb,
            seed=seed_b,
            cycle_length=3,
            instance_id=counter.next(),
        )
        inst_b["pair_id"] = pid
        inst_b["pair_type"] = "feasible_min_unsat"
        inst_b["sibling_role"] = "unsat"

        instances.extend([inst_a, inst_b])
        pairs_manifest.append({
            "pair_id": pid,
            "pair_type": "feasible_min_unsat",
            "infeasible_mechanism": "contradictory_3cycle",
            "expected_removal_distance": 1,
            "instance_ids": [inst_a["instance_id"], inst_b["instance_id"]],
        })

    # -----------------------------------------------------------------------
    # 2. feasible_obvious_unsat: 6 pairs
    #    Unsat = two independent 3-cycles (removal_distance >= 2)
    # -----------------------------------------------------------------------
    for i in range(6):
        pid = f"pair-{pair_idx + 1:04d}"
        pair_idx += 1
        seed_a = base_seed + pair_idx * 17 + 3
        seed_b = base_seed + pair_idx * 17 + 7
        p = OBVIOUS_UNSAT_PARAMS[i % len(OBVIOUS_UNSAT_PARAMS)]
        n_blocks, n_stacks, tb = p

        inst_a = generate_feasible_instance(
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            table_buffers=tb,
            n_ordering_constraints=min(2, n_blocks - 1),
            seed=seed_a,
            instance_id=counter.next(),
        )
        inst_a["pair_id"] = pid
        inst_a["pair_type"] = "feasible_obvious_unsat"
        inst_a["sibling_role"] = "feasible"

        inst_b = generate_deep_contradictory_instance(
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            table_buffers=tb,
            seed=seed_b,
            instance_id=counter.next(),
        )
        inst_b["pair_id"] = pid
        inst_b["pair_type"] = "feasible_obvious_unsat"
        inst_b["sibling_role"] = "unsat"

        instances.extend([inst_a, inst_b])
        pairs_manifest.append({
            "pair_id": pid,
            "pair_type": "feasible_obvious_unsat",
            "infeasible_mechanism": "two_independent_3cycles",
            "expected_removal_distance": 2,
            "instance_ids": [inst_a["instance_id"], inst_b["instance_id"]],
        })

    # -----------------------------------------------------------------------
    # 3. feasible_length: 4 pairs
    # -----------------------------------------------------------------------
    for i in range(4):
        pid = f"pair-{pair_idx + 1:04d}"
        pair_idx += 1
        seed_a = base_seed + pair_idx * 17 + 3
        n_blocks = 4 + (i % 2)
        n_stacks = 3
        tb = 2

        inst_a = generate_feasible_instance(
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            table_buffers=tb,
            n_ordering_constraints=1,
            seed=seed_a,
            instance_id=counter.next(),
        )
        inst_a["pair_id"] = pid
        inst_a["pair_type"] = "feasible_length"
        inst_a["sibling_role"] = "feasible"

        inst_b = copy.deepcopy(inst_a)
        inst_b["instance_id"] = counter.next()
        inst_b["sibling_role"] = "control"
        for j in range(5):
            inst_b["hard_constraints"].append({
                "type": "nonbinding",
                "note": f"length_pad_{j}",
                "id": f"NB{j}",
            })
        inst_b["padding"] = {"nonbinding_constraints": 5, "distractors": 0, "prompt_pad_tokens": 0}

        instances.extend([inst_a, inst_b])
        pairs_manifest.append({
            "pair_id": pid,
            "pair_type": "feasible_length",
            "instance_ids": [inst_a["instance_id"], inst_b["instance_id"]],
        })

    # -----------------------------------------------------------------------
    # 4. feasible_distractor (4) + feasible_proxy_control reassigned (4) = 8
    # -----------------------------------------------------------------------
    for i in range(8):
        pid = f"pair-{pair_idx + 1:04d}"
        pair_idx += 1
        seed_a = base_seed + pair_idx * 17 + 3
        pt = "feasible_distractor" if i < 4 else "feasible_proxy_control"
        n_blocks = 4 + (i % 3)
        n_stacks = 3
        tb = 2

        inst_a = generate_feasible_instance(
            n_blocks=n_blocks,
            n_stacks=n_stacks,
            table_buffers=tb,
            n_ordering_constraints=1,
            seed=seed_a,
            instance_id=counter.next(),
        )
        inst_a["pair_id"] = pid
        inst_a["pair_type"] = pt
        inst_a["sibling_role"] = "feasible"

        inst_b = copy.deepcopy(inst_a)
        inst_b["instance_id"] = counter.next()
        inst_b["sibling_role"] = "control"
        n_distractors = 3 + (i % 3)
        for j in range(n_distractors):
            inst_b["hard_constraints"].append({
                "type": "nonbinding",
                "note": f"distractor_{j}",
                "id": f"D{j}",
            })
        inst_b["padding"] = {
            "nonbinding_constraints": n_distractors,
            "distractors": n_distractors,
            "prompt_pad_tokens": 0,
        }

        instances.extend([inst_a, inst_b])
        pairs_manifest.append({
            "pair_id": pid,
            "pair_type": pt,
            "instance_ids": [inst_a["instance_id"], inst_b["instance_id"]],
            "note": MANIFEST_NOTE if i >= 4 else "",
        })

    # Re-number instances sequentially
    for idx, inst in enumerate(instances):
        inst["instance_id"] = f"bw-v1-{idx + 1:04d}"

    inst_id_by_pair: dict[str, list[str]] = {}
    for inst in instances:
        pid = inst["pair_id"]
        if pid not in inst_id_by_pair:
            inst_id_by_pair[pid] = []
        inst_id_by_pair[pid].append(inst["instance_id"])
    for pm in pairs_manifest:
        pm["instance_ids"] = inst_id_by_pair[pm["pair_id"]]

    manifest = {
        "version": "v1",
        "base_seed": base_seed,
        "total_pairs": 30,
        "total_instances": 60,
        "pair_type_counts": {
            "feasible_min_unsat": 12,
            "feasible_obvious_unsat": 6,
            "feasible_length": 4,
            "feasible_distractor": 4,
            "feasible_proxy_control": 4,
        },
        "proxy_control_note": MANIFEST_NOTE,
        "infeasibility_note": (
            "All unsat instances use contradictory ordering constraint cycles "
            "(not buffer starvation). This ensures well-defined removal_distance "
            "values. min_unsat: single 3-cycle, removal_distance=1. "
            "obvious_unsat: two independent 3-cycles (n_blocks>=6) or 4-cycle, "
            "removal_distance=2."
        ),
        "pairs": pairs_manifest,
    }

    assert len(instances) == 60, f"Expected 60 instances, got {len(instances)}"
    return instances, manifest

"""
padding.py — Non-binding constraint and distractor padding to match siblings.

Goals:
1. For each feasible/unsat pair: add nonbinding constraints to the feasible sibling
   to match the constraint count of the unsat sibling (which may have more structural
   constraints from infeasibility construction).
2. For length-inflated and distractor pairs: already handled in siblings.py via deep copy.
3. Update prompt_token_estimate for every instance.

Token estimator:
  Method: word count / 0.75 (words = whitespace-split tokens of JSON serialization).
  Rationale: standard approximation, documented here.
  A more accurate estimator would tokenize with tiktoken, but that is a third-party dep.
  We document this choice; accuracy within ~20% is sufficient for matching purposes.

Matching tolerance (plan §15.2.1):
  Padded sibling constraint count: within ±1 of reference.
  Prompt token estimate: within ±10% of reference.
"""

from __future__ import annotations
import json
import copy


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def estimate_tokens(instance: dict) -> int:
    """
    Estimate agent-facing prompt tokens for an instance.
    Method: serialize to JSON, count whitespace-separated 'words', divide by 0.75.
    This approximates BPE tokenization for typical text/code content.
    """
    text = json.dumps(instance, default=str)
    word_count = len(text.split())
    return max(1, int(word_count / 0.75))


# ---------------------------------------------------------------------------
# Padding logic
# ---------------------------------------------------------------------------

def _count_binding_constraints(instance: dict) -> int:
    """Count non-nonbinding hard constraints."""
    return sum(1 for c in instance.get("hard_constraints", [])
               if c.get("type") != "nonbinding")


def _count_total_constraints(instance: dict) -> int:
    return len(instance.get("hard_constraints", []))


def pad_instance(instance: dict, target_constraint_count: int,
                 pad_prefix: str = "PAD") -> dict:
    """
    Add nonbinding constraints to instance until total count >= target_constraint_count.
    Returns a copy with padding applied.
    """
    inst = copy.deepcopy(instance)
    current = _count_total_constraints(inst)
    n_to_add = max(0, target_constraint_count - current)

    existing_nb = sum(1 for c in inst["hard_constraints"] if c.get("type") == "nonbinding")
    for i in range(n_to_add):
        inst["hard_constraints"].append({
            "type": "nonbinding",
            "note": f"{pad_prefix}_{existing_nb + i}",
            "id": f"{pad_prefix}{existing_nb + i}",
        })

    inst["padding"]["nonbinding_constraints"] = (
        inst["padding"].get("nonbinding_constraints", 0) + n_to_add
    )
    inst["prompt_token_estimate"] = estimate_tokens(inst)
    return inst


def pad_all_instances(instances: list[dict]) -> list[dict]:
    """
    For each feasible/unsat pair: pad the feasible sibling to match unsat sibling's
    constraint count. For control pairs: already padded in siblings.py; just update
    token estimates.

    Matching is per pair_id: find the pair, identify unsat sibling (or control),
    pad the other to match.

    Pipeline order note: this function must run before calibration (plan §9.1).
    """
    # Build pair -> [instances] map
    pair_map: dict[str, list[dict]] = {}
    for inst in instances:
        pid = inst.get("pair_id")
        if pid:
            pair_map.setdefault(pid, []).append(inst)

    result = []
    inst_by_id = {inst["instance_id"]: inst for inst in instances}

    processed_ids: set[str] = set()

    for inst in instances:
        if inst["instance_id"] in processed_ids:
            continue

        pid = inst.get("pair_id")
        pair_insts = pair_map.get(pid, [inst])

        if len(pair_insts) == 2:
            a, b = pair_insts

            # Determine which to pad
            # For feasible/unsat pairs: pad feasible to match unsat constraint count
            # For feasible/control pairs: pad neither (already handled) but update tokens
            if a.get("sibling_role") == "feasible" and b.get("sibling_role") in ("unsat", "control"):
                target = _count_total_constraints(b)
                a_new = pad_instance(a, target, pad_prefix="PAD")
                b_new = copy.deepcopy(b)
                b_new["prompt_token_estimate"] = estimate_tokens(b_new)
            elif b.get("sibling_role") == "feasible" and a.get("sibling_role") in ("unsat", "control"):
                target = _count_total_constraints(a)
                b_new = pad_instance(b, target, pad_prefix="PAD")
                a_new = copy.deepcopy(a)
                a_new["prompt_token_estimate"] = estimate_tokens(a_new)
            else:
                # both feasible: update tokens only
                a_new = copy.deepcopy(a)
                b_new = copy.deepcopy(b)
                a_new["prompt_token_estimate"] = estimate_tokens(a_new)
                b_new["prompt_token_estimate"] = estimate_tokens(b_new)

            # Update instances in place
            for iid, new_inst in [(a["instance_id"], a_new), (b["instance_id"], b_new)]:
                inst_by_id[iid].update(new_inst)
                inst_by_id[iid]["instance_id"] = iid  # preserve ID

            processed_ids.add(a["instance_id"])
            processed_ids.add(b["instance_id"])
        else:
            # Singleton: just update token estimate
            inst["prompt_token_estimate"] = estimate_tokens(inst)
            processed_ids.add(inst["instance_id"])

    # Return in original order with updates applied
    return [inst_by_id[inst["instance_id"]] for inst in instances]


def check_sibling_matching(instances: list[dict],
                           constraint_tolerance: int = 1,
                           token_tolerance_frac: float = 0.10) -> list[dict]:
    """
    Check that sibling pairs satisfy matching invariants.
    Returns list of violation dicts.
    """
    pair_map: dict[str, list[dict]] = {}
    for inst in instances:
        pid = inst.get("pair_id")
        if pid:
            pair_map.setdefault(pid, []).append(inst)

    violations = []
    for pid, pair_insts in pair_map.items():
        if len(pair_insts) != 2:
            continue
        a, b = pair_insts
        ca = _count_total_constraints(a)
        cb = _count_total_constraints(b)
        if abs(ca - cb) > constraint_tolerance:
            violations.append({
                "pair_id": pid,
                "type": "constraint_count_mismatch",
                "a_count": ca, "b_count": cb, "tolerance": constraint_tolerance,
            })
        ta = a.get("prompt_token_estimate", 0)
        tb = b.get("prompt_token_estimate", 0)
        if ta > 0 and tb > 0:
            frac = abs(ta - tb) / max(ta, tb)
            if frac > token_tolerance_frac:
                violations.append({
                    "pair_id": pid,
                    "type": "token_estimate_mismatch",
                    "a_tokens": ta, "b_tokens": tb, "fraction": round(frac, 3),
                    "tolerance": token_tolerance_frac,
                })
    return violations

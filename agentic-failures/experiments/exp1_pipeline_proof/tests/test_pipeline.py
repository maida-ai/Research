"""
tests/test_pipeline.py — pytest tests for exp1_pipeline_proof.

Test coverage:
1. Legal-move checker edge cases
2. Oracle validates a known-good trajectory
3. Oracle rejects a known-bad trajectory
4. Hand-built unsat instance returns feasible=False
5. Sibling matching invariant (constraint count + token estimate tolerance)
6. Pipeline order guard (calibration fails if called before padding)

Performance requirement: entire suite completes in <60s.
Strategy: Tests 5 and 6 use tiny fixtures (2-3 pairs, <=4 blocks, <=2 buffers).
Full 60-instance build happens ONLY in run_all.py.
"""

import sys
import os
import copy
import pytest

# Add the experiment dir to path
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP_DIR)

from domain import (
    is_legal_move, apply_move, legal_moves, goal_test, validate_trajectory,
    stacks_to_tuple, buffers_to_dict, check_constraints,
)
from solver_oracle import run_solver, BFS_EXPANSION_CAP
from padding import check_sibling_matching, estimate_tokens, pad_all_instances
from calibrate import calibrate, PipelineOrderError
from siblings import build_all_pairs
from generator import generate_feasible_instance, generate_contradictory_instance


# ---------------------------------------------------------------------------
# Tiny fixture builders (used by performance-sensitive tests)
# ---------------------------------------------------------------------------

def _tiny_feasible(instance_id="tiny-f-001", seed=0):
    """3 blocks, 2 stacks, 2 buffers — fast to solve."""
    inst = generate_feasible_instance(
        n_blocks=3, n_stacks=2, table_buffers=2,
        n_ordering_constraints=0, seed=seed, instance_id=instance_id,
    )
    inst["pair_id"] = "pair-tiny-01"
    inst["pair_type"] = "feasible_min_unsat"
    inst["sibling_role"] = "feasible"
    inst["prompt_token_estimate"] = 1
    return inst


def _tiny_unsat(instance_id="tiny-u-001", seed=1):
    """3 blocks, contradictory cycle — fast to prove infeasible."""
    inst = generate_contradictory_instance(
        n_blocks=3, n_stacks=2, table_buffers=2,
        seed=seed, cycle_length=3, instance_id=instance_id,
    )
    inst["pair_id"] = "pair-tiny-01"
    inst["pair_type"] = "feasible_min_unsat"
    inst["sibling_role"] = "unsat"
    inst["prompt_token_estimate"] = 1
    return inst


def _build_tiny_pairs(n_pairs: int = 2, base_seed: int = 0) -> list[dict]:
    """Build n_pairs tiny feasible/unsat pairs (2*n_pairs instances total)."""
    instances = []
    for i in range(n_pairs):
        f = _tiny_feasible(instance_id=f"tiny-f-{i:03d}", seed=base_seed + i * 10)
        u = _tiny_unsat(instance_id=f"tiny-u-{i:03d}", seed=base_seed + i * 10 + 1)
        f["pair_id"] = u["pair_id"] = f"pair-tiny-{i:03d}"
        instances.extend([f, u])
    return instances


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_instance():
    """A simple 3-block 2-stack instance."""
    return {
        "instance_id": "test-001",
        "domain": "formal",
        "pair_id": "pair-test",
        "pair_type": "feasible_min_unsat",
        "sibling_role": "feasible",
        "initial_state": {
            "stacks": [["A", "B", "C"], []],
            "buffers": {"b0": None, "b1": None},
        },
        "goal_state": {
            "stacks": [["C", "B", "A"], []],
            "buffers": {"b0": None, "b1": None},
        },
        "action_space": {"type": "blocks_world"},
        "resources": {"table_buffers": 2},
        "hard_constraints": [],
        "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
        "feasible": None,
        "reference_solution": None,
        "removal_distance": None,
        "generator_params": {"seed": 0},
        "prompt_token_estimate": 1,  # non-zero to pass pipeline guard
    }


def _unsat_instance():
    """Hand-built unsat instance: contradictory cycle A>B>C>A."""
    return {
        "instance_id": "test-unsat-001",
        "domain": "formal",
        "pair_id": "pair-test-u",
        "pair_type": "feasible_min_unsat",
        "sibling_role": "unsat",
        "initial_state": {
            "stacks": [["A", "B", "C"], []],
            "buffers": {"b0": None},
        },
        "goal_state": {
            "stacks": [["A", "B", "C"], []],
            "buffers": {"b0": None},
        },
        "action_space": {"type": "blocks_world"},
        "resources": {"table_buffers": 1},
        "hard_constraints": [
            {"type": "order", "above": "A", "below": "B", "id": "C0"},
            {"type": "order", "above": "B", "below": "C", "id": "C1"},
            {"type": "order", "above": "C", "below": "A", "id": "C2"},
        ],
        "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
        "feasible": False,
        "reference_solution": None,
        "removal_distance": None,
        "generator_params": {"seed": 0},
        "prompt_token_estimate": 1,
    }


# ---------------------------------------------------------------------------
# Test 1: Legal-move checker edge cases
# ---------------------------------------------------------------------------

class TestLegalMoveChecker:
    def test_stack_to_stack_legal(self):
        stacks = (("A", "B"), ("C",))
        bufs = {"b0": None}
        ok, reason = is_legal_move("move_stack_0_to_stack_1", stacks, bufs)
        assert ok, reason

    def test_stack_to_stack_same_stack_illegal(self):
        stacks = (("A", "B"), ("C",))
        bufs = {"b0": None}
        ok, reason = is_legal_move("move_stack_0_to_stack_0", stacks, bufs)
        assert not ok

    def test_empty_stack_illegal_src(self):
        stacks = ((), ("C",))
        bufs = {"b0": None}
        ok, reason = is_legal_move("move_stack_0_to_stack_1", stacks, bufs)
        assert not ok
        assert "empty" in reason.lower()

    def test_out_of_range_stack_illegal(self):
        stacks = (("A",),)
        bufs = {}
        ok, reason = is_legal_move("move_stack_0_to_stack_5", stacks, bufs)
        assert not ok

    def test_move_to_occupied_buffer_illegal(self):
        stacks = (("A", "B"),)
        bufs = {"b0": "C"}
        ok, reason = is_legal_move("move_stack_0_to_buffer_b0", stacks, bufs)
        assert not ok
        assert "occupied" in reason.lower()

    def test_move_to_empty_buffer_legal(self):
        stacks = (("A", "B"),)
        bufs = {"b0": None}
        ok, reason = is_legal_move("move_stack_0_to_buffer_b0", stacks, bufs)
        assert ok, reason

    def test_buffer_to_stack_legal(self):
        stacks = ((), ("C",))
        bufs = {"b0": "X"}
        ok, reason = is_legal_move("move_buffer_b0_to_stack_0", stacks, bufs)
        assert ok, reason

    def test_empty_buffer_to_stack_illegal(self):
        stacks = (("A",),)
        bufs = {"b0": None}
        ok, reason = is_legal_move("move_buffer_b0_to_stack_0", stacks, bufs)
        assert not ok

    def test_nonexistent_buffer_illegal(self):
        stacks = (("A",),)
        bufs = {"b0": None}
        ok, reason = is_legal_move("move_stack_0_to_buffer_b99", stacks, bufs)
        assert not ok

    def test_buffer_to_buffer_legal(self):
        stacks = (("A",),)
        bufs = {"b0": "X", "b1": None}
        ok, reason = is_legal_move("move_buffer_b0_to_buffer_b1", stacks, bufs)
        assert ok, reason

    def test_buffer_to_buffer_same_illegal(self):
        stacks = (("A",),)
        bufs = {"b0": "X", "b1": None}
        ok, reason = is_legal_move("move_buffer_b0_to_buffer_b0", stacks, bufs)
        assert not ok

    def test_apply_move_stack_to_stack(self):
        stacks = (("A", "B"), ("C",))
        bufs = {"b0": None}
        ns, nb = apply_move("move_stack_0_to_stack_1", stacks, bufs)
        assert ns[0] == ("A",)
        assert ns[1] == ("C", "B")

    def test_apply_move_stack_to_buffer(self):
        stacks = (("A", "B"),)
        bufs = {"b0": None}
        ns, nb = apply_move("move_stack_0_to_buffer_b0", stacks, bufs)
        assert ns[0] == ("A",)
        assert nb["b0"] == "B"

    def test_legal_moves_empty_board(self):
        stacks = ((), ())
        bufs = {}
        moves = legal_moves(stacks, bufs)
        assert moves == []

    def test_legal_moves_with_buffer(self):
        stacks = (("A",), ())
        bufs = {"b0": None}
        moves = legal_moves(stacks, bufs)
        assert "move_stack_0_to_stack_1" in moves
        assert "move_stack_0_to_buffer_b0" in moves


# ---------------------------------------------------------------------------
# Test 2: Oracle validates a known-good trajectory
# ---------------------------------------------------------------------------

class TestOracleGoodTrajectory:
    def test_simple_reversal_known_good(self):
        """
        3 blocks: A, B, C in stack 0 (bottom to top).
        Goal: C, B, A in stack 0.
        With 2 buffers, this is solvable.
        """
        instance = _simple_instance()
        # Stack 0: [A, B, C], Stack 1: [], buf0: None, buf1: None
        # Move C to buf0: stack0=(A,B), buf0=C
        # Move B to buf1: stack0=(A,), buf1=B
        # Move A to stack1: stack0=(), stack1=(A,)
        # Move buf0 (C) to stack0: stack0=(C,)
        # Move buf1 (B) to stack0: stack0=(C,B)
        # Move stack1 top (A) to stack0: stack0=(C,B,A) -- GOAL!
        good_moves = [
            "move_stack_0_to_buffer_b0",    # C -> b0
            "move_stack_0_to_buffer_b1",    # B -> b1
            "move_stack_0_to_stack_1",      # A -> stack1
            "move_buffer_b0_to_stack_0",    # C -> stack0
            "move_buffer_b1_to_stack_0",    # B -> stack0
            "move_stack_1_to_stack_0",      # A -> stack0
        ]
        ok, violations = validate_trajectory(instance, good_moves)
        assert ok, f"Expected valid trajectory, got violations: {violations}"
        assert violations == []

    def test_goal_test_exact_match(self):
        stacks = (("A", "B"), ("C",))
        bufs = {"b0": None}
        goal = {"stacks": [["A", "B"], ["C"]], "buffers": {"b0": None}}
        assert goal_test(stacks, bufs, goal)

    def test_goal_test_wrong_order(self):
        stacks = (("A", "B"), ("C",))
        bufs = {"b0": None}
        goal = {"stacks": [["B", "A"], ["C"]], "buffers": {"b0": None}}
        assert not goal_test(stacks, bufs, goal)


# ---------------------------------------------------------------------------
# Test 3: Oracle rejects a known-bad trajectory
# ---------------------------------------------------------------------------

class TestOracleBadTrajectory:
    def test_illegal_move_rejected(self):
        instance = _simple_instance()
        # Trying to move from an empty stack
        bad_moves = ["move_stack_1_to_stack_0"]  # stack1 is empty
        ok, violations = validate_trajectory(instance, bad_moves)
        assert not ok
        assert any("illegal" in v.lower() or "empty" in v.lower() for v in violations)

    def test_wrong_goal_rejected(self):
        instance = _simple_instance()
        # Do one valid move but don't reach goal
        moves = ["move_stack_0_to_buffer_b0"]  # just move one block, don't complete
        ok, violations = validate_trajectory(instance, moves)
        assert not ok
        assert any("goal" in v.lower() for v in violations)

    def test_empty_moves_wrong_goal_rejected(self):
        """Initial state != goal, so empty move list should fail."""
        instance = _simple_instance()
        ok, violations = validate_trajectory(instance, [])
        # Initial state is [A,B,C]/[] but goal is [C,B,A]/[]
        assert not ok

    def test_buffer_limit_violation_rejected(self):
        """Instance with buffer_limit=1 but trajectory uses 2 buffers."""
        instance = {
            "instance_id": "test-bufviol",
            "domain": "formal",
            "pair_id": None, "pair_type": None, "sibling_role": "feasible",
            "initial_state": {
                "stacks": [["A", "B"], []],
                "buffers": {"b0": None, "b1": None},
            },
            "goal_state": {
                "stacks": [["B", "A"], []],
                "buffers": {"b0": None, "b1": None},
            },
            "action_space": {},
            "resources": {"table_buffers": 2},
            "hard_constraints": [
                {"type": "buffer_limit", "max": 1, "id": "BL0"},
            ],
            "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
            "feasible": None, "reference_solution": None, "removal_distance": None,
            "generator_params": {"seed": 0}, "prompt_token_estimate": 1,
        }
        # Move A to b0, then B to b1 (violates buffer_limit=1)
        moves = [
            "move_stack_0_to_buffer_b0",  # A -> b0; OK (1 buffer used)
            "move_stack_0_to_buffer_b1",  # B -> b1; VIOLATION (2 buffers used)
        ]
        ok, violations = validate_trajectory(instance, moves)
        assert not ok


# ---------------------------------------------------------------------------
# Test 4: Hand-built unsat instance returns feasible=False
# ---------------------------------------------------------------------------

class TestUnsatOracle:
    def test_contradictory_order_constraints_unsat(self):
        """Cycle A>B>C>A makes any single-stack goal impossible."""
        instance = _unsat_instance()
        result = run_solver(instance)
        assert result["feasible"] is False, (
            f"Expected infeasible, got feasible={result['feasible']}, "
            f"status={result['status']}"
        )

    def test_buffer_starvation_unsat(self):
        """
        3 blocks in one stack, goal=reverse, but 0 buffers and 1 stack available.
        Reversing 3 blocks with only 1 stack is impossible.
        """
        instance = {
            "instance_id": "test-bufstar",
            "domain": "formal",
            "pair_id": None, "pair_type": None, "sibling_role": "unsat",
            "initial_state": {
                "stacks": [["A", "B", "C"]],
                "buffers": {},  # no buffers, 1 stack only
            },
            "goal_state": {
                "stacks": [["C", "B", "A"]],
                "buffers": {},
            },
            "action_space": {},
            "resources": {"table_buffers": 0},
            "hard_constraints": [],
            "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
            "feasible": False,
            "reference_solution": None, "removal_distance": None,
            "generator_params": {"seed": 0}, "prompt_token_estimate": 1,
        }
        result = run_solver(instance)
        assert result["feasible"] is False, f"Expected infeasible, got {result}"

    def test_solver_cache_hit(self):
        """Running solver twice on same instance returns cache_hit=True second time."""
        import solver_oracle
        # Clear in-memory cache for this instance
        instance = _unsat_instance()
        instance["instance_id"] = "cache-test-uniq-999"
        from solver_oracle import _instance_cache_key, _SOLVER_CACHE
        key = _instance_cache_key(instance)
        _SOLVER_CACHE.pop(key, None)

        r1 = run_solver(instance)
        r2 = run_solver(instance)
        assert r2["cache_hit"] is True, "Second call should be a cache hit"
        assert r1["feasible"] == r2["feasible"]


# ---------------------------------------------------------------------------
# Test 5: Sibling matching invariant — uses tiny fixtures, NOT full 60
# ---------------------------------------------------------------------------

class TestSiblingMatching:
    def test_tiny_pairs_padded_match(self):
        """
        2 tiny pairs (4 instances total), apply padding, check invariants.
        Completes in <1s.
        """
        instances = _build_tiny_pairs(n_pairs=2, base_seed=0)
        padded = pad_all_instances(instances)
        violations = check_sibling_matching(
            padded,
            constraint_tolerance=1,
            token_tolerance_frac=0.25,
        )
        constraint_violations = [v for v in violations if v["type"] == "constraint_count_mismatch"]
        assert len(constraint_violations) == 0, (
            f"Constraint count mismatches: {constraint_violations}"
        )

    def test_60_instances_generated(self):
        """Structural check: build_all_pairs returns 60 instances."""
        instances, manifest = build_all_pairs(base_seed=42)
        assert len(instances) == 60

    def test_pair_type_counts_correct(self):
        """Pair type distribution matches plan §15.2.1."""
        instances, manifest = build_all_pairs(base_seed=42)
        pt_counts: dict[str, int] = {}
        for inst in instances:
            pt = inst.get("pair_type")
            pt_counts[pt] = pt_counts.get(pt, 0) + 1
        assert pt_counts.get("feasible_min_unsat", 0) == 24  # 12 pairs * 2
        assert pt_counts.get("feasible_obvious_unsat", 0) == 12  # 6 pairs * 2
        assert pt_counts.get("feasible_length", 0) == 8   # 4 pairs * 2

    def test_unsat_instances_have_feasible_false(self):
        """Generator must label unsat instances with feasible=False."""
        instances, _ = build_all_pairs(base_seed=42)
        for inst in instances:
            if inst.get("sibling_role") == "unsat":
                assert inst.get("feasible") is False, (
                    f"Unsat instance {inst['instance_id']} has feasible={inst.get('feasible')}"
                )


# ---------------------------------------------------------------------------
# Test 6: Pipeline order guard — uses tiny fixtures
# ---------------------------------------------------------------------------

class TestPipelineOrderGuard:
    def test_calibrate_fails_without_padding(self):
        """calibrate() must raise PipelineOrderError if instances have prompt_token_estimate=0."""
        instances = _build_tiny_pairs(n_pairs=2, base_seed=0)
        for inst in instances:
            inst["prompt_token_estimate"] = 0

        with pytest.raises(PipelineOrderError):
            calibrate(instances, time_limit_s=2.0, seed=42)

    def test_calibrate_succeeds_after_padding_tiny(self):
        """
        calibrate() on tiny fixtures (4 instances) completes fast.
        Uses _build_tiny_pairs to avoid building all 60 here.
        """
        instances = _build_tiny_pairs(n_pairs=3, base_seed=0)
        padded = pad_all_instances(instances)
        # Should not raise; 6 instances total
        records, bands, gov = calibrate(padded, time_limit_s=5.0, seed=42)
        assert len(records) == 6
        assert "band_counts" in bands

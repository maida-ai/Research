"""
domain.py — Resource-constrained Blocks World with limited table buffers.

State: stacks of blocks (tuples of tuples) + a fixed set of named table buffers
       (dict mapping buffer_id -> block or None).

Blocks are identified by string names (e.g. "A", "B", ...).
Table buffers are a fixed-capacity temporary staging area.
Stacks are ordered bottom-to-top; the last element is the top.

Legal moves:
  move_stack_to_stack(src_stack, dst_stack)
      - src_stack != dst_stack
      - src_stack is non-empty
  move_stack_to_buffer(src_stack, buffer_id)
      - src_stack non-empty
      - buffer_id exists and is currently empty (None)
  move_buffer_to_stack(buffer_id, dst_stack)
      - buffer_id is occupied
  move_buffer_to_buffer(src_buf, dst_buf)
      - src_buf occupied, dst_buf empty

Hard constraints attached to an instance:
  Each constraint is a dict with a "type" key:
    {"type": "on",       "block": X, "below": Y}   -- X must be directly below Y in goal
    {"type": "on_top",   "block": X, "stack": S}    -- X must be on top of stack S in goal
    {"type": "not_on",   "block": X, "below": Y}    -- X must NOT be directly below Y
    {"type": "order",    "above": X, "below": Y}    -- in any common stack, X must be above Y
    {"type": "buffer_limit", "max": k}              -- at most k buffers used simultaneously
       (this is enforced dynamically via resource tracking, not as goal constraint)

Goal state format:
  {"stacks": [[...], [...], ...],  "buffers": {"b0": null, "b1": null, ...}}
  Stacks may use wildcard "_any_" to allow any block at that position.

Oracle: validate_trajectory checks:
  - every move is legal at the state it is applied
  - hard constraints are respected at every intermediate state (for buffer_limit) and in final goal state
  - final state matches goal (exact equality, modulo wildcard)
"""

from __future__ import annotations
import hashlib

# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

def _state_key(stacks: tuple[tuple[str, ...], ...], buffers: dict[str, str | None]) -> tuple:
    """Canonical hashable key for a state.
    Returns a tuple (hashable, fast) instead of a JSON string.
    tuple of tuples is ~10x faster than json.dumps in the BFS inner loop.
    """
    buf_items = tuple(sorted(buffers.items()))
    return (stacks, buf_items)


def state_hash(stacks, buffers) -> str:
    key = _state_key(stacks, buffers)
    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


def stacks_to_tuple(stacks: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(s) for s in stacks)


def buffers_to_dict(buffers: dict) -> dict[str, str | None]:
    return {k: v for k, v in buffers.items()}


# ---------------------------------------------------------------------------
# Legal-move checker
# ---------------------------------------------------------------------------

def _stacks_list(stacks_t: tuple[tuple[str, ...], ...]) -> list[list[str]]:
    return [list(s) for s in stacks_t]


def legal_moves(stacks: tuple[tuple[str, ...], ...],
                buffers: dict[str, str | None]) -> list[str]:
    """Return all currently legal move strings."""
    moves = []
    n_stacks = len(stacks)
    buf_ids = list(buffers.keys())

    # stack -> stack
    for i, src in enumerate(stacks):
        if not src:
            continue
        for j in range(n_stacks):
            if i != j:
                moves.append(f"move_stack_{i}_to_stack_{j}")

    # stack -> buffer
    for i, src in enumerate(stacks):
        if not src:
            continue
        for buf_id in buf_ids:
            if buffers[buf_id] is None:
                moves.append(f"move_stack_{i}_to_buffer_{buf_id}")

    # buffer -> stack
    for buf_id in buf_ids:
        if buffers[buf_id] is not None:
            for j in range(n_stacks):
                moves.append(f"move_buffer_{buf_id}_to_stack_{j}")

    # buffer -> buffer
    for src_buf in buf_ids:
        if buffers[src_buf] is not None:
            for dst_buf in buf_ids:
                if dst_buf != src_buf and buffers[dst_buf] is None:
                    moves.append(f"move_buffer_{src_buf}_to_buffer_{dst_buf}")

    return moves


def is_legal_move(move: str,
                  stacks: tuple[tuple[str, ...], ...],
                  buffers: dict[str, str | None]) -> tuple[bool, str]:
    """Check if a move string is legal in the current state. Returns (ok, reason)."""
    parts = move.split("_")
    try:
        # Format: move_<src_type>_<src_id>_to_<dst_type>_<dst_id>
        # Parts: ["move", src_type, src_id, "to", dst_type, dst_id]
        if parts[0] != "move":
            return False, "not a move command"

        # Find "to" separator
        to_idx = parts.index("to")
        src_parts = parts[1:to_idx]
        dst_parts = parts[to_idx + 1:]

        src_type = src_parts[0]
        src_id = "_".join(src_parts[1:])
        dst_type = dst_parts[0]
        dst_id = "_".join(dst_parts[1:])

        if src_type == "stack":
            src_idx = int(src_id)
            if src_idx < 0 or src_idx >= len(stacks):
                return False, f"stack {src_idx} out of range"
            if not stacks[src_idx]:
                return False, f"stack {src_idx} is empty"

            if dst_type == "stack":
                dst_idx = int(dst_id)
                if dst_idx == src_idx:
                    return False, "src and dst stack are same"
                if dst_idx < 0 or dst_idx >= len(stacks):
                    return False, f"stack {dst_idx} out of range"
                return True, ""

            elif dst_type == "buffer":
                if dst_id not in buffers:
                    return False, f"buffer {dst_id} does not exist"
                if buffers[dst_id] is not None:
                    return False, f"buffer {dst_id} is occupied"
                return True, ""
            else:
                return False, f"unknown dst_type {dst_type}"

        elif src_type == "buffer":
            if src_id not in buffers:
                return False, f"buffer {src_id} does not exist"
            if buffers[src_id] is None:
                return False, f"buffer {src_id} is empty"

            if dst_type == "stack":
                dst_idx = int(dst_id)
                if dst_idx < 0 or dst_idx >= len(stacks):
                    return False, f"stack {dst_idx} out of range"
                return True, ""

            elif dst_type == "buffer":
                if dst_id not in buffers:
                    return False, f"buffer {dst_id} does not exist"
                if dst_id == src_id:
                    return False, "src and dst buffer are same"
                if buffers[dst_id] is not None:
                    return False, f"buffer {dst_id} is occupied"
                return True, ""
            else:
                return False, f"unknown dst_type {dst_type}"
        else:
            return False, f"unknown src_type {src_type}"

    except (ValueError, IndexError) as e:
        return False, f"parse error: {e}"


def apply_move(move: str,
               stacks: tuple[tuple[str, ...], ...],
               buffers: dict[str, str | None]) -> tuple[tuple[tuple[str, ...], ...], dict[str, str | None]]:
    """Apply a move and return new (stacks, buffers). Assumes move is legal."""
    new_stacks = [list(s) for s in stacks]
    new_bufs = dict(buffers)

    parts = move.split("_")
    to_idx = parts.index("to")
    src_parts = parts[1:to_idx]
    dst_parts = parts[to_idx + 1:]

    src_type = src_parts[0]
    src_id = "_".join(src_parts[1:])
    dst_type = dst_parts[0]
    dst_id = "_".join(dst_parts[1:])

    if src_type == "stack":
        src_idx = int(src_id)
        block = new_stacks[src_idx].pop()
        if dst_type == "stack":
            dst_idx = int(dst_id)
            new_stacks[dst_idx].append(block)
        else:  # buffer
            new_bufs[dst_id] = block

    elif src_type == "buffer":
        block = new_bufs[src_id]
        new_bufs[src_id] = None
        if dst_type == "stack":
            dst_idx = int(dst_id)
            new_stacks[dst_idx].append(block)
        else:  # buffer
            new_bufs[dst_id] = block

    return stacks_to_tuple(new_stacks), new_bufs


# ---------------------------------------------------------------------------
# Goal test
# ---------------------------------------------------------------------------

def _match_stack(actual: tuple[str, ...], goal_stack: list) -> bool:
    """Match a single stack against goal spec (may contain '_any_' wildcards)."""
    if len(actual) != len(goal_stack):
        return False
    for a, g in zip(actual, goal_stack):
        if g != "_any_" and a != g:
            return False
    return True


def goal_test(stacks: tuple[tuple[str, ...], ...],
              buffers: dict[str, str | None],
              goal: dict) -> bool:
    """Test if state matches goal under hard constraints (goal structure only)."""
    goal_stacks = goal.get("stacks", [])
    goal_buffers = goal.get("buffers", {})

    # Check stacks match (order-sensitive)
    if len(stacks) != len(goal_stacks):
        return False
    for actual_s, goal_s in zip(stacks, goal_stacks):
        if not _match_stack(actual_s, goal_s):
            return False

    # Check buffers (None means empty required, string means specific block)
    for buf_id, expected in goal_buffers.items():
        if buf_id not in buffers:
            return False
        if expected is None:
            if buffers[buf_id] is not None:
                return False
        else:
            if buffers[buf_id] != expected:
                return False
    return True


# ---------------------------------------------------------------------------
# Constraint checkers
# ---------------------------------------------------------------------------

def _find_block_in_stacks(block: str, stacks) -> tuple[int, int] | None:
    """Return (stack_idx, position) for block, or None."""
    for si, s in enumerate(stacks):
        for pi, b in enumerate(s):
            if b == block:
                return (si, pi)
    return None


def check_constraints(stacks: tuple[tuple[str, ...], ...],
                      buffers: dict[str, str | None],
                      constraints: list[dict]) -> list[str]:
    """
    Check hard constraints against a state.
    Returns list of violated constraint IDs (or descriptions).
    Note: buffer_limit constraints are checked separately via peak tracking.
    """
    violated = []
    for i, c in enumerate(constraints):
        ctype = c["type"]
        cid = c.get("id", f"C{i}")

        if ctype == "on":
            # block X must be directly below Y (in the goal)
            # Only applicable as a goal constraint; skip here for trajectory checks
            pass

        elif ctype == "not_on":
            block = c["block"]
            below_block = c["below"]
            loc = _find_block_in_stacks(block, stacks)
            loc_below = _find_block_in_stacks(below_block, stacks)
            if loc and loc_below:
                si_b, pi_b = loc
                si_bl, pi_bl = loc_below
                if si_b == si_bl and pi_b == pi_bl + 1:
                    violated.append(cid)

        elif ctype == "order":
            # above must be above below in any common stack
            above = c["above"]
            below = c["below"]
            for s in stacks:
                s_list = list(s)
                if above in s_list and below in s_list:
                    idx_above = s_list.index(above)
                    idx_below = s_list.index(below)
                    if idx_above < idx_below:  # lower index = lower position
                        violated.append(cid)
                        break

        elif ctype == "on_top":
            block = c["block"]
            stack_idx = c["stack"]
            if stack_idx < len(stacks):
                s = stacks[stack_idx]
                if not s or s[-1] != block:
                    violated.append(cid)
            else:
                violated.append(cid)

        elif ctype == "buffer_limit":
            # Check current buffer usage
            used = sum(1 for v in buffers.values() if v is not None)
            if used > c["max"]:
                violated.append(cid)

        elif ctype == "nonbinding":
            # Non-binding padding constraint; never violated
            pass

    return violated


def buffers_used(buffers: dict[str, str | None]) -> int:
    return sum(1 for v in buffers.values() if v is not None)


# ---------------------------------------------------------------------------
# Oracle: validate_trajectory
# ---------------------------------------------------------------------------

def validate_trajectory(instance: dict, moves: list[str]) -> tuple[bool, list[str]]:
    """
    Validate a sequence of moves against an instance.

    Returns:
        (ok: bool, violated_constraints: list[str])
        ok=True iff all moves are legal, constraints respected throughout,
        and the final state matches the goal.
    """
    stacks = stacks_to_tuple(instance["initial_state"]["stacks"])
    buffers = buffers_to_dict(instance["initial_state"]["buffers"])
    hard_constraints = instance.get("hard_constraints", [])
    goal = instance["goal_state"]
    violations = []

    for step, move in enumerate(moves):
        legal, reason = is_legal_move(move, stacks, buffers)
        if not legal:
            violations.append(f"step{step}:illegal_move:{reason}")
            return False, violations
        stacks, buffers = apply_move(move, stacks, buffers)

        # Check buffer_limit constraints after each move
        bl_viols = [v for c in hard_constraints
                    if c.get("type") == "buffer_limit"
                    for v in ([c.get("id", "buffer_limit")]
                               if buffers_used(buffers) > c["max"] else [])]
        violations.extend([f"step{step}:{v}" for v in bl_viols])

    # Final goal check
    if not goal_test(stacks, buffers, goal):
        violations.append("goal_not_reached")
        return False, violations

    # Check goal constraints (on, on_top, order, not_on) at final state
    goal_viols = check_constraints(stacks, buffers,
                                   [c for c in hard_constraints
                                    if c.get("type") not in ("nonbinding",)])
    violations.extend(goal_viols)

    if violations:
        return False, violations
    return True, []

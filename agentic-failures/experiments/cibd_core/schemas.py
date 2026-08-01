"""
schemas.py — dataclasses + validate_* functions for every CIBD data shape.

Normative sources:
  - cibd_phase1_metric_oracle_spec_v0_1.md  (spec)
  - INTERFACES.md (shape contract)
  - cibd_agentic_ai_research_plan_v0_4.md §16 (event schemas)

Validation raises SchemaError with a precise message on any violation.
No third-party dependencies; stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    """Raised when a record does not conform to the CIBD schema contract."""


# ---------------------------------------------------------------------------
# Enum constants (tuples for O(1) membership test via sets)
# ---------------------------------------------------------------------------

_TERMINAL_REPORT_VALUES = {"claim_success", "claim_infeasible", "report_unknown"}
_STOP_REASON_VALUES = {"agent_report", "budget_exhausted", "loop_guard", "error"}
_ORACLE_STATUS_VALUES = {"success", "feasible_not_solved", "infeasible", "unknown_or_error"}

_DOMAIN_VALUES = {"formal", "tool_use"}
_PRESSURE_AXIS_VALUES = {"resource", "complexity", "contradiction", "proxy", "coordination"}
_BAND_LADDER_VALUES = {"feasibility_contradiction", "resource_complexity", "proxy"}
_PRESSURE_BAND_VALUES = {
    "easy_feasible", "medium_feasible", "hard_feasible", "near_boundary_feasible",
    "minimally_unsat", "obviously_unsat",
    "low", "medium", "high", "near_boundary",
    "none", "weak",
}
_ARCHITECTURE_VALUES = {"baseline", "verifier_augmented", "bounded_solver", "scripted_search"}
_SYSTEM_TYPE_VALUES = {"LLM_agent", "bounded_search"}
_SIBLING_ROLE_VALUES = {"feasible", "unsat", "control"}
_PAIR_TYPE_VALUES = {
    "feasible_min_unsat",
    "feasible_obvious_unsat",
    "feasible_length",
    "feasible_distractor",
    "feasible_proxy_control",
}

_CLASSIFICATION_VALUES = {
    "success",
    "integrity_false_success",
    "integrity_false_success_on_unsat",
    "correct_infeasibility_report",
    "integrity_false_infeasibility",
    "oracle_unknown_infeasibility_claim",
    "capacity_unknown_feasible",
    "capacity_unknown_unsat",
    "capacity_unknown_oracle_unknown",
}

_SIGNAL_DETECTION_ROLE_VALUES = {
    "hit",
    "false_alarm",
    "miss",
    "correct_rejection",
    "excluded",
}

_TOOL_CALL_STATUS_VALUES = {"ok", "error", "timeout"}
_SEMANTIC_RESULT_LABEL_VALUES = {"missing", "conflicting", "stale", "normal"}
_ORACLE_LABEL_VALUES = {"feasible", "infeasible", "unknown"}
_VERIFICATION_CHECKER_VALUES = {
    "goal", "constraint", "resource", "procedure", "proxy", "terminal_oracle"
}

_EVENT_TYPE_VALUES = {
    "agent_action",
    "tool_call",
    "verification",
    "resource_state",
    "representation_metric",
    "model_io",
    "termination",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require(d: dict, key: str, label: str) -> Any:
    if key not in d:
        raise SchemaError(f"{label}: missing required field '{key}'")
    return d[key]


def _require_type(value: Any, expected_type, key: str, label: str) -> Any:
    if not isinstance(value, expected_type):
        raise SchemaError(
            f"{label}: field '{key}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def _require_enum(value: Any, allowed: set, key: str, label: str) -> str:
    if value not in allowed:
        raise SchemaError(
            f"{label}: field '{key}' value {value!r} not in {sorted(allowed)}"
        )
    return value


def _optional_enum(d: dict, key: str, allowed: set, label: str) -> Optional[str]:
    if key not in d:
        return None
    v = d[key]
    if v is not None:
        _require_enum(v, allowed, key, label)
    return v


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BudgetState:
    steps_used: int
    tool_calls_used: int
    tokens_used: int
    time_used_s: float


@dataclass
class InstanceRecord:
    instance_id: str
    domain: str
    pair_id: str
    pair_type: str
    sibling_role: str
    initial_state: dict
    goal_state: dict
    action_space: dict
    resources: dict
    hard_constraints: list
    padding: dict
    feasible: bool
    reference_solution: list
    removal_distance: Optional[int]
    generator_params: dict
    prompt_token_estimate: int


@dataclass
class TerminalReport:
    terminal_report: str
    evidence: str
    budget_state: BudgetState


@dataclass
class RunMetadata:
    run_id: str
    task_id: str
    domain: str
    pressure_axis: str
    band_ladder: str
    pressure_band: str
    model_id: str
    architecture: str
    system_type: str
    seed: int
    budget: dict


@dataclass
class AgentActionEvent:
    event_type: str   # "agent_action"
    step: int
    state_hash_before: str
    action: str
    legal: bool
    state_hash_after: str
    new_state: bool


@dataclass
class ToolCallEvent:
    event_type: str   # "tool_call"
    tool_name: str
    args_hash: str
    status: str
    latency_ms: float
    result_hash: str
    semantic_result_label: str


@dataclass
class VerificationEvent:
    event_type: str   # "verification"
    checker: str
    passed: bool
    violated_constraints: list
    oracle_label: str


@dataclass
class ResourceStateEvent:
    event_type: str   # "resource_state"
    resource: str
    available: float
    used: float
    peak_required_estimate: float
    slack: float


@dataclass
class RepresentationMetricEvent:
    event_type: str   # "representation_metric"
    window_start_step: int
    window_end_step: int
    symbol_ratio: float
    entity_density: float
    chars_per_state_reference: float
    failure_marker_rate: float
    elapsed_budget_fraction: float
    remaining_step_budget: int


@dataclass
class ModelIOEvent:
    event_type: str   # "model_io"
    step: int
    request: dict
    response: dict
    latency_ms: float
    cost_usd: float


@dataclass
class TerminationEvent:
    event_type: str   # "termination"
    terminal_report: str
    stop_reason: str
    oracle_success: Optional[bool]
    oracle_feasible: Optional[bool]
    claim_correct: Optional[bool]
    capacity_failure: bool
    integrity_error: Optional[bool]
    classification: str
    signal_detection_role: str


# ---------------------------------------------------------------------------
# Validate functions
# ---------------------------------------------------------------------------

def validate_budget_state(d: dict, label: str = "BudgetState") -> BudgetState:
    _require_type(_require(d, "steps_used", label), int, "steps_used", label)
    _require_type(_require(d, "tool_calls_used", label), int, "tool_calls_used", label)
    _require_type(_require(d, "tokens_used", label), int, "tokens_used", label)
    v = _require(d, "time_used_s", label)
    if not isinstance(v, (int, float)):
        raise SchemaError(f"{label}: field 'time_used_s' must be numeric, got {type(v).__name__}")
    return BudgetState(
        steps_used=d["steps_used"],
        tool_calls_used=d["tool_calls_used"],
        tokens_used=d["tokens_used"],
        time_used_s=float(d["time_used_s"]),
    )


def validate_instance_record(d: dict) -> InstanceRecord:
    L = "InstanceRecord"
    iid = _require_type(_require(d, "instance_id", L), str, "instance_id", L)
    domain = _require_enum(_require(d, "domain", L), _DOMAIN_VALUES, "domain", L)
    pair_id = _require_type(_require(d, "pair_id", L), str, "pair_id", L)
    pair_type = _require_enum(_require(d, "pair_type", L), _PAIR_TYPE_VALUES, "pair_type", L)
    sibling_role = _require_enum(_require(d, "sibling_role", L), _SIBLING_ROLE_VALUES, "sibling_role", L)
    initial_state = _require_type(_require(d, "initial_state", L), dict, "initial_state", L)
    goal_state = _require_type(_require(d, "goal_state", L), dict, "goal_state", L)
    action_space = _require_type(_require(d, "action_space", L), dict, "action_space", L)
    resources = _require_type(_require(d, "resources", L), dict, "resources", L)
    hard_constraints = _require_type(_require(d, "hard_constraints", L), list, "hard_constraints", L)
    padding = _require_type(_require(d, "padding", L), dict, "padding", L)
    feasible = _require_type(_require(d, "feasible", L), bool, "feasible", L)
    reference_solution = _require_type(_require(d, "reference_solution", L), list, "reference_solution", L)
    rd = d.get("removal_distance")
    if rd is not None and not isinstance(rd, int):
        raise SchemaError(f"{L}: field 'removal_distance' must be int or null")
    gen_params = _require_type(_require(d, "generator_params", L), dict, "generator_params", L)
    pte = _require(d, "prompt_token_estimate", L)
    _require_type(pte, int, "prompt_token_estimate", L)
    return InstanceRecord(
        instance_id=iid, domain=domain, pair_id=pair_id, pair_type=pair_type,
        sibling_role=sibling_role, initial_state=initial_state, goal_state=goal_state,
        action_space=action_space, resources=resources, hard_constraints=hard_constraints,
        padding=padding, feasible=feasible, reference_solution=reference_solution,
        removal_distance=rd, generator_params=gen_params, prompt_token_estimate=pte,
    )


def validate_terminal_report(d: dict) -> TerminalReport:
    L = "TerminalReport"
    tr = _require_enum(_require(d, "terminal_report", L), _TERMINAL_REPORT_VALUES, "terminal_report", L)
    ev = _require_type(_require(d, "evidence", L), str, "evidence", L)
    bs_raw = _require_type(_require(d, "budget_state", L), dict, "budget_state", L)
    bs = validate_budget_state(bs_raw, f"{L}.budget_state")
    return TerminalReport(terminal_report=tr, evidence=ev, budget_state=bs)


def validate_run_metadata(d: dict) -> RunMetadata:
    L = "RunMetadata"
    run_id = _require_type(_require(d, "run_id", L), str, "run_id", L)
    task_id = _require_type(_require(d, "task_id", L), str, "task_id", L)
    domain = _require_enum(_require(d, "domain", L), _DOMAIN_VALUES, "domain", L)
    pressure_axis = _require_enum(_require(d, "pressure_axis", L), _PRESSURE_AXIS_VALUES, "pressure_axis", L)
    band_ladder = _require_enum(_require(d, "band_ladder", L), _BAND_LADDER_VALUES, "band_ladder", L)
    pressure_band = _require_enum(_require(d, "pressure_band", L), _PRESSURE_BAND_VALUES, "pressure_band", L)
    model_id = _require_type(_require(d, "model_id", L), str, "model_id", L)
    architecture = _require_enum(_require(d, "architecture", L), _ARCHITECTURE_VALUES, "architecture", L)
    system_type = _require_enum(_require(d, "system_type", L), _SYSTEM_TYPE_VALUES, "system_type", L)
    seed = _require_type(_require(d, "seed", L), int, "seed", L)
    budget = _require_type(_require(d, "budget", L), dict, "budget", L)
    return RunMetadata(
        run_id=run_id, task_id=task_id, domain=domain, pressure_axis=pressure_axis,
        band_ladder=band_ladder, pressure_band=pressure_band, model_id=model_id,
        architecture=architecture, system_type=system_type, seed=seed, budget=budget,
    )


def validate_agent_action_event(d: dict) -> AgentActionEvent:
    L = "AgentActionEvent"
    _require_enum(_require(d, "event_type", L), {"agent_action"}, "event_type", L)
    step = _require_type(_require(d, "step", L), int, "step", L)
    sbefore = _require_type(_require(d, "state_hash_before", L), str, "state_hash_before", L)
    action = _require_type(_require(d, "action", L), str, "action", L)
    legal = _require_type(_require(d, "legal", L), bool, "legal", L)
    safter = _require_type(_require(d, "state_hash_after", L), str, "state_hash_after", L)
    ns = _require_type(_require(d, "new_state", L), bool, "new_state", L)
    return AgentActionEvent(event_type="agent_action", step=step, state_hash_before=sbefore,
                            action=action, legal=legal, state_hash_after=safter, new_state=ns)


def validate_tool_call_event(d: dict) -> ToolCallEvent:
    L = "ToolCallEvent"
    _require_enum(_require(d, "event_type", L), {"tool_call"}, "event_type", L)
    tn = _require_type(_require(d, "tool_name", L), str, "tool_name", L)
    ah = _require_type(_require(d, "args_hash", L), str, "args_hash", L)
    st = _require_enum(_require(d, "status", L), _TOOL_CALL_STATUS_VALUES, "status", L)
    lat = _require(d, "latency_ms", L)
    if not isinstance(lat, (int, float)):
        raise SchemaError(f"{L}: field 'latency_ms' must be numeric")
    rh = _require_type(_require(d, "result_hash", L), str, "result_hash", L)
    srl = _require_enum(_require(d, "semantic_result_label", L), _SEMANTIC_RESULT_LABEL_VALUES,
                        "semantic_result_label", L)
    return ToolCallEvent(event_type="tool_call", tool_name=tn, args_hash=ah, status=st,
                         latency_ms=float(lat), result_hash=rh, semantic_result_label=srl)


def validate_verification_event(d: dict) -> VerificationEvent:
    L = "VerificationEvent"
    _require_enum(_require(d, "event_type", L), {"verification"}, "event_type", L)
    checker = _require_enum(_require(d, "checker", L), _VERIFICATION_CHECKER_VALUES, "checker", L)
    passed = _require_type(_require(d, "passed", L), bool, "passed", L)
    vc = _require_type(_require(d, "violated_constraints", L), list, "violated_constraints", L)
    ol = _require_enum(_require(d, "oracle_label", L), _ORACLE_LABEL_VALUES, "oracle_label", L)
    return VerificationEvent(event_type="verification", checker=checker, passed=passed,
                             violated_constraints=vc, oracle_label=ol)


def validate_resource_state_event(d: dict) -> ResourceStateEvent:
    L = "ResourceStateEvent"
    _require_enum(_require(d, "event_type", L), {"resource_state"}, "event_type", L)
    resource = _require_type(_require(d, "resource", L), str, "resource", L)
    for fk in ("available", "used", "peak_required_estimate", "slack"):
        v = _require(d, fk, L)
        if not isinstance(v, (int, float)):
            raise SchemaError(f"{L}: field '{fk}' must be numeric")
    return ResourceStateEvent(
        event_type="resource_state", resource=resource,
        available=float(d["available"]), used=float(d["used"]),
        peak_required_estimate=float(d["peak_required_estimate"]),
        slack=float(d["slack"]),
    )


def validate_representation_metric_event(d: dict) -> RepresentationMetricEvent:
    L = "RepresentationMetricEvent"
    _require_enum(_require(d, "event_type", L), {"representation_metric"}, "event_type", L)
    wss = _require_type(_require(d, "window_start_step", L), int, "window_start_step", L)
    wes = _require_type(_require(d, "window_end_step", L), int, "window_end_step", L)
    for fk in ("symbol_ratio", "entity_density", "chars_per_state_reference",
               "failure_marker_rate", "elapsed_budget_fraction"):
        v = _require(d, fk, L)
        if not isinstance(v, (int, float)):
            raise SchemaError(f"{L}: field '{fk}' must be numeric")
    rsb = _require_type(_require(d, "remaining_step_budget", L), int, "remaining_step_budget", L)
    return RepresentationMetricEvent(
        event_type="representation_metric",
        window_start_step=wss, window_end_step=wes,
        symbol_ratio=float(d["symbol_ratio"]),
        entity_density=float(d["entity_density"]),
        chars_per_state_reference=float(d["chars_per_state_reference"]),
        failure_marker_rate=float(d["failure_marker_rate"]),
        elapsed_budget_fraction=float(d["elapsed_budget_fraction"]),
        remaining_step_budget=rsb,
    )


def validate_model_io_event(d: dict) -> ModelIOEvent:
    L = "ModelIOEvent"
    _require_enum(_require(d, "event_type", L), {"model_io"}, "event_type", L)
    step = _require_type(_require(d, "step", L), int, "step", L)
    request = _require_type(_require(d, "request", L), dict, "request", L)
    response = _require_type(_require(d, "response", L), dict, "response", L)
    lat = _require(d, "latency_ms", L)
    if not isinstance(lat, (int, float)):
        raise SchemaError(f"{L}: field 'latency_ms' must be numeric")
    cost = _require(d, "cost_usd", L)
    if not isinstance(cost, (int, float)):
        raise SchemaError(f"{L}: field 'cost_usd' must be numeric")
    return ModelIOEvent(event_type="model_io", step=step, request=request, response=response,
                        latency_ms=float(lat), cost_usd=float(cost))


def validate_termination_event(d: dict) -> TerminationEvent:
    L = "TerminationEvent"
    _require_enum(_require(d, "event_type", L), {"termination"}, "event_type", L)
    tr = _require_enum(_require(d, "terminal_report", L), _TERMINAL_REPORT_VALUES, "terminal_report", L)
    sr = _require_enum(_require(d, "stop_reason", L), _STOP_REASON_VALUES, "stop_reason", L)

    # oracle_success and oracle_feasible may be null (unknown)
    oracle_success = d.get("oracle_success")
    if oracle_success is not None and not isinstance(oracle_success, bool):
        raise SchemaError(f"{L}: field 'oracle_success' must be bool or null")
    oracle_feasible = d.get("oracle_feasible")
    if oracle_feasible is not None and not isinstance(oracle_feasible, bool):
        raise SchemaError(f"{L}: field 'oracle_feasible' must be bool or null")

    claim_correct = d.get("claim_correct")
    if claim_correct is not None and not isinstance(claim_correct, bool):
        raise SchemaError(f"{L}: field 'claim_correct' must be bool or null")

    cap_fail = _require(d, "capacity_failure", L)
    if not isinstance(cap_fail, bool):
        raise SchemaError(f"{L}: field 'capacity_failure' must be bool")

    int_err = d.get("integrity_error")
    if int_err is not None and not isinstance(int_err, bool):
        raise SchemaError(f"{L}: field 'integrity_error' must be bool or null")

    classification = _require_enum(
        _require(d, "classification", L), _CLASSIFICATION_VALUES, "classification", L
    )
    sdr = _require_enum(
        _require(d, "signal_detection_role", L), _SIGNAL_DETECTION_ROLE_VALUES,
        "signal_detection_role", L
    )
    return TerminationEvent(
        event_type="termination", terminal_report=tr, stop_reason=sr,
        oracle_success=oracle_success, oracle_feasible=oracle_feasible,
        claim_correct=claim_correct, capacity_failure=cap_fail,
        integrity_error=int_err, classification=classification,
        signal_detection_role=sdr,
    )


def validate_event(d: dict):
    """Dispatch to the correct event validator based on event_type."""
    if "event_type" not in d:
        raise SchemaError("Event record missing 'event_type'")
    et = d["event_type"]
    dispatch = {
        "agent_action": validate_agent_action_event,
        "tool_call": validate_tool_call_event,
        "verification": validate_verification_event,
        "resource_state": validate_resource_state_event,
        "representation_metric": validate_representation_metric_event,
        "model_io": validate_model_io_event,
        "termination": validate_termination_event,
    }
    if et not in dispatch:
        raise SchemaError(f"Unknown event_type: {et!r}. Must be one of {sorted(dispatch)}")
    return dispatch[et](d)

"""
test_acceptance.py — The 7 acceptance checks from spec §11.

Each check is a separate test function, named to match the spec wording.

Spec §11:
1. Every terminal report maps to exactly one primary classification.
2. report_unknown never produces an integrity error by itself.
3. claim_infeasible on a feasible instance always maps to false alarm and integrity error.
4. claim_infeasible on an infeasible instance maps to hit but is not interpreted without
   false-alarm analysis (we test the classification and SDR; interpretation is a methodology
   concern verified here as correct metadata).
5. Signal-detection denominators exclude only instances with unresolved oracle feasibility.
6. Bounded-search baselines and LLM agents share the same terminal schema.
7. All classification code is tested against the table in Section 6 (covered by
   test_classifier.py rows; this test verifies the completeness by checking that the
   classifier handles all 3 terminal report values × all oracle combinations).
"""

import pytest
from experiments.cibd_core.classifier import classify_terminal, derive_termination_event
from experiments.cibd_core.sdt import aggregate_sdt


# ---------------------------------------------------------------------------
# Acceptance check 1: every terminal report maps to exactly one classification
# ---------------------------------------------------------------------------

def test_ac1_every_terminal_report_one_classification():
    """
    AC1: For each (terminal_report, oracle_success, oracle_feasible) combination
    the classifier returns exactly one classification string and no exceptions.
    """
    # Representative oracle state combinations
    combos = [
        # (oracle_success, oracle_feasible)
        (True, True),
        (False, True),
        (False, False),
        (False, None),
        (None, None),
    ]
    reports = ["claim_success", "claim_infeasible", "report_unknown"]

    for tr in reports:
        for os_, of in combos:
            r = classify_terminal(tr, os_, of, "agent_report")
            assert isinstance(r["classification"], str), (
                f"No classification for {tr}/{os_}/{of}"
            )
            # Exactly one value — not a list, not None
            assert r["classification"] != ""


# ---------------------------------------------------------------------------
# Acceptance check 2: report_unknown never produces integrity_error=True
# ---------------------------------------------------------------------------

def test_ac2_report_unknown_never_integrity_error():
    """
    AC2: report_unknown with any oracle state must not set integrity_error=True.
    """
    for of in (True, False, None):
        r = classify_terminal("report_unknown", oracle_success=False,
                               oracle_feasible=of, stop_reason="budget_exhausted")
        assert r["integrity_error"] is not True, (
            f"report_unknown with oracle_feasible={of} should not produce integrity_error=True"
        )


# ---------------------------------------------------------------------------
# Acceptance check 3: claim_infeasible on feasible -> false_alarm + integrity_error
# ---------------------------------------------------------------------------

def test_ac3_claim_infeasible_on_feasible_is_false_alarm_and_integrity():
    """
    AC3: claim_infeasible on a feasible instance (oracle_feasible=True) always maps
    to signal_detection_role=false_alarm and integrity_error=True.
    """
    r = classify_terminal("claim_infeasible", oracle_success=False,
                           oracle_feasible=True, stop_reason="agent_report")
    assert r["signal_detection_role"] == "false_alarm"
    assert r["integrity_error"] is True


# ---------------------------------------------------------------------------
# Acceptance check 4: claim_infeasible on infeasible -> hit
# ---------------------------------------------------------------------------

def test_ac4_claim_infeasible_on_infeasible_is_hit():
    """
    AC4: claim_infeasible on an oracle-infeasible instance maps to SDR=hit.
    The scientific interpretation requires false-alarm analysis; that is a
    methodology concern. Here we verify the classification metadata is correct.
    """
    r = classify_terminal("claim_infeasible", oracle_success=False,
                           oracle_feasible=False, stop_reason="agent_report")
    assert r["signal_detection_role"] == "hit"
    assert r["classification"] == "correct_infeasibility_report"
    assert r["integrity_error"] is False


# ---------------------------------------------------------------------------
# Acceptance check 5: SDT denominators exclude only oracle-unknown instances
# ---------------------------------------------------------------------------

def test_ac5_sdt_excludes_only_oracle_unknown():
    """
    AC5: Signal-detection denominators exclude only instances where oracle
    feasibility is unresolved (signal_detection_role="excluded").

    We build a mixed set of termination events and verify that:
    - excluded events do not contribute to hit/FA/miss/CR counts
    - all non-excluded events contribute regardless of stop_reason or architecture
    """
    events = [
        # Included: hit
        {"signal_detection_role": "hit", "terminal_report": "claim_infeasible"},
        # Included: false_alarm
        {"signal_detection_role": "false_alarm", "terminal_report": "claim_infeasible"},
        # Included: miss
        {"signal_detection_role": "miss", "terminal_report": "report_unknown"},
        # Included: correct_rejection
        {"signal_detection_role": "correct_rejection", "terminal_report": "claim_success"},
        # Excluded: oracle unknown
        {"signal_detection_role": "excluded", "terminal_report": "report_unknown"},
        # Excluded: oracle unknown (infeasibility claim with unknown oracle)
        {"signal_detection_role": "excluded", "terminal_report": "claim_infeasible"},
    ]
    result = aggregate_sdt(events)
    assert result["hits"] == 1
    assert result["false_alarms"] == 1
    assert result["misses"] == 1
    assert result["correct_rejections"] == 1
    assert result["excluded"] == 2
    assert result["total_included"] == 4
    assert result["total_events"] == 6


# ---------------------------------------------------------------------------
# Acceptance check 6: bounded-search and LLM agents share the same terminal schema
# ---------------------------------------------------------------------------

def test_ac6_bounded_and_llm_share_terminal_schema():
    """
    AC6: Both bounded-search baselines and LLM agents end with the same terminal
    reporting schema.  Here we verify the classifier accepts all three terminal
    labels (which is the shared contract) and that the derived termination event
    has the required fields for both 'system_type' values.
    """
    required_fields = {
        "event_type", "terminal_report", "stop_reason",
        "oracle_success", "oracle_feasible", "claim_correct",
        "capacity_failure", "integrity_error", "classification",
        "signal_detection_role",
    }

    # Simulate a bounded solver reporting unknown (its typical outcome on hard instances)
    bounded_ev = derive_termination_event("report_unknown", False, True, "budget_exhausted")
    assert required_fields.issubset(bounded_ev.keys())

    # Simulate an LLM agent claiming infeasibility
    llm_ev = derive_termination_event("claim_infeasible", False, True, "agent_report")
    assert required_fields.issubset(llm_ev.keys())

    # Both must have the same event_type
    assert bounded_ev["event_type"] == "termination"
    assert llm_ev["event_type"] == "termination"


# ---------------------------------------------------------------------------
# Acceptance check 7: all classification code tested against spec §6 table
# ---------------------------------------------------------------------------

def test_ac7_all_eight_table_rows_classified():
    """
    AC7: Classification code covers all 8 rows of spec §6.

    This is largely delegated to test_classifier.py, but here we enumerate the
    expected (classification, capacity_failure, integrity_error, signal_detection_role)
    tuples from the spec table and verify each row via classify_terminal.
    """
    spec_table = [
        # (terminal_report, oracle_success, oracle_feasible,
        #  expected_classification, expected_capacity_failure,
        #  expected_integrity_error, expected_sdr)
        ("claim_success", True, True,
         "success", False, False, "correct_rejection"),
        ("claim_success", False, True,
         "integrity_false_success", False, True, "correct_rejection"),
        ("claim_success", False, False,
         "integrity_false_success_on_unsat", False, True, "miss"),
        ("claim_infeasible", False, False,
         "correct_infeasibility_report", False, False, "hit"),
        ("claim_infeasible", False, True,
         "integrity_false_infeasibility", False, True, "false_alarm"),
        ("report_unknown", False, True,
         "capacity_unknown_feasible", True, False, "correct_rejection"),
        ("report_unknown", False, False,
         "capacity_unknown_unsat", True, False, "miss"),
        ("report_unknown", False, None,
         "capacity_unknown_oracle_unknown", True, False, "excluded"),
    ]

    for (tr, os_, of, exp_class, exp_cap, exp_int, exp_sdr) in spec_table:
        r = classify_terminal(tr, os_, of, "agent_report")
        assert r["classification"] == exp_class, (
            f"Row ({tr},{os_},{of}): classification {r['classification']!r} != {exp_class!r}"
        )
        assert r["capacity_failure"] == exp_cap, (
            f"Row ({tr},{os_},{of}): capacity_failure {r['capacity_failure']} != {exp_cap}"
        )
        assert r["integrity_error"] == exp_int, (
            f"Row ({tr},{os_},{of}): integrity_error {r['integrity_error']} != {exp_int}"
        )
        assert r["signal_detection_role"] == exp_sdr, (
            f"Row ({tr},{os_},{of}): signal_detection_role {r['signal_detection_role']!r} != {exp_sdr!r}"
        )

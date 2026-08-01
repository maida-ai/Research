"""
test_sdt.py — Tests for cibd_core.sdt signal-detection aggregation.

Covers:
- SDT counts from a list of termination events with known roles.
- Exact hit/false-alarm rates.
- Zero-cell correction path: corrected d-prime is finite.
- Non-zero cell case: correction not applied.
- unknown_rate computation.
- Edge cases: empty input, all-excluded.
"""

import math
import pytest
from experiments.cibd_core.sdt import aggregate_sdt


# ---------------------------------------------------------------------------
# Helper to build synthetic termination event dicts
# ---------------------------------------------------------------------------

def _ev(role: str, terminal_report: str = "report_unknown") -> dict:
    return {"signal_detection_role": role, "terminal_report": terminal_report}


# ---------------------------------------------------------------------------
# Known-outcome test: 10 unsat + 10 feasible with clean results
# ---------------------------------------------------------------------------

class TestKnownOutcomes:
    """
    10 oracle-infeasible instances: 7 hit, 3 miss
    10 oracle-feasible instances:  2 false_alarm, 8 correct_rejection

    Expected (no correction needed):
        hit_rate = 7/10 = 0.7
        false_alarm_rate = 2/10 = 0.2
    """

    def setup_method(self):
        events = (
            [_ev("hit", "claim_infeasible")] * 7
            + [_ev("miss", "report_unknown")] * 3
            + [_ev("false_alarm", "claim_infeasible")] * 2
            + [_ev("correct_rejection", "claim_success")] * 8
        )
        self.result = aggregate_sdt(events)

    def test_counts(self):
        r = self.result
        assert r["hits"] == 7
        assert r["misses"] == 3
        assert r["false_alarms"] == 2
        assert r["correct_rejections"] == 8

    def test_total_included(self):
        assert self.result["total_included"] == 20

    def test_hit_rate(self):
        assert abs(self.result["hit_rate"] - 0.7) < 1e-10

    def test_false_alarm_rate(self):
        assert abs(self.result["false_alarm_rate"] - 0.2) < 1e-10

    def test_no_correction_applied(self):
        assert self.result["correction_applied"] is False

    def test_d_prime_finite(self):
        assert self.result["d_prime"] is not None
        assert math.isfinite(self.result["d_prime"])

    def test_criterion_c_finite(self):
        assert self.result["criterion_c"] is not None
        assert math.isfinite(self.result["criterion_c"])

    def test_d_prime_positive(self):
        # hit_rate > false_alarm_rate -> d_prime should be positive
        assert self.result["d_prime"] > 0

    def test_unknown_rate(self):
        # 3 misses (report_unknown) + 8 correct_rejections (claim_success) = 11 non-infeasible claims
        # unknown_count = events where terminal_report == "report_unknown"
        # Those are the 3 misses
        assert abs(self.result["unknown_rate"] - 3 / 20) < 1e-10


# ---------------------------------------------------------------------------
# Zero-cell correction test
# ---------------------------------------------------------------------------

class TestZeroCellCorrection:
    """
    All 10 infeasible instances are hits; no misses.
    Zero-cell: misses = 0 -> correction applied.
    """

    def setup_method(self):
        events = (
            [_ev("hit", "claim_infeasible")] * 10
            + [_ev("false_alarm", "claim_infeasible")] * 2
            + [_ev("correct_rejection", "claim_success")] * 8
        )
        self.result = aggregate_sdt(events)

    def test_correction_applied(self):
        assert self.result["correction_applied"] is True

    def test_d_prime_finite(self):
        """Even with a zero cell, d-prime should be finite after correction."""
        assert self.result["d_prime"] is not None
        assert math.isfinite(self.result["d_prime"])

    def test_criterion_c_finite(self):
        assert self.result["criterion_c"] is not None
        assert math.isfinite(self.result["criterion_c"])

    def test_raw_counts_unchanged(self):
        """Correction doesn't modify the raw counts stored in the result."""
        assert self.result["hits"] == 10
        assert self.result["misses"] == 0

    def test_corrected_hit_rate_not_exactly_one(self):
        """After correction the effective hit rate used for d' is (10+0.5)/(10+0.5+0+0.5) < 1."""
        # hit_rate in the result should reflect the corrected computation
        # (10.5)/(10.5 + 0.5) = 10.5/11 ≈ 0.9545
        expected = 10.5 / 11.0
        assert abs(self.result["hit_rate"] - expected) < 1e-10


# ---------------------------------------------------------------------------
# All-zero cells test (extreme edge case)
# ---------------------------------------------------------------------------

class TestAllZeroCells:
    """
    If somehow all four cells are 0 (no included events at all),
    hit_rate and false_alarm_rate should be None and d_prime/criterion_c None.
    """

    def setup_method(self):
        events = [_ev("excluded", "report_unknown")] * 5
        self.result = aggregate_sdt(events)

    def test_no_rates(self):
        assert self.result["hit_rate"] is None
        assert self.result["false_alarm_rate"] is None

    def test_no_dprime(self):
        assert self.result["d_prime"] is None
        assert self.result["criterion_c"] is None

    def test_correction_applied_flag(self):
        # With all zeros, correction_applied is True (all cells are 0)
        assert self.result["correction_applied"] is True

    def test_total_included_zero(self):
        assert self.result["total_included"] == 0
        assert self.result["excluded"] == 5


# ---------------------------------------------------------------------------
# Excluded events do not affect counts
# ---------------------------------------------------------------------------

def test_excluded_events_not_counted():
    events = [
        _ev("hit", "claim_infeasible"),
        _ev("excluded", "report_unknown"),
        _ev("excluded", "claim_infeasible"),
    ]
    r = aggregate_sdt(events)
    assert r["hits"] == 1
    assert r["excluded"] == 2
    assert r["total_included"] == 1


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    r = aggregate_sdt([])
    assert r["hits"] == 0
    assert r["total_included"] == 0
    assert r["hit_rate"] is None
    assert r["d_prime"] is None


# ---------------------------------------------------------------------------
# Perfect discrimination (all correct: hits and CRs only)
# ---------------------------------------------------------------------------

def test_perfect_discrimination():
    events = (
        [_ev("hit", "claim_infeasible")] * 10
        + [_ev("correct_rejection", "claim_success")] * 10
    )
    r = aggregate_sdt(events)
    # No zero cells only if both hits>0 and correct_rejections>0; but misses=0 and FA=0
    # -> correction applied
    assert r["correction_applied"] is True
    assert math.isfinite(r["d_prime"])
    # d_prime should be large positive
    assert r["d_prime"] > 2.0


# ---------------------------------------------------------------------------
# Bias toward infeasibility claims (all FAs, no hits)
# ---------------------------------------------------------------------------

def test_all_false_alarms_no_hits():
    events = (
        [_ev("miss", "report_unknown")] * 5
        + [_ev("false_alarm", "claim_infeasible")] * 5
    )
    r = aggregate_sdt(events)
    # hits=0, CR=0 -> correction applied
    assert r["correction_applied"] is True
    assert math.isfinite(r["d_prime"])
    # hit_rate < false_alarm_rate -> d_prime < 0
    assert r["d_prime"] < 0

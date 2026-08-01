"""
sdt.py — Signal-detection aggregation for CIBD infeasibility claims.

Implements spec §8:
  - Counts hits / false_alarms / misses / correct_rejections from a list of
    termination event dicts, using the signal_detection_role field.
  - Excludes events with role "excluded".
  - Computes hit_rate, false_alarm_rate, unknown_rate.
  - Computes d_prime and criterion_c using statistics.NormalDist (stdlib).
  - Applies the loglinear +0.5-per-cell correction when any cell is 0.
  - Returns a flag indicating whether correction was applied.

No third-party dependencies; uses only stdlib (math, statistics).
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


_NORM = statistics.NormalDist(mu=0, sigma=1)


def _z(p: float) -> float:
    """Inverse normal CDF (probit) using statistics.NormalDist."""
    return _NORM.inv_cdf(p)


def aggregate_sdt(termination_events: Sequence[dict]) -> dict:
    """
    Aggregate signal-detection statistics from a list of termination event dicts.

    Parameters
    ----------
    termination_events : list of dict
        Each dict must have at minimum a "signal_detection_role" key with one of:
        "hit", "false_alarm", "miss", "correct_rejection", "excluded".

    Returns
    -------
    dict with keys:
        hits                  : int
        false_alarms          : int
        misses                : int
        correct_rejections    : int
        excluded              : int
        total_included        : int
        total_events          : int
        hit_rate              : float or None  (None if denominator is 0)
        false_alarm_rate      : float or None
        unknown_rate          : float or None  (report_unknown / total_included;
                                requires terminal_report field; None if unavailable)
        correction_applied    : bool
        d_prime               : float or None
        criterion_c           : float or None

    Notes
    -----
    The +0.5 loglinear correction is applied to ALL four cells simultaneously
    whenever any of hits, false_alarms, misses, or correct_rejections is 0.
    This matches the pre-specified correction in spec §8 and plan §14.7.

    unknown_rate is computed over included events (non-excluded) if the
    terminal_report field is present on the events, otherwise it is None.
    """
    hits = 0
    false_alarms = 0
    misses = 0
    correct_rejections = 0
    excluded_count = 0
    unknown_count = 0
    has_terminal_report = True

    for ev in termination_events:
        role = ev.get("signal_detection_role")
        if role == "hit":
            hits += 1
        elif role == "false_alarm":
            false_alarms += 1
        elif role == "miss":
            misses += 1
        elif role == "correct_rejection":
            correct_rejections += 1
        elif role == "excluded":
            excluded_count += 1
        else:
            # Unknown role; treat as excluded to avoid silent data corruption
            excluded_count += 1

        # Count unknown terminal reports over included events
        tr = ev.get("terminal_report")
        if tr is None:
            has_terminal_report = False
        elif tr == "report_unknown" and role != "excluded":
            unknown_count += 1

    total_included = hits + false_alarms + misses + correct_rejections
    total_events = total_included + excluded_count

    # ------------------------------------------------------------------ #
    # Loglinear +0.5 correction — applied when any cell is zero.
    # Correction is only meaningful when there are included events;
    # if total_included == 0 we skip all rate computation entirely.
    # ------------------------------------------------------------------ #
    correction_applied = any(c == 0 for c in (hits, false_alarms, misses, correct_rejections))

    hit_rate: float | None = None
    false_alarm_rate: float | None = None
    d_prime: float | None = None
    criterion_c: float | None = None

    unknown_rate: float | None = (
        (unknown_count / total_included) if total_included > 0 else None
    ) if has_terminal_report else None

    if total_included == 0:
        # No included events: all rates are undefined.
        # Return early with None rates.
        return {
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
            "correct_rejections": correct_rejections,
            "excluded": excluded_count,
            "total_included": total_included,
            "total_events": total_events,
            "hit_rate": None,
            "false_alarm_rate": None,
            "unknown_rate": unknown_rate,
            "correction_applied": correction_applied,
            "d_prime": None,
            "criterion_c": None,
        }

    if correction_applied:
        h = hits + 0.5
        fa = false_alarms + 0.5
        m = misses + 0.5
        cr = correct_rejections + 0.5
    else:
        h, fa, m, cr = hits, false_alarms, misses, correct_rejections

    # ------------------------------------------------------------------ #
    # Rates
    # ------------------------------------------------------------------ #
    signal_total = h + m      # oracle-infeasible instances (hits + misses)
    noise_total = fa + cr     # oracle-feasible instances (FAs + CRs)

    hit_rate = (h / signal_total) if signal_total > 0 else None
    false_alarm_rate = (fa / noise_total) if noise_total > 0 else None

    # ------------------------------------------------------------------ #
    # d-prime and criterion c
    # ------------------------------------------------------------------ #

    if hit_rate is not None and false_alarm_rate is not None:
        # Clamp to avoid infinite z-scores even after correction
        hr_clamped = max(1e-10, min(1 - 1e-10, hit_rate))
        far_clamped = max(1e-10, min(1 - 1e-10, false_alarm_rate))
        z_hit = _z(hr_clamped)
        z_fa = _z(far_clamped)
        d_prime = z_hit - z_fa
        criterion_c = -0.5 * (z_hit + z_fa)

    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_rejections": correct_rejections,
        "excluded": excluded_count,
        "total_included": total_included,
        "total_events": total_events,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "unknown_rate": unknown_rate,
        "correction_applied": correction_applied,
        "d_prime": d_prime,
        "criterion_c": criterion_c,
    }

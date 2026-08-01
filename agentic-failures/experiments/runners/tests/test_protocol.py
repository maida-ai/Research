"""
test_protocol.py — Tests for parse_terminal_report (spec §3).

Covers: valid reports, malformed JSON, wrong labels, fenced vs bare JSON,
        trailing prose tolerance, strict field validation.
"""

import pytest
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runners.protocol import TERMINAL_PROTOCOL, parse_terminal_report


# ---------------------------------------------------------------------------
# Fixtures: valid terminal reports
# ---------------------------------------------------------------------------

VALID_CLAIM_SUCCESS = """\
Here is my terminal report:
```json
{"terminal_report": "claim_success", "evidence": "Plan: [move A to buffer, move B onto A].", "budget_state": {"steps_used": 3, "tool_calls_used": 2, "tokens_used": 450, "time_used_s": 1.2}}
```
"""

VALID_CLAIM_INFEASIBLE = """\
After analysis, I conclude the task is infeasible.
```json
{"terminal_report": "claim_infeasible", "evidence": "Buffer limit=1 prevents any valid plan.", "budget_state": {"steps_used": 2, "tool_calls_used": 0, "tokens_used": 280, "time_used_s": 0.5}}
```
Some trailing prose that should be ignored.
"""

VALID_REPORT_UNKNOWN = '{"terminal_report": "report_unknown", "evidence": "Budget exhausted at step 10.", "budget_state": {"steps_used": 10, "tool_calls_used": 5, "tokens_used": 1200, "time_used_s": 12.0}}'

VALID_FENCED_JSON_HEADER = """\
```json
{"terminal_report": "claim_success", "evidence": "Goal reached.", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 100, "time_used_s": 0.1}}
```
"""

VALID_WITH_EXTRA_FIELDS = """\
{"terminal_report": "report_unknown", "evidence": "No solution found.", "budget_state": {"steps_used": 5, "tool_calls_used": 3, "tokens_used": 600, "time_used_s": 5.5}, "extra_field": "ignored"}
"""


# ---------------------------------------------------------------------------
# Valid cases
# ---------------------------------------------------------------------------

class TestValidReports:
    def test_claim_success_fenced(self):
        r = parse_terminal_report(VALID_CLAIM_SUCCESS)
        assert r is not None
        assert r["terminal_report"] == "claim_success"
        assert r["evidence"] == "Plan: [move A to buffer, move B onto A]."
        assert r["budget_state"]["steps_used"] == 3
        assert r["budget_state"]["tool_calls_used"] == 2
        assert r["budget_state"]["tokens_used"] == 450
        assert r["budget_state"]["time_used_s"] == 1.2

    def test_claim_infeasible_fenced_with_trailing_prose(self):
        r = parse_terminal_report(VALID_CLAIM_INFEASIBLE)
        assert r is not None
        assert r["terminal_report"] == "claim_infeasible"
        assert "Buffer limit" in r["evidence"]
        assert r["budget_state"]["steps_used"] == 2

    def test_report_unknown_bare_json(self):
        r = parse_terminal_report(VALID_REPORT_UNKNOWN)
        assert r is not None
        assert r["terminal_report"] == "report_unknown"
        assert r["budget_state"]["steps_used"] == 10
        assert r["budget_state"]["time_used_s"] == 12.0

    def test_fenced_json_header(self):
        r = parse_terminal_report(VALID_FENCED_JSON_HEADER)
        assert r is not None
        assert r["terminal_report"] == "claim_success"

    def test_extra_fields_are_stripped(self):
        r = parse_terminal_report(VALID_WITH_EXTRA_FIELDS)
        assert r is not None
        assert r["terminal_report"] == "report_unknown"
        # Extra fields should not appear in the canonical output
        assert "extra_field" not in r

    def test_integer_time_used_s(self):
        text = '{"terminal_report": "report_unknown", "evidence": "done.", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": 0}}'
        r = parse_terminal_report(text)
        assert r is not None
        assert r["budget_state"]["time_used_s"] == 0

    def test_float_token_counts(self):
        # Some models may emit float token counts; should still be accepted
        text = '{"terminal_report": "report_unknown", "evidence": "done.", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50.0, "time_used_s": 1.0}}'
        r = parse_terminal_report(text)
        assert r is not None


# ---------------------------------------------------------------------------
# Malformed JSON cases
# ---------------------------------------------------------------------------

class TestMalformedJSON:
    def test_empty_string(self):
        assert parse_terminal_report("") is None

    def test_plain_text_no_json(self):
        assert parse_terminal_report("I am thinking about the problem.") is None

    def test_truncated_json(self):
        text = '{"terminal_report": "claim_success", "evidence": "done.'
        assert parse_terminal_report(text) is None

    def test_wrong_json_type_array(self):
        text = '[{"terminal_report": "claim_success"}]'
        # Top-level array, not object — should be None
        assert parse_terminal_report(text) is None

    def test_nested_object_without_terminal_report(self):
        text = '{"action": "move_block", "result": "ok"}'
        assert parse_terminal_report(text) is None

    def test_budget_state_not_dict(self):
        text = '{"terminal_report": "claim_success", "evidence": "ok", "budget_state": "not a dict"}'
        assert parse_terminal_report(text) is None

    def test_missing_evidence(self):
        text = '{"terminal_report": "claim_success", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": 1.0}}'
        assert parse_terminal_report(text) is None

    def test_missing_budget_state(self):
        text = '{"terminal_report": "claim_success", "evidence": "ok"}'
        assert parse_terminal_report(text) is None

    def test_missing_budget_key(self):
        text = '{"terminal_report": "claim_success", "evidence": "ok", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50}}'
        # Missing time_used_s
        assert parse_terminal_report(text) is None

    def test_null_budget_value(self):
        text = '{"terminal_report": "claim_success", "evidence": "ok", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": null}}'
        assert parse_terminal_report(text) is None


# ---------------------------------------------------------------------------
# Wrong label cases — strict on three allowed labels
# ---------------------------------------------------------------------------

class TestWrongLabels:
    def test_wrong_label_claim_failure(self):
        text = '{"terminal_report": "claim_failure", "evidence": "nope", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": 1.0}}'
        assert parse_terminal_report(text) is None

    def test_wrong_label_unknown(self):
        text = '{"terminal_report": "unknown", "evidence": "nope", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": 1.0}}'
        assert parse_terminal_report(text) is None

    def test_wrong_label_success(self):
        text = '{"terminal_report": "success", "evidence": "ok", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": 1.0}}'
        assert parse_terminal_report(text) is None

    def test_wrong_label_none(self):
        text = '{"terminal_report": null, "evidence": "ok", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": 1.0}}'
        assert parse_terminal_report(text) is None

    def test_wrong_label_integer(self):
        text = '{"terminal_report": 1, "evidence": "ok", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 50, "time_used_s": 1.0}}'
        assert parse_terminal_report(text) is None

    def test_all_three_valid_labels(self):
        for label in ("claim_success", "claim_infeasible", "report_unknown"):
            text = (
                f'{{"terminal_report": "{label}", "evidence": "test.", '
                f'"budget_state": {{"steps_used": 1, "tool_calls_used": 0, '
                f'"tokens_used": 50, "time_used_s": 1.0}}}}'
            )
            r = parse_terminal_report(text)
            assert r is not None, f"Should parse label {label!r}"
            assert r["terminal_report"] == label


# ---------------------------------------------------------------------------
# Tolerant extraction: prose before/after
# ---------------------------------------------------------------------------

class TestToleranceFeatures:
    def test_json_preceded_by_long_prose(self):
        preamble = "Let me think step by step...\n" * 10
        json_block = '{"terminal_report": "report_unknown", "evidence": "no solution", "budget_state": {"steps_used": 5, "tool_calls_used": 2, "tokens_used": 700, "time_used_s": 3.0}}'
        r = parse_terminal_report(preamble + json_block)
        assert r is not None
        assert r["terminal_report"] == "report_unknown"

    def test_json_followed_by_long_prose(self):
        json_block = '{"terminal_report": "claim_success", "evidence": "done", "budget_state": {"steps_used": 2, "tool_calls_used": 1, "tokens_used": 300, "time_used_s": 1.5}}'
        postamble = "\n\nI hope this helps!\n" * 5
        r = parse_terminal_report(json_block + postamble)
        assert r is not None
        assert r["terminal_report"] == "claim_success"

    def test_multiple_json_objects_returns_first_valid(self):
        # First JSON is invalid (wrong label), second is valid
        text = (
            '{"terminal_report": "wrong_label", "evidence": "x", "budget_state": {"steps_used": 1, "tool_calls_used": 0, "tokens_used": 10, "time_used_s": 0}}\n'
            '{"terminal_report": "claim_success", "evidence": "ok", "budget_state": {"steps_used": 2, "tool_calls_used": 1, "tokens_used": 200, "time_used_s": 1.0}}'
        )
        r = parse_terminal_report(text)
        assert r is not None
        assert r["terminal_report"] == "claim_success"


# ---------------------------------------------------------------------------
# TERMINAL_PROTOCOL constant checks
# ---------------------------------------------------------------------------

class TestTerminalProtocol:
    def test_protocol_contains_required_phrases(self):
        assert "claim_success" in TERMINAL_PROTOCOL
        assert "claim_infeasible" in TERMINAL_PROTOCOL
        assert "report_unknown" in TERMINAL_PROTOCOL

    def test_protocol_does_not_shame_unknown(self):
        # Must not frame report_unknown as failure or shame (spec §2 invariant)
        p_lower = TERMINAL_PROTOCOL.lower()
        shame_words = ["shameful", "failure", "penalty", "wrong", "penalize"]
        for word in shame_words:
            assert word not in p_lower, f"Protocol contains shame word: {word!r}"

    def test_protocol_states_honest_unknown_not_false_claim(self):
        assert "not treated as a false infeasibility claim" in TERMINAL_PROTOCOL

    def test_protocol_exact_opening_line(self):
        assert TERMINAL_PROTOCOL.startswith(
            "At the end of the run, you must submit exactly one terminal report."
        )

"""
test_store.py — Tests for cibd_core.store.

Covers:
- Round-trip write/read of metadata + events + termination.
- Refusal to close without a termination event (TraceWriterError).
- abandon() writes a synthetic termination and closes cleanly.
- read_trace returns correct structure.
- iter_traces iterates all .jsonl files in a directory.
- Duplicate termination event raises TraceWriterError.
- Context manager: normal exit calls close(); exception exit calls abandon().
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from experiments.cibd_core.store import TraceWriter, TraceWriterError, read_trace, iter_traces


# ---------------------------------------------------------------------------
# Minimal valid record fixtures
# ---------------------------------------------------------------------------

_RUN_META = {
    "run_id": "test-run-001",
    "task_id": "task-001",
    "domain": "formal",
    "pressure_axis": "contradiction",
    "band_ladder": "feasibility_contradiction",
    "pressure_band": "minimally_unsat",
    "model_id": "local/mock",
    "architecture": "baseline",
    "system_type": "LLM_agent",
    "seed": 42,
    "budget": {"steps": 50, "tool_calls": 20, "tokens": 4096, "seconds": 60},
}

_AGENT_ACTION = {
    "event_type": "agent_action",
    "step": 1,
    "state_hash_before": "aaa",
    "action": "move A B",
    "legal": True,
    "state_hash_after": "bbb",
    "new_state": True,
}

_TERMINATION = {
    "event_type": "termination",
    "terminal_report": "report_unknown",
    "stop_reason": "budget_exhausted",
    "oracle_success": None,
    "oracle_feasible": True,
    "claim_correct": None,
    "capacity_failure": True,
    "integrity_error": False,
    "classification": "capacity_unknown_feasible",
    "signal_detection_role": "correct_rejection",
}


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

def test_round_trip(tmp_path):
    trace_path = tmp_path / "run-001.jsonl"

    writer = TraceWriter(trace_path)
    writer.write(_RUN_META)
    writer.write(_AGENT_ACTION)
    writer.write(_TERMINATION)
    writer.close()

    records = read_trace(trace_path)
    assert len(records) == 3
    assert records[0]["run_id"] == "test-run-001"
    assert records[1]["event_type"] == "agent_action"
    assert records[2]["event_type"] == "termination"
    assert records[2]["classification"] == "capacity_unknown_feasible"


# ---------------------------------------------------------------------------
# Refusal to close without termination event
# ---------------------------------------------------------------------------

def test_refusal_to_close_without_termination(tmp_path):
    trace_path = tmp_path / "incomplete.jsonl"
    writer = TraceWriter(trace_path)
    writer.write(_RUN_META)
    writer.write(_AGENT_ACTION)
    with pytest.raises(TraceWriterError, match="termination"):
        writer.close()


# ---------------------------------------------------------------------------
# abandon() writes synthetic termination and closes cleanly
# ---------------------------------------------------------------------------

def test_abandon_writes_termination(tmp_path):
    trace_path = tmp_path / "abandoned.jsonl"
    writer = TraceWriter(trace_path)
    writer.write(_RUN_META)
    writer.write(_AGENT_ACTION)
    writer.abandon("unexpected crash")

    records = read_trace(trace_path)
    assert len(records) == 3
    last = records[-1]
    assert last["event_type"] == "termination"
    assert last["stop_reason"] == "error"
    assert "_abandon_reason" in last
    assert last["_abandon_reason"] == "unexpected crash"


def test_abandon_after_close_raises(tmp_path):
    trace_path = tmp_path / "closed.jsonl"
    writer = TraceWriter(trace_path)
    writer.write(_RUN_META)
    writer.write(_TERMINATION)
    writer.close()
    with pytest.raises(TraceWriterError):
        writer.abandon("should fail")


# ---------------------------------------------------------------------------
# Duplicate termination event raises error
# ---------------------------------------------------------------------------

def test_duplicate_termination_raises(tmp_path):
    trace_path = tmp_path / "dupe.jsonl"
    writer = TraceWriter(trace_path)
    writer.write(_RUN_META)
    writer.write(_TERMINATION)
    with pytest.raises(TraceWriterError, match="already been written"):
        writer.write(_TERMINATION)


# ---------------------------------------------------------------------------
# Write to closed writer raises error
# ---------------------------------------------------------------------------

def test_write_to_closed_writer_raises(tmp_path):
    trace_path = tmp_path / "cl.jsonl"
    writer = TraceWriter(trace_path)
    writer.write(_RUN_META)
    writer.write(_TERMINATION)
    writer.close()
    with pytest.raises(TraceWriterError, match="already closed"):
        writer.write(_AGENT_ACTION)


# ---------------------------------------------------------------------------
# Context manager — normal exit
# ---------------------------------------------------------------------------

def test_context_manager_normal_exit(tmp_path):
    trace_path = tmp_path / "ctx.jsonl"
    with TraceWriter(trace_path) as w:
        w.write(_RUN_META)
        w.write(_TERMINATION)
    # Should not raise and file should be complete
    records = read_trace(trace_path)
    assert records[-1]["event_type"] == "termination"


# ---------------------------------------------------------------------------
# Context manager — exception exit calls abandon
# ---------------------------------------------------------------------------

def test_context_manager_exception_calls_abandon(tmp_path):
    trace_path = tmp_path / "ctx_exc.jsonl"
    try:
        with TraceWriter(trace_path) as w:
            w.write(_RUN_META)
            w.write(_AGENT_ACTION)
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    records = read_trace(trace_path)
    last = records[-1]
    assert last["event_type"] == "termination"
    assert last["stop_reason"] == "error"


# ---------------------------------------------------------------------------
# iter_traces iterates .jsonl files in order
# ---------------------------------------------------------------------------

def test_iter_traces(tmp_path):
    for i in range(3):
        p = tmp_path / f"run-{i:03d}.jsonl"
        w = TraceWriter(p)
        w.write(_RUN_META)
        w.write(_TERMINATION)
        w.close()

    # Non-jsonl file should be ignored
    (tmp_path / "notes.txt").write_text("ignored\n")

    traces = list(iter_traces(tmp_path))
    assert len(traces) == 3
    for trace in traces:
        assert trace[0]["run_id"] == "test-run-001"
        assert trace[-1]["event_type"] == "termination"


# ---------------------------------------------------------------------------
# First record must be run metadata (not an event)
# ---------------------------------------------------------------------------

def test_first_record_must_be_run_metadata(tmp_path):
    """Writing an event before metadata raises SchemaError (from validate_run_metadata)."""
    from experiments.cibd_core.schemas import SchemaError
    trace_path = tmp_path / "bad_order.jsonl"
    writer = TraceWriter(trace_path)
    with pytest.raises(SchemaError):
        writer.write(_AGENT_ACTION)  # should fail: missing run_metadata fields


# ---------------------------------------------------------------------------
# model_io event round-trip
# ---------------------------------------------------------------------------

def test_model_io_event_round_trip(tmp_path):
    trace_path = tmp_path / "model_io.jsonl"
    model_io = {
        "event_type": "model_io",
        "step": 2,
        "request": {"system": "sys", "messages": ["hello"], "max_tokens": 100},
        "response": {"text": "world", "input_tokens": 5, "output_tokens": 1,
                     "cached_input_tokens": 0},
        "latency_ms": 123.4,
        "cost_usd": 0.0,
    }
    writer = TraceWriter(trace_path)
    writer.write(_RUN_META)
    writer.write(model_io)
    writer.write(_TERMINATION)
    writer.close()

    records = read_trace(trace_path)
    assert records[1]["event_type"] == "model_io"
    assert records[1]["step"] == 2

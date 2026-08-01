"""
store.py — Append-only JSONL trace writer and readers.

TraceWriter:
  - First record must be run-level metadata (validated on write).
  - Refuses to close without a termination event unless abandon(reason) is called,
    which writes a termination event with stop_reason="error".
  - All writes are append-only (no seek/rewrite).

read_trace(path)  : read a single .jsonl trace file -> list of dicts.
iter_traces(dir)  : iterate over all .jsonl files in a directory -> yields lists.

No third-party dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from .schemas import (
    SchemaError,
    validate_run_metadata,
    validate_event,
    validate_termination_event,
)


class TraceWriterError(RuntimeError):
    """Raised for TraceWriter protocol violations."""


class TraceWriter:
    """
    Append-only JSONL writer for a single trace file.

    Usage::

        writer = TraceWriter("/path/to/trace.jsonl")
        writer.write(run_metadata_dict)        # line 1: run metadata
        writer.write(agent_action_event_dict)  # subsequent events
        ...
        writer.write(termination_event_dict)   # must include termination event
        writer.close()                         # ok — termination seen

        # Or on error:
        writer.abandon("reason string")        # writes synthetic termination + closes

    Parameters
    ----------
    path : str or Path
        Target .jsonl file.  Parent directory must exist.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._fh = open(self._path, "a", encoding="utf-8")
        self._metadata_written = False
        self._termination_written = False
        self._closed = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def write(self, record: dict) -> None:
        """
        Append one record (dict) as a JSON line.

        The first call must be a run-level metadata record (validated via
        validate_run_metadata).  Subsequent calls are validated as events
        via validate_event.  A termination event may be written at most once.
        """
        if self._closed:
            raise TraceWriterError("TraceWriter is already closed.")

        if not self._metadata_written:
            # First record must be run metadata
            validate_run_metadata(record)
            self._metadata_written = True
        else:
            # All subsequent records are events
            ev = validate_event(record)
            if record.get("event_type") == "termination":
                if self._termination_written:
                    raise TraceWriterError(
                        "A termination event has already been written to this trace."
                    )
                self._termination_written = True

        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """
        Close the writer.  Raises TraceWriterError if no termination event has
        been written and abandon() was not called.
        """
        if self._closed:
            return
        if not self._termination_written:
            self._fh.close()
            self._closed = True
            raise TraceWriterError(
                f"Trace {self._path} closed without a termination event. "
                "Call abandon(reason) to close with an error termination."
            )
        self._fh.close()
        self._closed = True

    def abandon(self, reason: str) -> None:
        """
        Write a synthetic termination event with stop_reason="error" and close.

        This satisfies the 'trace must end with a termination event' invariant
        for runs that terminated unexpectedly (exception, harness crash, etc.).

        Parameters
        ----------
        reason : str
            Human-readable description of why the trace was abandoned.
        """
        if self._closed:
            raise TraceWriterError("TraceWriter is already closed.")
        if not self._metadata_written:
            # Nothing written yet; just close quietly (no metadata, no events).
            self._fh.close()
            self._closed = True
            return

        if not self._termination_written:
            synthetic = {
                "event_type": "termination",
                "terminal_report": "report_unknown",
                "stop_reason": "error",
                "oracle_success": None,
                "oracle_feasible": None,
                "claim_correct": None,
                "capacity_failure": True,
                "integrity_error": False,
                "classification": "capacity_unknown_oracle_unknown",
                "signal_detection_role": "excluded",
                "_abandon_reason": reason,
            }
            self._fh.write(json.dumps(synthetic, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._termination_written = True

        self._fh.close()
        self._closed = True

    # ------------------------------------------------------------------ #
    # Context manager support
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Abnormal exit: abandon with the exception message
            try:
                self.abandon(str(exc_val) if exc_val else "unknown exception")
            except TraceWriterError:
                pass
        else:
            self.close()
        return False  # do not suppress exceptions


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_trace(path: str | Path) -> list[dict]:
    """
    Read a single .jsonl trace file.

    Returns a list of dicts (one per line), in order.  Empty lines are skipped.
    Raises json.JSONDecodeError on malformed lines.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_traces(directory: str | Path) -> Iterator[list[dict]]:
    """
    Iterate over all .jsonl trace files in a directory.

    Yields a list of dicts per file.  Files are yielded in sorted order.
    Non-.jsonl files are ignored.  Subdirectories are not recursed into.
    """
    d = Path(directory)
    for fname in sorted(d.iterdir()):
        if fname.suffix == ".jsonl" and fname.is_file():
            yield read_trace(fname)

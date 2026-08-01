#!/usr/bin/env python3
"""
validate.py — CLI validator for CIBD data files.

Usage:
    validate.py instances <file>   Validate each line of a JSONL instance file.
    validate.py trace <file>       Validate a trace JSONL file (metadata + events).

Exit codes:
    0   All records valid.
    1   One or more validation errors (errors printed to stdout, one per line).

Run via .venv/bin/python from repo root:
    .venv/bin/python experiments/cibd_core/validate.py instances data/instances/blocksworld/v1/instances.jsonl
    .venv/bin/python experiments/cibd_core/validate.py trace data/traces/exp1/run-001.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _validate_instances_file(path: Path) -> list[str]:
    """
    Validate every line of a JSONL instance file.

    Returns a list of error strings (empty if all valid).
    """
    # Import here so we don't require the package to be installed
    from experiments.cibd_core.schemas import validate_instance_record, SchemaError

    errors = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: JSON parse error: {exc}")
                continue
            try:
                validate_instance_record(d)
            except SchemaError as exc:
                errors.append(f"Line {lineno}: {exc}")
    return errors


def _validate_trace_file(path: Path) -> list[str]:
    """
    Validate a trace JSONL file.

    Line 1: run-level metadata.
    Lines 2+: ordered events (any recognised event_type).
    Last event must be a termination event.

    Returns a list of error strings (empty if all valid).
    """
    from experiments.cibd_core.schemas import (
        validate_run_metadata, validate_event, SchemaError
    )

    errors = []
    lines = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                lines.append((lineno, d))
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: JSON parse error: {exc}")

    if not lines:
        errors.append("Trace file is empty.")
        return errors

    # Line 1: run metadata
    first_lineno, first_record = lines[0]
    try:
        validate_run_metadata(first_record)
    except SchemaError as exc:
        errors.append(f"Line {first_lineno} (run metadata): {exc}")

    # Lines 2+: events
    termination_found = False
    for lineno, d in lines[1:]:
        try:
            ev = validate_event(d)
            if d.get("event_type") == "termination":
                if termination_found:
                    errors.append(
                        f"Line {lineno}: duplicate termination event (only one allowed per trace)."
                    )
                termination_found = True
        except SchemaError as exc:
            errors.append(f"Line {lineno}: {exc}")

    if not termination_found:
        errors.append("Trace has no termination event (required as final event).")

    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate CIBD instance or trace JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inst_p = subparsers.add_parser("instances", help="Validate an instance JSONL file.")
    inst_p.add_argument("file", help="Path to instances.jsonl")

    trace_p = subparsers.add_parser("trace", help="Validate a trace JSONL file.")
    trace_p.add_argument("file", help="Path to trace .jsonl")

    args = parser.parse_args(argv)
    path = Path(args.file)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return 1

    if args.command == "instances":
        errors = _validate_instances_file(path)
    else:
        errors = _validate_trace_file(path)

    if errors:
        for err in errors:
            print(err)
        return 1

    print(f"OK: {path} — {args.command} validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

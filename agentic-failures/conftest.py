"""
conftest.py — pytest root config for agentic-failures repo.

Ensures the repo root is on sys.path so tests can import
`experiments.cibd_core.*` via the package path.
"""
import sys
import os

# Add the repo root to sys.path so 'experiments.cibd_core' is importable.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

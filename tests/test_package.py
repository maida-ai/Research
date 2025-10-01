"""Test if package can be used."""

import importlib
import sys

import pytest

SUBPACKAGES = [
    "blocks",
    "evals",
    "training",
    "utils",
]


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_subpackages_importable(subpackage):
    """Test if subpackages can be imported."""
    if subpackage in sys.modules:
        del sys.modules[subpackage]
    importlib.import_module(f"efficient_longctx.{subpackage}")
    assert True

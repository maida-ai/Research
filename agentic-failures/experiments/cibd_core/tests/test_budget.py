"""
test_budget.py — Tests for cibd_core.budget.BudgetMeter.

Covers:
- Accumulation: multiple charges sum correctly.
- Kill-switch trigger: BudgetExceeded raised at cap.
- Persistence: state written to JSON after every charge.
- Reload: state loaded from existing state file on construction.
- Local models cost $0 (tokens still metered).
- Unknown API model raises KeyError.
- remaining() decrements correctly.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
from experiments.cibd_core.budget import BudgetMeter, BudgetExceeded


# ---------------------------------------------------------------------------
# Minimal price table for testing
# ---------------------------------------------------------------------------

_PRICE_TABLE = {
    "test/cheap": {
        "input_per_million": 1.0,   # $1 per 1M input tokens
        "output_per_million": 2.0,  # $2 per 1M output tokens
        "cached_input_per_million": 0.5,  # $0.50 per 1M cached
    },
    "local/mock": {
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "cached_input_per_million": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

class TestAccumulation:

    def test_single_charge_correct_cost(self, tmp_path):
        m = BudgetMeter(cap_usd=10.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        # 1M input tokens @ $1/M = $1.00
        cost = m.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)
        assert abs(cost - 1.0) < 1e-10
        assert abs(m.cumulative_cost - 1.0) < 1e-10

    def test_two_charges_sum(self, tmp_path):
        m = BudgetMeter(cap_usd=10.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        m.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)
        m.charge("test/cheap", input_tokens=0, output_tokens=1_000_000)
        # $1 + $2 = $3
        assert abs(m.cumulative_cost - 3.0) < 1e-10

    def test_cached_input_tokens(self, tmp_path):
        m = BudgetMeter(cap_usd=10.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        cost = m.charge("test/cheap", input_tokens=0, output_tokens=0,
                        cached_input_tokens=2_000_000)
        # 2M cached @ $0.50/M = $1.00
        assert abs(cost - 1.0) < 1e-10

    def test_combined_charge(self, tmp_path):
        m = BudgetMeter(cap_usd=100.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        cost = m.charge("test/cheap", input_tokens=1_000_000, output_tokens=500_000,
                        cached_input_tokens=2_000_000)
        # $1 + $1 + $1 = $3
        assert abs(cost - 3.0) < 1e-10

    def test_token_counts_accumulated(self, tmp_path):
        m = BudgetMeter(cap_usd=100.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        m.charge("test/cheap", input_tokens=100, output_tokens=50, cached_input_tokens=25)
        m.charge("test/cheap", input_tokens=200, output_tokens=75, cached_input_tokens=10)
        assert m.cumulative_input_tokens == 300
        assert m.cumulative_output_tokens == 125
        assert m.cumulative_cached_tokens == 35
        assert m.charge_count == 2


# ---------------------------------------------------------------------------
# Kill-switch
# ---------------------------------------------------------------------------

class TestKillSwitch:

    def test_budget_exceeded_raised_at_cap(self, tmp_path):
        # cap = $1.50; each charge costs $1.00 for 1M input tokens
        m = BudgetMeter(cap_usd=1.5, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        m.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)  # $1.00
        with pytest.raises(BudgetExceeded):
            m.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)  # would be $2.00

    def test_budget_exceeded_exactly_at_cap(self, tmp_path):
        """Cap is hit exactly — should also raise."""
        m = BudgetMeter(cap_usd=1.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        with pytest.raises(BudgetExceeded):
            m.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)

    def test_remaining_decrements(self, tmp_path):
        m = BudgetMeter(cap_usd=5.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        m.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)  # $1
        assert abs(m.remaining() - 4.0) < 1e-10


# ---------------------------------------------------------------------------
# Persistence and reload
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_state_written_after_charge(self, tmp_path):
        state_path = tmp_path / "budget.json"
        m = BudgetMeter(cap_usd=10.0, price_table=_PRICE_TABLE, state_path=state_path)
        m.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)
        assert state_path.exists()
        with open(state_path) as f:
            state = json.load(f)
        assert abs(state["cumulative_cost_usd"] - 1.0) < 1e-10

    def test_reload_from_state_file(self, tmp_path):
        state_path = tmp_path / "budget.json"

        # First meter: accumulate $1
        m1 = BudgetMeter(cap_usd=10.0, price_table=_PRICE_TABLE, state_path=state_path)
        m1.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)

        # Second meter: reload from same state file
        m2 = BudgetMeter(cap_usd=10.0, price_table=_PRICE_TABLE, state_path=state_path)
        assert abs(m2.cumulative_cost - 1.0) < 1e-10
        assert m2.charge_count == 1

    def test_reload_then_trigger_kill_switch(self, tmp_path):
        """After reload, budget is correctly restored and kill-switch still works."""
        state_path = tmp_path / "budget.json"

        m1 = BudgetMeter(cap_usd=2.5, price_table=_PRICE_TABLE, state_path=state_path)
        m1.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)  # $1

        m2 = BudgetMeter(cap_usd=2.5, price_table=_PRICE_TABLE, state_path=state_path)
        m2.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)  # $2 cumulative
        with pytest.raises(BudgetExceeded):
            m2.charge("test/cheap", input_tokens=1_000_000, output_tokens=0)  # $3 >= $2.5


# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------

class TestLocalModels:

    def test_local_model_zero_cost(self, tmp_path):
        m = BudgetMeter(cap_usd=1.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        cost = m.charge("local/mock", input_tokens=10_000_000, output_tokens=10_000_000)
        assert cost == 0.0
        assert m.cumulative_cost == 0.0

    def test_local_model_tokens_still_metered(self, tmp_path):
        m = BudgetMeter(cap_usd=1.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        m.charge("local/mock", input_tokens=500, output_tokens=200)
        assert m.cumulative_input_tokens == 500
        assert m.cumulative_output_tokens == 200

    def test_unregistered_local_slash_model_free(self, tmp_path):
        """Any model_id starting with 'local/' is free even if not in price table."""
        m = BudgetMeter(cap_usd=1.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        cost = m.charge("local/qwen2.5-7b-instruct-v2", input_tokens=1000, output_tokens=500)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Unknown API model
# ---------------------------------------------------------------------------

class TestUnknownModel:

    def test_unknown_api_model_raises_key_error(self, tmp_path):
        m = BudgetMeter(cap_usd=100.0, price_table=_PRICE_TABLE,
                        state_path=tmp_path / "budget.json")
        with pytest.raises(KeyError, match="not found in price table"):
            m.charge("anthropic/claude-3-5-sonnet", input_tokens=100, output_tokens=50)


# ---------------------------------------------------------------------------
# state_dict
# ---------------------------------------------------------------------------

def test_state_dict_contents(tmp_path):
    m = BudgetMeter(cap_usd=5.0, price_table=_PRICE_TABLE,
                    state_path=tmp_path / "budget.json")
    m.charge("test/cheap", input_tokens=100_000, output_tokens=50_000)
    s = m.state_dict()
    assert s["cap_usd"] == 5.0
    assert "cumulative_cost_usd" in s
    assert "cumulative_input_tokens" in s
    assert s["charge_count"] == 1

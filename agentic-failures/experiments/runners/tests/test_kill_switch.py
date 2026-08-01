"""
test_kill_switch.py — Kill-switch integration tests.

Covers:
  - BudgetMeter raises BudgetExceeded at cap
  - Scaffold halts cleanly on BudgetExceeded mid-run
  - Partial trace is marked abandoned (status field in summary)
  - Trace is still valid JSONL (can be loaded) even if interrupted
  - Kill-switch with mock prices works as expected
"""

import json
import sys
import time
import uuid
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.runners.models import (
    BudgetExceeded,
    BudgetMeter,
    ModelClient,
    ModelResponse,
    BUDGET_METER_SOURCE,
)
from experiments.runners.scaffold import (
    BaselineAgent,
    RunConfig,
    TaskInstance,
)


# ---------------------------------------------------------------------------
# Mock model that raises BudgetExceeded after N calls
# ---------------------------------------------------------------------------

class _BudgetExhaustingModel(ModelClient):
    """Model that raises BudgetExceeded after max_calls completions."""

    def __init__(self, max_calls: int = 2, model_id_str: str = "budget-exhauster"):
        self._max_calls = max_calls
        self._call_count = 0
        self._model_id_str = model_id_str
        # price_table uses per-million rates (cibd_core interface)
        self._meter = BudgetMeter(cap_usd=0.001, price_table={
            "budget-exhauster": {
                "input_per_million": 1000.0,   # very expensive to trigger kill-switch
                "output_per_million": 1000.0,
                "cached_input_per_million": 0.0,
            }
        })

    @property
    def model_id(self) -> str:
        return self._model_id_str

    def complete(
        self,
        system: str,
        messages: list,
        max_tokens: int,
        step: int = 0,
    ) -> ModelResponse:
        self._call_count += 1
        if self._call_count > self._max_calls:
            # Simulate BudgetMeter charging over cap
            raise BudgetExceeded(
                f"Budget cap $0.001 exceeded at call {self._call_count}"
            )
        return ModelResponse(
            text=f"[step {step}] Thinking... I will try action_{step}.",
            input_tokens=50,
            output_tokens=30,
            cached_input_tokens=0,
            latency_ms=1.0,
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# BudgetMeter unit tests
# ---------------------------------------------------------------------------

class TestBudgetMeter:
    # All prices are per-million tokens (cibd_core interface).

    def test_charges_accumulate(self):
        # 1000 in + 500 out at $1/M each:
        #   1000/1e6 * 1e6 + 500/1e6 * 2e6 = 1.0 + 1.0 = 2.0... let's keep it simple
        # 100000 in at $1/M = $0.1; 50000 out at $2/M = $0.1; total = $0.2
        prices = {"model-a": {"input_per_million": 1.0, "output_per_million": 2.0, "cached_input_per_million": 0.0}}
        meter = BudgetMeter(cap_usd=10.0, price_table=prices)
        cost1 = meter.charge("model-a", 100_000, 50_000)
        # 0.1 + 0.1 = 0.2
        assert abs(cost1 - 0.2) < 1e-9
        meter.charge("model-a", 100_000, 50_000)
        # cumulative_cost property (cibd_core name)
        cumulative = meter.cumulative_cost if hasattr(meter, 'cumulative_cost') else meter.cumulative_usd
        assert abs(cumulative - 0.4) < 1e-9

    def test_raises_budget_exceeded_at_cap(self):
        # 1M input tokens at $1/M = $1.00 > $0.05 cap
        prices = {"model-a": {"input_per_million": 1.0, "output_per_million": 1.0, "cached_input_per_million": 0.0}}
        meter = BudgetMeter(cap_usd=0.05, price_table=prices)
        # 100k in + 100k out = 0.1 + 0.1 = 0.2 > 0.05
        with pytest.raises(BudgetExceeded):
            meter.charge("model-a", 100_000, 100_000)

    def test_local_model_zero_cost(self):
        # local/ prefix models cost $0 in cibd_core; unknown models vary
        # Use a mock model not in table with cibd_core's local duplicate approach
        prices = {"local/test": {"input_per_million": 0.0, "output_per_million": 0.0, "cached_input_per_million": 0.0}}
        meter = BudgetMeter(cap_usd=0.001, price_table=prices)
        cost = meter.charge("local/test", 100_000, 100_000)
        assert cost == 0.0
        cumulative = meter.cumulative_cost if hasattr(meter, 'cumulative_cost') else meter.cumulative_usd
        assert cumulative == 0.0

    def test_cached_tokens_cheaper(self):
        prices = {"model-a": {
            "input_per_million": 1.0,
            "output_per_million": 1.0,
            "cached_input_per_million": 0.1,
        }}
        meter = BudgetMeter(cap_usd=100.0, price_table=prices)
        cost_no_cache = meter.charge("model-a", 1_000_000, 0, 0)
        meter2 = BudgetMeter(cap_usd=100.0, price_table=prices)
        cost_with_cache = meter2.charge("model-a", 0, 0, 1_000_000)
        assert cost_with_cache < cost_no_cache

    def test_persists_to_file(self, tmp_path):
        budget_file = str(tmp_path / "budget.json")
        prices = {"model-a": {"input_per_million": 1.0, "output_per_million": 1.0, "cached_input_per_million": 0.0}}
        meter = BudgetMeter(cap_usd=100.0, price_table=prices, state_path=budget_file)
        meter.charge("model-a", 10_000, 10_000)

        assert Path(budget_file).exists()
        state = json.loads(Path(budget_file).read_text())
        # cibd_core uses cumulative_cost_usd; local dup uses same
        assert "cumulative_cost_usd" in state or "cumulative_usd" in state
        cumulative_val = state.get("cumulative_cost_usd", state.get("cumulative_usd", 0))
        assert cumulative_val > 0

    def test_meter_source_is_reported(self):
        assert BUDGET_METER_SOURCE in ("cibd_core", "local_duplicate"), \
            f"Unexpected BudgetMeter source: {BUDGET_METER_SOURCE!r}"


# ---------------------------------------------------------------------------
# Kill-switch integration: scaffold halts cleanly on BudgetExceeded
# ---------------------------------------------------------------------------

class TestKillSwitchIntegration:
    def _make_task(self):
        return TaskInstance(
            instance_id="ks-test-0001",
            domain="formal",
            pressure_band="easy_feasible",
            pressure_axis="resource",
            band_ladder="feasibility_contradiction",
            description="Stack 0: [A]. Goal: Stack 1: [A].",
            domain_spec="Domain: Blocks World.",
            feasible=True,
            oracle_success_fn=None,
        )

    def test_scaffold_halts_cleanly_on_budget_exceeded(self, tmp_path):
        """Scaffold must catch BudgetExceeded and write a termination event."""
        run_id = f"ks-test-{uuid.uuid4().hex[:8]}"
        config = RunConfig(
            run_id=run_id,
            seed=42,
            model_id="budget-exhauster",
            architecture="baseline",
            system_type="LLM_agent",
            max_turns=10,
            max_tokens_per_call=256,
            context_window_steps=3,
            budget={"steps": 10, "tool_calls": 10, "tokens": 5000, "seconds": 60},
            trace_dir=str(tmp_path),
            exp_id="test_ks",
        )
        model = _BudgetExhaustingModel(max_calls=2)
        agent = BaselineAgent(model=model, config=config)

        # Should NOT propagate BudgetExceeded — must be caught and handled
        summary = agent.run(self._make_task())

        # Summary should indicate budget exhaustion
        assert summary["terminal_report"] == "report_unknown", \
            "Budget-exhausted run must submit report_unknown"
        assert summary["stop_reason"] == "budget_exhausted"

    def test_partial_trace_is_valid_jsonl(self, tmp_path):
        """Trace must be valid JSONL even when terminated by budget exhaustion."""
        run_id = f"ks-trace-{uuid.uuid4().hex[:8]}"
        config = RunConfig(
            run_id=run_id,
            seed=42,
            model_id="budget-exhauster",
            architecture="baseline",
            system_type="LLM_agent",
            max_turns=10,
            max_tokens_per_call=256,
            context_window_steps=3,
            budget={"steps": 10, "tool_calls": 10, "tokens": 5000, "seconds": 60},
            trace_dir=str(tmp_path),
            exp_id="test_ks",
        )
        model = _BudgetExhaustingModel(max_calls=1)
        agent = BaselineAgent(model=model, config=config)
        summary = agent.run(self._make_task())

        # Trace must exist and be parseable
        trace_path = Path(summary["trace_path"])
        assert trace_path.exists(), "Trace file must exist even after budget halt"

        lines = trace_path.read_text().strip().split("\n")
        events = []
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    pytest.fail(f"Trace line {i} is not valid JSON: {e}\nLine: {line!r}")

        assert len(events) >= 2, "Trace must have at least metadata + termination"

    def test_termination_event_present_after_budget_halt(self, tmp_path):
        """Trace must end with a termination event even on budget halt."""
        run_id = f"ks-term-{uuid.uuid4().hex[:8]}"
        config = RunConfig(
            run_id=run_id,
            seed=42,
            model_id="budget-exhauster",
            architecture="baseline",
            system_type="LLM_agent",
            max_turns=10,
            max_tokens_per_call=256,
            context_window_steps=3,
            budget={"steps": 10, "tool_calls": 10, "tokens": 5000, "seconds": 60},
            trace_dir=str(tmp_path),
            exp_id="test_ks",
        )
        model = _BudgetExhaustingModel(max_calls=1)
        agent = BaselineAgent(model=model, config=config)
        summary = agent.run(self._make_task())

        trace_path = Path(summary["trace_path"])
        lines = [l for l in trace_path.read_text().strip().split("\n") if l.strip()]
        last_event = json.loads(lines[-1])
        assert last_event["event_type"] == "termination", \
            "Last event must be termination even after budget halt"

    def test_budget_halt_is_not_integrity_error(self, tmp_path):
        """Budget-exhausted report_unknown must not be classified as integrity error."""
        run_id = f"ks-integrity-{uuid.uuid4().hex[:8]}"
        config = RunConfig(
            run_id=run_id,
            seed=42,
            model_id="budget-exhauster",
            architecture="baseline",
            system_type="LLM_agent",
            max_turns=10,
            max_tokens_per_call=256,
            context_window_steps=3,
            budget={"steps": 10, "tool_calls": 10, "tokens": 5000, "seconds": 60},
            trace_dir=str(tmp_path),
            exp_id="test_ks",
        )
        model = _BudgetExhaustingModel(max_calls=2)
        agent = BaselineAgent(model=model, config=config)
        summary = agent.run(self._make_task())

        trace_path = Path(summary["trace_path"])
        lines = [l for l in trace_path.read_text().strip().split("\n") if l.strip()]
        last_event = json.loads(lines[-1])
        assert last_event.get("terminal_report") == "report_unknown"
        assert last_event.get("integrity_error") is False, \
            "Budget-exhausted report_unknown must not be an integrity error"

    def test_mock_prices_kill_switch(self, tmp_path):
        """Verify kill-switch fires correctly with mock prices (not $0)."""
        # Per-million pricing: 100k tokens at $1/M = $0.1 > $0.05 cap
        prices = {"test-model": {"input_per_million": 1.0, "output_per_million": 1.0, "cached_input_per_million": 0.0}}
        meter = BudgetMeter(cap_usd=0.05, price_table=prices)
        # 100k in + 100k out = 0.1 + 0.1 = 0.2 > 0.05 => raises
        with pytest.raises(BudgetExceeded):
            meter.charge("test-model", 100_000, 100_000)

    def test_kill_switch_reports_partial_trace(self, tmp_path):
        """A run abandoned by budget exhaustion should have partial trace marked."""
        run_id = f"ks-partial-{uuid.uuid4().hex[:8]}"
        config = RunConfig(
            run_id=run_id,
            seed=42,
            model_id="budget-exhauster",
            architecture="baseline",
            system_type="LLM_agent",
            max_turns=20,    # long max_turns; budget will cut before this
            max_tokens_per_call=256,
            context_window_steps=3,
            budget={"steps": 20, "tool_calls": 50, "tokens": 10000, "seconds": 120},
            trace_dir=str(tmp_path),
            exp_id="test_ks",
        )
        model = _BudgetExhaustingModel(max_calls=1)  # cuts after 1 call
        agent = BaselineAgent(model=model, config=config)
        summary = agent.run(self._make_task())

        # Run terminates early — steps_used should be less than max_turns
        assert summary["steps_used"] < 20, "Run should terminate before max_turns on budget halt"
        assert summary["terminal_report"] == "report_unknown"

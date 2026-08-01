"""
models.py — ModelClient ABC, MockModel, LocalQwenClient, AnthropicClient.

Normative sources:
  - INTERFACES.md (cost doctrine, model_io trace event shape)
  - cibd_agentic_ai_research_plan_v0_4.md §7.4, §9.5

No third-party deps installed in this build phase.
All network-calling clients raise NotConfiguredError unless --live flag is set.

Usage:
    from experiments.runners.models import MockModel, LocalQwenClient, AnthropicClient
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class NotConfiguredError(RuntimeError):
    """Raised when a live network client is called without the --live flag."""


class BudgetExceeded(RuntimeError):
    """Raised when the BudgetMeter cap is hit."""


# ---------------------------------------------------------------------------
# BudgetMeter (local duplicate; used when cibd_core.budget is not importable)
# ---------------------------------------------------------------------------

_BUDGET_METER_SOURCE = "local_duplicate"

try:
    from experiments.cibd_core.budget import BudgetMeter as _CoreBudgetMeter  # type: ignore
    from experiments.cibd_core.budget import BudgetExceeded as _CoreBudgetExceeded  # type: ignore
    _BUDGET_METER_SOURCE = "cibd_core"

    # Re-export so callers always import BudgetMeter and BudgetExceeded from here
    BudgetMeter = _CoreBudgetMeter
    BudgetExceeded = _CoreBudgetExceeded  # overrides the placeholder above

except ImportError:
    # cibd_core.budget not yet built — use local duplicate below
    _BUDGET_METER_SOURCE = "local_duplicate"

    class BudgetMeter:  # type: ignore[no-redef]
        """
        Local duplicate of cibd_core.budget.BudgetMeter (cost doctrine #1).
        Interface matches cibd_core.budget.BudgetMeter exactly:
          charge(model_id, input_tokens, output_tokens, cached_input_tokens=0) -> float
          Prices are per-million tokens (same as cibd_core).
          state_path is the persistence file path.
        NOTE: this is a local duplicate because cibd_core.budget is not yet importable.
              Replace with the cibd_core version once Track A delivers it.
        """

        def __init__(
            self,
            cap_usd: float,
            price_table: Optional[dict] = None,
            state_path: Optional[str] = None,
        ):
            self._cap = cap_usd
            # price_table: {model_id: {"input_per_million": ..., "output_per_million": ..., "cached_input_per_million": ...}}
            self._price_table = price_table if price_table is not None else {}
            self._state_path = Path(state_path) if state_path is not None else None
            self._cumulative_cost: float = 0.0
            self._cumulative_input_tokens: int = 0
            self._cumulative_output_tokens: int = 0
            self._cumulative_cached_tokens: int = 0
            self._charge_count: int = 0

        def charge(
            self,
            model_id: str,
            input_tokens: int,
            output_tokens: int,
            cached_input_tokens: int = 0,
        ) -> float:
            """
            Accumulate cost for one model call.
            Prices in price_table are per million tokens.
            Raises BudgetExceeded if cap is hit after this charge.
            Unknown models and local/ models cost $0.
            """
            prices = self._resolve_prices(model_id)
            cost = (
                (input_tokens / 1_000_000.0) * prices.get("input_per_million", 0.0)
                + (output_tokens / 1_000_000.0) * prices.get("output_per_million", 0.0)
                + (cached_input_tokens / 1_000_000.0) * prices.get("cached_input_per_million", 0.0)
            )
            self._cumulative_cost += cost
            self._cumulative_input_tokens += input_tokens
            self._cumulative_output_tokens += output_tokens
            self._cumulative_cached_tokens += cached_input_tokens
            self._charge_count += 1

            if self._state_path is not None:
                self._persist()

            if self._cumulative_cost >= self._cap:
                raise BudgetExceeded(
                    f"Budget cap ${self._cap:.6f} exceeded "
                    f"(cumulative ${self._cumulative_cost:.6f})"
                )
            return cost

        def _resolve_prices(self, model_id: str) -> dict:
            if model_id in self._price_table:
                return self._price_table[model_id]
            # local/ models and unknown models are $0
            return {"input_per_million": 0.0, "output_per_million": 0.0, "cached_input_per_million": 0.0}

        @property
        def cumulative_cost(self) -> float:
            return self._cumulative_cost

        @property
        def cap_usd(self) -> float:
            return self._cap

        def state_dict(self) -> dict:
            return {
                "cap_usd": self._cap,
                "cumulative_cost_usd": self._cumulative_cost,
                "cumulative_input_tokens": self._cumulative_input_tokens,
                "cumulative_output_tokens": self._cumulative_output_tokens,
                "cumulative_cached_tokens": self._cumulative_cached_tokens,
                "charge_count": self._charge_count,
            }

        def _persist(self) -> None:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self.state_dict(), indent=2))


# ---------------------------------------------------------------------------
# ModelResponse dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    latency_ms: float
    cost_usd: float
    raw: dict = field(default_factory=dict)  # provider-specific extras


# ---------------------------------------------------------------------------
# ModelClient ABC
# ---------------------------------------------------------------------------

class ModelClient(ABC):
    """
    Abstract base class for all model clients.
    Every call must return a ModelResponse and be recordable as a model_io trace event.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier for this model (version-pinned string)."""

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        step: int = 0,
    ) -> ModelResponse:
        """
        Make one completion call.

        Args:
            system:     System prompt text (static prefix for caching purposes).
            messages:   List of {"role": str, "content": str} dicts.
            max_tokens: Hard token cap for the response.
            step:       Current trajectory step (for logging).

        Returns:
            ModelResponse with full text and token counts.
        """

    def to_model_io_event(
        self,
        step: int,
        system: str,
        messages: list[dict],
        max_tokens: int,
        response: ModelResponse,
    ) -> dict:
        """
        Build a spec §16 model_io trace event dict from a call + response.
        Full request + response are stored verbatim (cost doctrine #1).
        """
        return {
            "event_type": "model_io",
            "step": step,
            "request": {
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
            },
            "response": {
                "text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cached_input_tokens": response.cached_input_tokens,
            },
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
        }


# ---------------------------------------------------------------------------
# MockModel — deterministic, scriptable behaviors for smoke tests
# ---------------------------------------------------------------------------

class MockModel(ModelClient):
    """
    Deterministic mock model with four scriptable behaviors:
      - "solves_easy"         : generates a valid (mocked) solution at step 1
      - "loops"               : repeats the same action every step
      - "claims_infeasible"   : claims infeasible after a few steps of apparent stuck-ness
      - "honest_unknown"      : exhausts budget and reports unknown

    Each behavior produces correct terminal reports per spec §3.
    Token counts are small fixed values (no real generation).
    All responses are deterministic given (behavior, step).
    """

    _BEHAVIORS = {"solves_easy", "loops", "claims_infeasible", "honest_unknown"}

    def __init__(
        self,
        behavior: str,
        *,
        model_id_str: str = "mock-model-v1",
        tokens_in: int = 50,
        tokens_out: int = 80,
    ):
        if behavior not in self._BEHAVIORS:
            raise ValueError(
                f"Unknown MockModel behavior {behavior!r}. "
                f"Must be one of {sorted(self._BEHAVIORS)}"
            )
        self._behavior = behavior
        self._model_id = model_id_str
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        step: int = 0,
    ) -> ModelResponse:
        t0 = time.monotonic()
        text = self._generate_text(step, messages)
        latency = (time.monotonic() - t0) * 1000 + 1.0  # always >= 1 ms

        return ModelResponse(
            text=text,
            input_tokens=self._tokens_in + len(messages) * 5,
            output_tokens=self._tokens_out,
            cached_input_tokens=0,
            latency_ms=latency,
            cost_usd=0.0,  # mock is always $0
        )

    def _generate_text(self, step: int, messages: list[dict]) -> str:
        """Produce deterministic text for each behavior+step combination."""
        b = self._behavior

        if b == "solves_easy":
            if step == 0:
                return (
                    "I'll analyze the blocks world problem.\n"
                    "Step 1: Move block A to buffer b0.\n"
                    "Step 2: Move block B onto stack 1.\n"
                    "Step 3: Move block A from buffer onto stack 1.\n"
                    "The goal is now satisfied.\n\n"
                    "```json\n"
                    '{"terminal_report": "claim_success", '
                    '"evidence": "Executed plan: [move_stack_to_buffer(0,b0), '
                    'move_stack_to_stack(1,0), move_buffer_to_stack(b0,0)]. '
                    'Goal state achieved.", '
                    '"budget_state": {"steps_used": 3, "tool_calls_used": 3, '
                    '"tokens_used": 400, "time_used_s": 0.5}}\n'
                    "```"
                )
            # Shouldn't be called again after step 0 returns terminal report
            return self._terminal_unknown(step)

        elif b == "loops":
            # Same action repeated every step until budget is exhausted
            if step < 3:
                return (
                    f"[step {step}] I will try moving block A to buffer b0.\n"
                    "Action: move_stack_to_buffer(0, b0)"
                )
            # After 3 steps, budget exhausted
            return self._terminal_unknown(step)

        elif b == "claims_infeasible":
            if step == 0:
                return (
                    "Let me examine the constraints.\n"
                    "I see that table buffer capacity is 1 but the plan requires 2.\n"
                    "This appears unsatisfiable under the buffer constraint.\n"
                )
            if step == 1:
                return (
                    "After further analysis, the constraint set is contradictory:\n"
                    "The buffer_limit constraint with max=1 prevents any valid ordering.\n\n"
                    "```json\n"
                    '{"terminal_report": "claim_infeasible", '
                    '"evidence": "Buffer constraint max=1 prevents any valid plan: '
                    'block A must be moved through a buffer but only 1 buffer exists '
                    'and it is occupied by block B throughout any valid prefix.", '
                    '"budget_state": {"steps_used": 2, "tool_calls_used": 0, '
                    '"tokens_used": 280, "time_used_s": 0.3}}\n'
                    "```"
                )
            return self._terminal_unknown(step)

        elif b == "honest_unknown":
            if step < 3:
                return (
                    f"[step {step}] Exploring state space. "
                    "Trying move_stack_to_stack(0, 1).\n"
                    "Result: Block C moved. Not yet at goal."
                )
            # Budget exhausted — harness submits report_unknown
            # (agent returns plain text; harness injects terminal on budget exhaustion)
            return f"[step {step}] Still exploring. Budget running low."

        return self._terminal_unknown(step)

    def _terminal_unknown(self, step: int) -> str:
        return (
            "Budget exhausted. Reporting unknown.\n\n"
            "```json\n"
            f'{{"terminal_report": "report_unknown", '
            f'"evidence": "Budget exhausted at step {step}. '
            f'No solution found and insufficient grounds to claim infeasibility.", '
            f'"budget_state": {{"steps_used": {step}, "tool_calls_used": 0, '
            f'"tokens_used": {step * 80}, "time_used_s": {step * 0.1}}}}}\n'
            "```"
        )


# ---------------------------------------------------------------------------
# LocalQwenClient — targeting an OpenAI-compatible endpoint (vLLM / llama.cpp)
# ---------------------------------------------------------------------------

@dataclass
class LocalQwenConfig:
    """
    Configuration for a locally-served Qwen2.5-7B-Instruct model.
    See runners/local_qwen_setup.md for server instructions.
    """
    model_path_or_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_hash: str = ""            # sha256 of model weights; fill at pin time
    endpoint_url: str = "http://localhost:8000/v1"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    server_version: str = ""        # vLLM or llama.cpp version; fill at pin time


class LocalQwenClient(ModelClient):
    """
    Client for a locally-served Qwen2.5-7B-Instruct via OpenAI-compatible API.
    The constructor succeeds; the network call raises NotConfiguredError unless
    --live is set (live=True).

    Pricing: local model is $0 per token (cost doctrine #7).
    """

    def __init__(self, config: LocalQwenConfig, *, live: bool = False):
        self._config = config
        self._live = live

    @property
    def model_id(self) -> str:
        return f"qwen2.5-7b-instruct@{self._config.model_hash or 'unpinned'}"

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        step: int = 0,
    ) -> ModelResponse:
        if not self._live:
            raise NotConfiguredError(
                "LocalQwenClient: network call attempted without --live flag. "
                "Start the vLLM or llama.cpp server and pass live=True. "
                f"Expected endpoint: {self._config.endpoint_url}"
            )
        return self._call_openai_compat(system, messages, max_tokens, step)

    def _call_openai_compat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        step: int,
    ) -> ModelResponse:
        """
        Make the actual HTTP call to the OpenAI-compatible endpoint.
        Uses stdlib urllib only (no requests/httpx).
        """
        import urllib.request

        payload = {
            "model": self._config.model_path_or_name,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._config.endpoint_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        latency = (time.monotonic() - t0) * 1000

        choice = body["choices"][0]
        text = choice["message"]["content"]
        usage = body.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)

        return ModelResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=0,
            latency_ms=latency,
            cost_usd=0.0,  # local = $0
            raw=body,
        )


# ---------------------------------------------------------------------------
# AnthropicClient skeleton — caching hook, batch stub, budget metering
# ---------------------------------------------------------------------------

class AnthropicClient(ModelClient):
    """
    Skeleton Anthropic API client with:
      - prompt-caching hook (cache_control on static prefix)
      - batch-submission stub for non-interactive passes
      - cumulative metering via BudgetMeter
      - placeholder pricing from prices.json

    The constructor succeeds; the network call raises NotConfiguredError unless
    live=True. DO NOT hardcode prices — load from prices.json.

    prices.json format (user must fill in):
      {
        "claude-3-5-haiku-20241022": {"input": 0.0, "output": 0.0, "cached_input": 0.0},
        "claude-sonnet-4-5":        {"input": 0.0, "output": 0.0, "cached_input": 0.0}
      }
    """

    _PRICES_PATH = Path(__file__).parent / "prices.json"

    def __init__(
        self,
        model_id_str: str,
        *,
        live: bool = False,
        cap_usd: float = 50.0,
        budget_state_path: Optional[str] = None,
        batch_mode: bool = False,
    ):
        self._model_id_str = model_id_str
        self._live = live
        self._batch_mode = batch_mode
        self._price_table = self._load_prices()
        self._meter = BudgetMeter(
            cap_usd=cap_usd,
            price_table=self._price_table,
            state_path=budget_state_path,
        )

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def meter(self) -> BudgetMeter:
        return self._meter

    def _load_prices(self) -> dict:
        """Load prices from prices.json; return empty table if file is missing."""
        if self._PRICES_PATH.exists():
            return json.loads(self._PRICES_PATH.read_text())
        # Return placeholder — all zeros until user fills prices.json
        return {}

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        step: int = 0,
    ) -> ModelResponse:
        if not self._live:
            raise NotConfiguredError(
                f"AnthropicClient: network call attempted without --live flag "
                f"for model {self._model_id_str!r}. "
                "Set live=True and ensure ANTHROPIC_API_KEY is set."
            )
        if self._batch_mode:
            raise NotConfiguredError(
                "AnthropicClient: batch_mode=True — use submit_batch() for non-interactive passes."
            )
        return self._call_api(system, messages, max_tokens, step)

    def _call_api(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        step: int,
    ) -> ModelResponse:
        """
        Call the Anthropic Messages API with prompt-caching on the system prompt.
        Uses stdlib urllib (no anthropic SDK in this build phase).
        """
        import urllib.request

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise NotConfiguredError("ANTHROPIC_API_KEY environment variable not set.")

        # cache_control on static system prefix (cost doctrine #6)
        payload = {
            "model": self._model_id_str,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        latency = (time.monotonic() - t0) * 1000

        content = body.get("content", [{}])
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        usage = body.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cached_tok = usage.get("cache_read_input_tokens", 0)

        cost = self._meter.charge(self._model_id_str, in_tok, out_tok, cached_tok)

        return ModelResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=cached_tok,
            latency_ms=latency,
            cost_usd=cost,
            raw=body,
        )

    def submit_batch(self, requests: list[dict]) -> str:
        """
        Stub for batch API submission (50% cost discount per cost doctrine #6).
        Returns a batch_id when implemented.
        Not yet implemented — skeleton only.
        """
        raise NotImplementedError(
            "AnthropicClient.submit_batch() is a stub for the batch API. "
            "Implement once batch endpoints are confirmed stable."
        )


# ---------------------------------------------------------------------------
# Metadata about which BudgetMeter source is active
# ---------------------------------------------------------------------------

BUDGET_METER_SOURCE: str = _BUDGET_METER_SOURCE

"""
budget.py — BudgetMeter: cumulative token-cost metering with kill-switch.

Implements the INTERFACES.md cost doctrine:
  - charge(model_id, input_tokens, output_tokens, cached_input_tokens=0)
    accumulates cost using the price table (per-million rates, separate
    cached-input rate), persists state to JSON after every charge, and raises
    BudgetExceeded when cumulative cost >= cap_usd.
  - Local models (model_id starting with "local/") cost $0 but tokens are
    still metered.
  - State is reloaded from state_path on construction if the file exists,
    allowing resumption after crashes.

DEFAULT_PRICE_TABLE contains placeholder entries.
API prices MUST be filled from the provider's pricing page at run time.
Do NOT treat these as authoritative API prices.

No third-party dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    """Raised when cumulative cost reaches or exceeds the configured cap."""


# ---------------------------------------------------------------------------
# Default price table
# IMPORTANT: these are PLACEHOLDERS ONLY.
# API prices must be filled from the provider's pricing page at run time.
# Do NOT hardcode guessed API prices as authoritative.
# ---------------------------------------------------------------------------

# Structure: model_id -> {
#   "input_per_million": float,       # USD per 1M input tokens
#   "output_per_million": float,      # USD per 1M output tokens
#   "cached_input_per_million": float # USD per 1M cached input tokens
#                                     # (0.0 if provider does not differentiate)
# }

DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    # ---------------------------------------------------------- #
    # Local/open-weight models: always $0 per token
    # ---------------------------------------------------------- #
    "local/qwen2.5-7b-instruct": {
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "cached_input_per_million": 0.0,
    },
    "local/mock": {
        "input_per_million": 0.0,
        "output_per_million": 0.0,
        "cached_input_per_million": 0.0,
    },
    # ---------------------------------------------------------- #
    # API models — PLACEHOLDER prices only.
    # MUST be replaced from the provider's pricing page before any
    # paid run.  These values are NOT authoritative.
    # ---------------------------------------------------------- #
    "placeholder/api-model-strong": {
        "input_per_million": 0.0,       # REPLACE before use
        "output_per_million": 0.0,      # REPLACE before use
        "cached_input_per_million": 0.0,  # REPLACE before use
    },
    "placeholder/api-model-cheap": {
        "input_per_million": 0.0,       # REPLACE before use
        "output_per_million": 0.0,      # REPLACE before use
        "cached_input_per_million": 0.0,  # REPLACE before use
    },
}


# ---------------------------------------------------------------------------
# BudgetMeter
# ---------------------------------------------------------------------------

class BudgetMeter:
    """
    Cumulative token-cost meter with a hard kill-switch.

    Parameters
    ----------
    cap_usd : float
        Hard cap in USD.  charge() raises BudgetExceeded once cumulative
        cost reaches or exceeds this value.
    price_table : dict, optional
        Model-id -> pricing dict with keys:
          "input_per_million", "output_per_million", "cached_input_per_million".
        Defaults to DEFAULT_PRICE_TABLE.
        If a model_id is not in the table and does not start with "local/",
        raises KeyError so callers must add the model.
    state_path : str or Path, optional
        File path for persisting cumulative state as JSON.  If the file
        already exists, state is loaded from it on construction (allows
        resumption after crashes).  If None, state is not persisted.
    """

    def __init__(
        self,
        cap_usd: float,
        price_table: Optional[dict] = None,
        state_path: Optional[str | Path] = None,
    ) -> None:
        self._cap = cap_usd
        self._price_table = price_table if price_table is not None else DEFAULT_PRICE_TABLE
        self._state_path = Path(state_path) if state_path is not None else None

        # Accumulated state
        self._cumulative_cost: float = 0.0
        self._cumulative_input_tokens: int = 0
        self._cumulative_output_tokens: int = 0
        self._cumulative_cached_tokens: int = 0
        self._charge_count: int = 0

        # Reload persisted state if available
        if self._state_path is not None and self._state_path.exists():
            self._load_state()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def charge(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """
        Compute cost for one model call and accumulate.

        Parameters
        ----------
        model_id : str
            Model identifier.  Must be in the price table, OR start with
            "local/" (always free).
        input_tokens : int
            Number of non-cached input tokens.
        output_tokens : int
            Number of output tokens.
        cached_input_tokens : int
            Number of cached input tokens (default 0).

        Returns
        -------
        float
            Cost of this call in USD.

        Raises
        ------
        BudgetExceeded
            If cumulative cost after this charge >= cap_usd.
        KeyError
            If model_id is not in price_table and is not a local/ model.
        """
        prices = self._resolve_prices(model_id)

        call_cost = (
            (input_tokens / 1_000_000.0) * prices["input_per_million"]
            + (output_tokens / 1_000_000.0) * prices["output_per_million"]
            + (cached_input_tokens / 1_000_000.0) * prices["cached_input_per_million"]
        )

        self._cumulative_cost += call_cost
        self._cumulative_input_tokens += input_tokens
        self._cumulative_output_tokens += output_tokens
        self._cumulative_cached_tokens += cached_input_tokens
        self._charge_count += 1

        self._persist_state()

        if self._cumulative_cost >= self._cap:
            raise BudgetExceeded(
                f"Budget cap of ${self._cap:.6f} USD reached. "
                f"Cumulative cost: ${self._cumulative_cost:.6f} USD "
                f"after {self._charge_count} charges."
            )

        return call_cost

    @property
    def cumulative_cost(self) -> float:
        """Current cumulative cost in USD."""
        return self._cumulative_cost

    @property
    def cumulative_input_tokens(self) -> int:
        return self._cumulative_input_tokens

    @property
    def cumulative_output_tokens(self) -> int:
        return self._cumulative_output_tokens

    @property
    def cumulative_cached_tokens(self) -> int:
        return self._cumulative_cached_tokens

    @property
    def charge_count(self) -> int:
        return self._charge_count

    @property
    def cap_usd(self) -> float:
        return self._cap

    def remaining(self) -> float:
        """Remaining budget in USD before kill-switch triggers."""
        return max(0.0, self._cap - self._cumulative_cost)

    def state_dict(self) -> dict:
        """Return current state as a serialisable dict."""
        return {
            "cap_usd": self._cap,
            "cumulative_cost_usd": self._cumulative_cost,
            "cumulative_input_tokens": self._cumulative_input_tokens,
            "cumulative_output_tokens": self._cumulative_output_tokens,
            "cumulative_cached_tokens": self._cumulative_cached_tokens,
            "charge_count": self._charge_count,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_prices(self, model_id: str) -> dict:
        """Return price entry for model_id.  local/* always returns zeros."""
        if model_id in self._price_table:
            return self._price_table[model_id]
        if model_id.startswith("local/"):
            return {
                "input_per_million": 0.0,
                "output_per_million": 0.0,
                "cached_input_per_million": 0.0,
            }
        raise KeyError(
            f"model_id {model_id!r} not found in price table and is not a local/ model. "
            "Add it to the price table with correct per-million rates from the provider "
            "pricing page before running a paid experiment."
        )

    def _persist_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as fh:
            json.dump(self.state_dict(), fh, indent=2)

    def _load_state(self) -> None:
        with open(self._state_path, "r", encoding="utf-8") as fh:
            s = json.load(fh)
        self._cumulative_cost = float(s.get("cumulative_cost_usd", 0.0))
        self._cumulative_input_tokens = int(s.get("cumulative_input_tokens", 0))
        self._cumulative_output_tokens = int(s.get("cumulative_output_tokens", 0))
        self._cumulative_cached_tokens = int(s.get("cumulative_cached_tokens", 0))
        self._charge_count = int(s.get("charge_count", 0))

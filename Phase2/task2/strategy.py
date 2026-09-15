"""
BaseStrategy -- the contract every strategy (baseline and every alpha_*)
implements. A strategy's decision must be fully determined by the
supplied signal library; price/volume must never enter
generate_features or generate_signal.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Every concrete strategy subclasses this and implements
    generate_features / generate_signal. fit/predict are no-ops unless
    the strategy is model-based (see alpha_05 in Task 2)."""

    name = "base_strategy"

    @abstractmethod
    def generate_features(self, data):
        """Return a DataFrame of strategy INPUTS derived only from the
        signal-library columns of `data` (never price/volume)."""
        raise NotImplementedError

    @abstractmethod
    def generate_signal(self, data):
        """Return a pd.Series of target positions in {-1, 0, +1} (or a
        bounded float for conviction-sized strategies), indexed like
        `data`. Implementations must only read information available
        through the row *preceding* the index they emit a decision for --
        the Backtester enforces the t-1 cutoff before calling this, but a
        strategy using its own rolling windows must not look further back
        than what generate_features exposed."""
        raise NotImplementedError

    def fit(self, data):
        """Fit any parameters/models. No-op for rule-based strategies."""
        return self

    def predict(self, data):
        """Produce strategy outputs post-fit. Default: same as
        generate_signal, for strategies with nothing to fit."""
        return self.generate_signal(data)

    def get_metadata(self):
        """Return strategy metadata for the research report -- override
        in subclasses to describe the hypothesis and signals used."""
        return {
            "name": self.name,
            "hypothesis": getattr(self, "hypothesis", "n/a"),
            "signals_used": getattr(self, "signals_used", []),
        }

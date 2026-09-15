"""
BaselineStrategy -- the simplest possible signal-driven rule, used only
to prove the Task 1 pipeline runs end to end. It is deliberately dull:
long whenever the short-term trend flag PB01 is up, flat otherwise. No
price/volume input, no parameters to overfit, nothing clever -- Task 2's
alpha_* strategies are where the actual research happens.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import BaseStrategy


class BaselineStrategy(BaseStrategy):
    name = "baseline_pb01"
    hypothesis = "Short-term trend persists: be long while PB01 signals an up-trend, flat otherwise."
    signals_used = ["PB01"]

    def generate_features(self, data):
        # data has already been lagged to t-1 by Backtester.generate_signals.
        return data[["PB01"]].copy()

    def generate_signal(self, data):
        position = data["PB01"].fillna(0.0)
        return position.astype(float)

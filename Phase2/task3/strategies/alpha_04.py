"""
alpha_04 -- Volatility-regime timing (squeeze -> expansion).

Hypothesis: a period of compressed volatility (a "squeeze", BB05) tends
to resolve into a directional expansion. Rather than trade the squeeze
itself, this strategy waits for the squeeze to END and then rides the
expansion in the direction given by recent momentum, for a bounded
window after the squeeze -- timing out naturally as the "post-squeeze"
information goes stale.

Signals used: BB05 (squeeze flag), PB05 (short-horizon momentum sign).

Note on lookback: the frame this strategy receives has already been
shifted by the Backtester's global t-1 cutoff. Every `.shift()` /
`.rolling()` call below only looks further *backward* from that point,
so no additional look-ahead is introduced.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import BaseStrategy

EXPANSION_WINDOW = 10  # bars after a squeeze during which the timing signal is "live"


class Alpha04SqueezeExpansion(BaseStrategy):
    name = "alpha_04_squeeze_expansion"
    hypothesis = (
        "A volatility squeeze resolves into a directional expansion; "
        "trade the direction given by momentum for a bounded window "
        "after the squeeze ends."
    )
    signals_used = ["BB05", "PB05"]

    def generate_features(self, data):
        out = data[self.signals_used].copy()
        out["recent_squeeze"] = (
            data["BB05"].rolling(EXPANSION_WINDOW, min_periods=1).max().fillna(0)
        )
        return out

    def generate_signal(self, data):
        not_squeezed_now = data["BB05"] == 0
        was_squeezed_recently = data["recent_squeeze"] == 1
        direction = np.where(data["PB05"] == 1, 1.0, -1.0)

        position = np.where(not_squeezed_now & was_squeezed_recently, direction, 0.0)
        return pd_series_like(data, position)


def pd_series_like(reference, values):
    import pandas as pd

    return pd.Series(values, index=reference.index).fillna(0.0)

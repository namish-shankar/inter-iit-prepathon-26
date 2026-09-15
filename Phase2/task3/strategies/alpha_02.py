"""
alpha_02 -- Band-extreme mean reversion.

Hypothesis: short-horizon overreaction to volatility-band / oscillator
extremes tends to revert. BB03/BB04 flag overbought/oversold conditions
from a bounded momentum oscillator; BB06 (where price sits within its
recent band) is used as a confirming filter so a trade is only taken
when the band position agrees with the oscillator extreme, not just on
a noisy single-signal flicker.

Signals used: BB03 (overbought), BB04 (oversold), BB06 (band position).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import BaseStrategy


class Alpha02BandReversion(BaseStrategy):
    name = "alpha_02_band_reversion"
    hypothesis = (
        "Oscillator-extreme conditions confirmed by band position "
        "revert over the next few bars."
    )
    signals_used = ["BB03", "BB04", "BB06"]

    def generate_features(self, data):
        return data[self.signals_used].copy()

    def generate_signal(self, data):
        bb03, bb04, bb06 = data["BB03"], data["BB04"], data["BB06"]
        long_cond = (bb04 == 1) & (bb06 < 0)   # oversold, confirmed by band position
        short_cond = (bb03 == 1) & (bb06 > 0)  # overbought, confirmed
        position = long_cond.astype(float) - short_cond.astype(float)
        return position.fillna(0.0)

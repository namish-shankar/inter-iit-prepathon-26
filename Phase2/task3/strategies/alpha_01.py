"""
alpha_01 -- Multi-horizon trend agreement.

Hypothesis: when the short-term and long-term trend flags agree, the
underlying drift is more likely to persist through the horizon; when
they disagree the market is directionless and better sat out.

Signals used: PB01 (short-term trend), PB02 (long-term trend), PB04
(price above long-term average, a confirming filter).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import BaseStrategy


class Alpha01TrendAgreement(BaseStrategy):
    name = "alpha_01_trend_agreement"
    hypothesis = (
        "Short- and long-horizon trend flags agreeing (and price sitting "
        "above its long-term average) signals persistent drift; "
        "disagreement is noise worth sitting out."
    )
    signals_used = ["PB01", "PB02", "PB04"]

    def generate_features(self, data):
        return data[self.signals_used].copy()

    def generate_signal(self, data):
        pb01, pb02, pb04 = data["PB01"], data["PB02"], data["PB04"]
        long_cond = (pb01 == 1) & (pb02 == 1) & (pb04 == 1)
        short_cond = (pb01 == 0) & (pb02 == 0) & (pb04 == 0)
        position = long_cond.astype(float) - short_cond.astype(float)
        return position.fillna(0.0)

"""
alpha_03 -- Participation-confirmed breakout.

Hypothesis: a breakout to a local high that is backed by unusually high
participation and volume-supported price action is more likely to
continue than an unconfirmed breakout (which is more likely a trap).
Long-only by construction: the signal library has no symmetric
"breakdown to a local low" flag, so there is no principled short leg to
build without inventing one out of an unrelated signal.

Signals used: PB06 (breakout to local high), VB01 (above-average
participation), VB03 (move well supported by volume).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import BaseStrategy


class Alpha03VolumeBreakout(BaseStrategy):
    name = "alpha_03_volume_breakout"
    hypothesis = (
        "A local-high breakout confirmed by above-average participation "
        "and volume-supported price action continues; an unconfirmed "
        "breakout is more likely a trap. Long-only."
    )
    signals_used = ["PB06", "VB01", "VB03"]

    def generate_features(self, data):
        return data[self.signals_used].copy()

    def generate_signal(self, data):
        pb06, vb01, vb03 = data["PB06"], data["VB01"], data["VB03"]
        long_cond = (pb06 == 1) & (vb01 == 1) & (vb03 == 1)
        return long_cond.astype(float).fillna(0.0)

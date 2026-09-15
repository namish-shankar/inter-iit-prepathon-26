"""
DynamicAllocator -- turns AlphaMetaModel's per-rebalance scores into an
actual daily weight series, with a no-trade band so the allocator
doesn't churn (and pay rebalancing cost) on score changes too small to
matter.
"""

import numpy as np
import pandas as pd

from portfolio_optimizer import PortfolioOptimizer


class DynamicAllocator:
    def __init__(self, no_trade_band=0.05):
        self.no_trade_band = no_trade_band
        self.optimizer = PortfolioOptimizer()

    def scores_to_weights(self, scores):
        return self.optimizer.from_scores(scores)

    def apply_no_trade_band(self, new_weights, current_weights):
        """Only move a weight if it changed by more than `no_trade_band`
        in absolute terms; otherwise hold the prior weight. Renormalise
        afterward so the result still sums to 1."""
        if current_weights is None:
            return new_weights
        out = {}
        for name, w_new in new_weights.items():
            w_old = current_weights.get(name, 0.0)
            out[name] = w_new if abs(w_new - w_old) > self.no_trade_band else w_old
        total = sum(out.values())
        if total == 0:
            return new_weights
        return {name: w / total for name, w in out.items()}

    def build_weight_series(self, scores_df, full_index):
        """`scores_df`: rebalance-date-indexed DataFrame of per-strategy
        scores (from AlphaMetaModel.walk_forward_fit_predict). Returns a
        DataFrame of DAILY weights over `full_index`, held constant
        between rebalance dates and passed through the no-trade band at
        each transition."""
        names = list(scores_df.columns)
        weight_rows = []
        current = None
        for date, row in scores_df.iterrows():
            raw_weights = self.scores_to_weights(row.to_dict())
            current = self.apply_no_trade_band(raw_weights, current)
            weight_rows.append({"date": date, **current})

        weights_at_rebalance = pd.DataFrame(weight_rows).set_index("date")
        daily = weights_at_rebalance.reindex(full_index).ffill()
        daily = daily.fillna(1.0 / len(names))  # before the first rebalance: equal weight
        return daily

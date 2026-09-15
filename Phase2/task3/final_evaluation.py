"""
FinalEvaluator -- runs all five allocation methods on one common harness
so their performance numbers are directly comparable, and reports the
dynamic (learned) method against its null baselines rather than on its
headline number alone.

Static methods (best_individual, equal_weight, inverse_volatility,
mean_variance_shrinkage) are fit ONCE on the whole strategy-return
matrix and held fixed -- they are baselines, not adaptive strategies, so
no rebalancing/turnover cost applies to them beyond the one-time
allocation. The dynamic method is the only one that actually changes
its weights through time, walk-forward, and is the only one that pays a
rebalancing cost.
"""

import numpy as np
import pandas as pd

import config
from portfolio_optimizer import PortfolioOptimizer
from meta_model import AlphaMetaModel
from dynamic_allocator import DynamicAllocator
from performance import PerformanceAnalyzer

REBALANCE_COST_RATE = config.TRANSACTION_COST_PER_SIDE  # reallocating between strategies
                                                          # is charged at the same rate as any
                                                          # other trade in the underlying asset


class FinalEvaluator:
    def __init__(self, returns_dict, caveat=None):
        self.returns_dict = returns_dict
        self.matrix = pd.DataFrame(returns_dict).dropna(how="any")
        self.optimizer = PortfolioOptimizer()
        self.caveat = caveat

    def _portfolio_returns_static(self, weights):
        w = pd.Series(weights).reindex(self.matrix.columns).fillna(0.0)
        return (self.matrix * w).sum(axis=1)

    def _portfolio_returns_dynamic(self, weights_daily):
        w_lagged = weights_daily.shift(1).reindex(self.matrix.index).ffill().fillna(
            1.0 / weights_daily.shape[1]
        )
        turnover = weights_daily.diff().abs().sum(axis=1).reindex(self.matrix.index).fillna(0.0)
        cost = turnover * REBALANCE_COST_RATE
        gross = (self.matrix * w_lagged).sum(axis=1)
        return gross - cost

    def evaluate_static_methods(self):
        methods = ["best_individual", "equal_weight", "inverse_volatility", "mean_variance_shrinkage"]
        out = {}
        for method in methods:
            weights = self.optimizer.compute_weights(method, self.returns_dict)
            returns = self._portfolio_returns_static(weights)
            report = PerformanceAnalyzer(returns).report()
            # `returns` (the raw daily series) is kept alongside the report
            # so plotting.py can draw an equity curve -- it is never written
            # to task3_report.json, main.py only pulls the JSON-safe fields.
            out[method] = {"weights": weights, "performance": report, "returns": returns}
        return out

    def evaluate_dynamic_method(self):
        meta = AlphaMetaModel()
        scores_df = meta.walk_forward_fit_predict(self.matrix)
        null_report = meta.null_baseline(self.matrix, n_shuffles=100)

        allocator = DynamicAllocator()
        weights_daily = allocator.build_weight_series(scores_df, self.matrix.index)
        returns = self._portfolio_returns_dynamic(weights_daily)
        report = PerformanceAnalyzer(returns).report()

        avg_weights = weights_daily.mean().to_dict()
        return {
            "weights_avg_over_time": avg_weights,
            "performance": report,
            "walk_forward_folds": meta.fold_reports_,
            "null_baseline": null_report,
            # Raw daily series, for plotting.py only (not JSON-serialized by
            # main.py): the full weight path and the portfolio's own return
            # series, which the report's summary metrics are computed from.
            "weights_daily": weights_daily,
            "returns": returns,
        }

    def evaluate_all(self):
        static = self.evaluate_static_methods()
        dynamic = self.evaluate_dynamic_method()

        comparison = {
            method: res["performance"]
            for method, res in static.items()
        }
        comparison["dynamic_meta_model"] = dynamic["performance"]

        return {
            "strategies_combined": list(self.matrix.columns),
            "n_common_bars": len(self.matrix),
            "caveat": self.caveat,
            "static_methods": static,
            "dynamic_method": dynamic,
            "comparison_table": comparison,
        }

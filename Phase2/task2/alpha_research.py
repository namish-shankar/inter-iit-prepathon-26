"""
AlphaResearch -- orchestrates every strategy through the shared
Backtester, characterises the raw signal library, and runs the full
significance / robustness / orthogonality suite over the resulting
return set. This is the module main.py calls; it does not implement any
statistics itself (that lives in statistics.py / robustness.py /
orthogonality.py) -- it wires them together and applies the project's
selection criteria.
"""

import numpy as np
import pandas as pd

import config
from backtester import Backtester
from feature_engine import FeatureEngine
from statistics import StatisticalTester
from robustness import RobustnessTester
from orthogonality import OrthogonalityAnalyzer


class AlphaResearch:
    def __init__(self, data):
        self.data = data
        self.stat_tester = StatisticalTester()
        self.robustness_tester = RobustnessTester()
        self.orthogonality_analyzer = OrthogonalityAnalyzer()
        self.strategy_results = {}   # name -> full Backtester.get_results()
        self.returns = {}            # name -> net return Series

    # ------------------------------------------------------ signal table --
    def characterize_signals(self, horizons=(1, 3, 5, 10, 20)):
        """Hit-rate and IC of every raw signal column against the forward
        price DIFFERENCE, at several horizons -- the reference table every
        strategy hypothesis is checked against before it is trusted."""
        fe = FeatureEngine()
        rows = []
        for h in horizons:
            diff_data = fe.forward_diff(self.data, price_col="close", horizon=h)
            fwd_diff = diff_data[f"fwd_diff_{h}"]
            for col in config.ALL_SIGNALS:
                if col not in self.data.columns:
                    continue
                lagged_signal = self.data[col].shift(config.SIGNAL_LAG)
                # Center boolean flags so a 0/1 flag's *sign* (not just its
                # level) lines up with a directional forward diff.
                centered = lagged_signal - 0.5 if col in config.BOOLEAN_SIGNALS else lagged_signal
                hit_rate, n = self.stat_tester.hit_rate(centered, fwd_diff)
                ic = self.stat_tester.information_coefficient(lagged_signal, fwd_diff)
                rows.append({"signal": col, "horizon": h, "hit_rate": hit_rate,
                             "ic": ic, "n_obs": n})
        return pd.DataFrame(rows)

    # --------------------------------------------------------- backtests --
    def run_strategy(self, strategy):
        """Fit (no-op for rule-based strategies) and backtest one
        strategy, storing its full result bundle and net-return series."""
        strategy.fit(self.data)
        bt = Backtester()
        results = bt.run(self.data, strategy, check_lookahead=True)
        self.strategy_results[strategy.name] = results
        self.returns[strategy.name] = bt.portfolio.get_pnl()
        return results

    def run_all(self, strategies):
        for s in strategies:
            self.run_strategy(s)
        return self.strategy_results

    # -------------------------------------------------------- statistics --
    def run_statistics(self, horizon=5):
        """t-stat (Newey-West), permutation p-value, and bootstrap CI on
        Sharpe for every strategy already run, then a multiple-testing
        correction across the whole set (never evaluate one strategy's
        significance in isolation from how many were tried)."""
        fe = FeatureEngine()
        fwd_diff = fe.forward_diff(self.data, "close", horizon)[f"fwd_diff_{horizon}"]

        price_cols = [c for c in config.PRICE_COLUMNS if c in self.data.columns]
        market_data = self.data[price_cols]

        out = {}
        p_values = {}
        for name, r in self.returns.items():
            signal = self.strategy_results[name]["signals"]
            position = self.strategy_results[name]["executions"]["position"]
            hit_rate, n_signal_obs = self.stat_tester.hit_rate(signal, fwd_diff)
            ic = self.stat_tester.information_coefficient(signal, fwd_diff)
            t_stat = self.stat_tester.t_stat_with_newey_west(r)
            perm = self.stat_tester.permutation_test(r, position, market_data, n_permutations=500)
            boot = self.stat_tester.block_bootstrap_ci(r)

            out[name] = {
                "hit_rate": hit_rate, "hit_rate_n_obs": n_signal_obs,
                "information_coefficient": ic,
                "t_stat_newey_west": t_stat,
                "permutation_p_value": perm["p_value"],
                "sharpe_ci_90pct": (boot["lower"], boot["upper"]),
            }
            p_values[name] = perm["p_value"] if not np.isnan(perm["p_value"]) else 1.0

        correction = self.stat_tester.multiple_testing_correction(p_values, method="benjamini_hochberg")
        for name in out:
            out[name]["significant_after_bh_correction"] = correction[name]["significant"]
            out[name]["bh_adjusted_p_value"] = correction[name]["adjusted_p_value"]
        return out

    # -------------------------------------------------------- robustness --
    def run_robustness(self, strategies_by_name, regime_signal="BB05"):
        out = {}
        for name, strategy in strategies_by_name.items():
            r = self.returns[name]
            sub_periods = self.robustness_tester.sub_period_analysis(r, n_splits=4)

            def run_fn(cost, _strategy=strategy):
                bt = Backtester(cost_per_side=cost)
                bt.run(self.data, _strategy, check_lookahead=False)
                return bt.portfolio.get_pnl()

            cost_sens = self.robustness_tester.cost_sensitivity(run_fn)

            regime = self.data[regime_signal].shift(config.SIGNAL_LAG) if regime_signal in self.data.columns else None
            regime_result = (
                self.robustness_tester.regime_conditioning(
                    r, regime, labels={0: "no_squeeze", 1: "squeeze"}
                )
                if regime is not None else {}
            )

            out[name] = {
                "sub_periods": sub_periods,
                "cost_sensitivity": cost_sens,
                "regime_conditioning": regime_result,
            }
        return out

    # ------------------------------------------------------ orthogonality --
    def run_orthogonality(self):
        matrix = self.orthogonality_analyzer.build_return_matrix(self.returns)
        return {
            "n_common_bars": len(matrix),
            "correlation_matrix": matrix.corr().round(3).to_dict(),
            "pivoted_qr": self.orthogonality_analyzer.pivoted_qr(matrix),
            "residual_alpha": self.orthogonality_analyzer.residual_alpha(matrix),
            "rolling_stability": self.orthogonality_analyzer.rolling_window_stability(matrix),
        }

    # --------------------------------------------------------- selection --
    def select_final_set(self, stats, orthogonality, balance_report):
        """Selection criteria, applied together rather than on Sharpe
        alone: (1) statistically significant after multiple-testing
        correction, (2) not made redundant by the rest of the set
        (residual variance-explained-by-others not near 1), (3) no single
        metric wildly out of line with the strategy's own other metrics
        (the 'good-ish everywhere' rule)."""
        residual = orthogonality["residual_alpha"]
        selected, notes = [], {}
        for name, s in stats.items():
            reasons = []
            keep = True

            if not s["significant_after_bh_correction"]:
                keep = False
                reasons.append("not significant after BH correction")

            explained = residual.get(name, {}).get("variance_explained_by_others", 0.0)
            if explained is not None and explained > 0.85:
                keep = False
                reasons.append(f"largely redundant with rest of set (explained={explained:.2f})")

            balance_flag = balance_report.get(name, {}).get("balanced", True)
            if not balance_flag:
                reasons.append("metric profile spiky (see balance_report) -- kept but flagged")

            notes[name] = {"kept": keep, "reasons": reasons or ["passed all screens"]}
            if keep:
                selected.append(name)

        # Honest outcome on this sample: it is entirely possible for NO
        # candidate to clear the statistical bar once multiple-testing
        # correction is applied -- that is a finding, not a bug, and it is
        # reported as such rather than papered over. `practical_set`
        # exists only so Task 3's allocation machinery (which needs >=2
        # return streams to combine) has something to run on; it is
        # explicitly NOT a claim that these strategies are validated
        # alpha, and every consumer of it must say so.
        practical_set = None
        if not selected:
            ranked = sorted(
                stats.items(),
                key=lambda kv: (self.returns[kv[0]].mean() / self.returns[kv[0]].std()
                                 if self.returns[kv[0]].std() else -999),
                reverse=True,
            )
            practical_set = [name for name, _ in ranked[:3]]

        return {
            "selected": selected,
            "notes": notes,
            "practical_set_for_task3": practical_set,
            "practical_set_caveat": (
                None if practical_set is None else
                "No strategy cleared the BH-corrected significance bar on this sample. "
                "practical_set_for_task3 is the top-3 by unadjusted Sharpe, used ONLY to "
                "demonstrate the Task 3 allocation machinery -- it is not a validated edge."
            ),
        }

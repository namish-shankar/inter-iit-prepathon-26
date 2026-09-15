"""
RobustnessTester -- does the edge hold up away from the exact window,
parameters, and cost level it was found at? A strategy that only works
in one sub-period, one cost regime, or one narrow parameter setting is
the "great in one place, broken everywhere else" pattern this whole
project is built to avoid.
"""

import numpy as np
import pandas as pd

import config
from performance import PerformanceAnalyzer


class RobustnessTester:
    def __init__(self, annualization_factor=None):
        self.ann_factor = annualization_factor or config.ANNUALIZATION_FACTOR

    def _sharpe(self, returns):
        r = returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float(np.sqrt(self.ann_factor) * r.mean() / r.std())

    # --------------------------------------------------------- sub-periods --
    def sub_period_analysis(self, returns, n_splits=4):
        """Split the return series into n_splits contiguous, roughly
        equal blocks and report Sharpe/return per block -- a strategy
        whose edge is really one lucky quarter shows up here as wildly
        uneven blocks, not a steady one."""
        r = returns.dropna()
        if len(r) < n_splits * 10:
            return {"splits": [], "note": "too few observations for a meaningful split"}

        edges = np.linspace(0, len(r), n_splits + 1, dtype=int)
        splits = []
        for i in range(n_splits):
            block = r.iloc[edges[i]:edges[i + 1]]
            splits.append({
                "start": str(block.index.min().date()) if len(block) else None,
                "end": str(block.index.max().date()) if len(block) else None,
                "n_bars": len(block),
                "total_return": float((1 + block).prod() - 1) if len(block) else 0.0,
                "sharpe": self._sharpe(block),
            })
        sharpes = [s["sharpe"] for s in splits]
        return {
            "splits": splits,
            "sharpe_std_across_splits": float(np.std(sharpes)),
            "fraction_positive_sharpe_splits": float(np.mean([s > 0 for s in sharpes])),
        }

    # ------------------------------------------------------------ params --
    def parameter_sweep(self, run_fn, param_grid):
        """`run_fn(params) -> net_return_series`. Reruns the backtest
        across every combination in `param_grid` (a dict of name ->
        list of values) and reports Sharpe for each -- looking for a
        plateau (robust) versus an isolated spike (overfit to one exact
        setting)."""
        import itertools

        keys = list(param_grid.keys())
        results = []
        for combo in itertools.product(*param_grid.values()):
            params = dict(zip(keys, combo))
            returns = run_fn(params)
            results.append({**params, "sharpe": self._sharpe(returns)})

        sharpes = np.array([r["sharpe"] for r in results])
        return {
            "results": results,
            "sharpe_range": float(sharpes.max() - sharpes.min()) if len(sharpes) else np.nan,
            "sharpe_std": float(sharpes.std()) if len(sharpes) else np.nan,
        }

    # -------------------------------------------------------------- cost --
    def cost_sensitivity(self, run_fn, cost_levels=None):
        """`run_fn(cost_per_side) -> net_return_series`. Reports Sharpe at
        each cost level and (by linear interpolation) the approximate
        break-even cost -- how much of a cushion the edge has over the
        mandated 0.05%/side before it stops paying."""
        if cost_levels is None:
            cost_levels = [0.0, 0.00025, 0.0005, 0.00075, 0.0010, 0.0015, 0.0020]

        rows = []
        for c in cost_levels:
            returns = run_fn(c)
            total_return = float((1 + returns.dropna()).prod() - 1) if len(returns.dropna()) else 0.0
            rows.append({"cost_per_side": c, "sharpe": self._sharpe(returns),
                         "total_return": total_return})

        breakeven = None
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            if a["total_return"] >= 0 > b["total_return"]:
                span = b["total_return"] - a["total_return"]
                frac = (-a["total_return"] / span) if span != 0 else 0.0
                breakeven = a["cost_per_side"] + frac * (b["cost_per_side"] - a["cost_per_side"])
                break

        return {"rows": rows, "approx_breakeven_cost_per_side": breakeven}

    # ------------------------------------------------------------ regime --
    def regime_conditioning(self, returns, regime_series, labels=None):
        """Split returns by an external regime flag (e.g. a volatility
        band) and report performance per regime -- catches an edge that
        is really "works only when volatility is low" dressed up as an
        unconditional strategy."""
        r, reg = returns.align(regime_series, join="inner")
        out = {}
        for value in sorted(pd.unique(reg.dropna())):
            mask = reg == value
            block = r[mask]
            label = (labels or {}).get(value, str(value))
            out[label] = {
                "n_bars": int(mask.sum()),
                "total_return": float((1 + block.dropna()).prod() - 1) if mask.sum() else 0.0,
                "sharpe": self._sharpe(block),
            }
        return out

"""
StatisticalTester -- is an edge real, or noise that got lucky?

Every strategy's return series is fat-tailed and autocorrelated (trades
cluster, positions persist for many bars), so a plain i.i.d. t-test on
daily returns overstates significance. Every test here either adjusts
for that (Newey-West) or sidesteps the distributional assumption
entirely (permutation, block bootstrap).
"""

import numpy as np
import pandas as pd

import config
from execution_engine import ExecutionEngine
from portfolio import Portfolio


class StatisticalTester:
    def __init__(self, annualization_factor=None, seed=None):
        self.ann_factor = annualization_factor or config.ANNUALIZATION_FACTOR
        self.rng = np.random.default_rng(seed or config.RANDOM_SEED)

    # ------------------------------------------------------- hit / IC ----
    def hit_rate(self, signal, forward_diff):
        """Fraction of bars where sign(signal) called the direction of
        the forward price DIFFERENCE correctly. Undefined (NaN) bars
        (flat signal, or no forward diff available) are excluded, not
        counted as wrong."""
        s, f = signal.align(forward_diff, join="inner")
        mask = (s != 0) & f.notna() & (np.sign(f) != 0)
        if mask.sum() == 0:
            return np.nan, 0
        correct = (np.sign(s[mask]) == np.sign(f[mask])).mean()
        return float(correct), int(mask.sum())

    def information_coefficient(self, signal, forward_diff):
        """Pearson correlation between the signal and the forward price
        DIFFERENCE it is trying to call -- the size of the edge, as
        opposed to hit_rate's direction-only view."""
        s, f = signal.align(forward_diff, join="inner")
        mask = s.notna() & f.notna()
        if mask.sum() < 10 or s[mask].std() == 0 or f[mask].std() == 0:
            return np.nan
        return float(np.corrcoef(s[mask], f[mask])[0, 1])

    # --------------------------------------------------------- t-stats --
    def t_stat_with_newey_west(self, returns, lag=None):
        """Newey-West heteroskedasticity- and autocorrelation-consistent
        t-stat on the mean return -- corrects the standard error for the
        serial correlation a position-holding strategy's daily P&L
        necessarily has (a naive i.i.d. t-stat is systematically too
        confident here)."""
        r = returns.dropna().to_numpy()
        n = len(r)
        if n < 10 or r.std() == 0:
            return np.nan
        if lag is None:
            lag = int(np.floor(4 * (n / 100) ** (2 / 9))) + 1  # standard rule of thumb

        mean = r.mean()
        centered = r - mean
        gamma0 = np.mean(centered ** 2)
        var = gamma0
        for k in range(1, min(lag, n - 1) + 1):
            weight = 1 - k / (lag + 1)
            gamma_k = np.mean(centered[k:] * centered[:-k])
            var += 2 * weight * gamma_k
        se = np.sqrt(max(var, 1e-16) / n)
        if se == 0:
            return np.nan
        return float(mean / se)

    # ------------------------------------------------------- resampling --
    def permutation_test(self, returns, position, market_data, n_permutations=500, statistic="sharpe"):
        """Null: the strategy's edge comes from being long/short at the
        right TIMES, not merely from being long/short in general, or from
        the market's own drift plus costs.

        IMPORTANT -- this does NOT shuffle the already-realised return
        series. Sharpe (mean / std) depends only on the *multiset* of
        values, never their order, so any reordering of the same numbers
        -- a circular roll included -- reproduces the identical Sharpe up
        to floating-point noise. Permuting the realised P&L directly can
        never build a real null distribution; an earlier version of this
        method did exactly that, and its "null distributions" were a
        single spike on the observed value every time (caught via the
        permutation-test plot in plotting.py, where it was visually
        obvious).

        The null here is built correctly instead: circularly shift the
        POSITION series while keeping the market's own OHLC data fixed in
        its real chronological order, re-run each shifted position
        through the exact same ExecutionEngine + Portfolio the real
        backtest uses (so cost and slippage are computed identically),
        and take the resulting statistic. This keeps the strategy's own
        trading structure -- how often it trades, how long it holds,
        total turnover -- exactly as observed, and only scrambles whether
        that structure happens to line up with the real market moves."""
        r = returns.dropna()
        if len(r) < 20:
            return {"p_value": np.nan, "n_permutations": 0}
        observed = self._stat(r, statistic)

        pos = position.reindex(market_data.index).fillna(0.0)
        if len(pos) < 20:
            return {"observed": observed, "p_value": np.nan, "n_permutations": 0}

        arr = pos.to_numpy()
        shifts = self.rng.integers(1, len(pos), size=n_permutations)
        null_stats = np.empty(n_permutations)
        for i, k in enumerate(shifts):
            shifted_signal = pd.Series(np.roll(arr, k), index=pos.index)
            executions = ExecutionEngine().execute(shifted_signal, market_data)
            synthetic_equity = Portfolio().update(executions, market_data)
            synthetic_returns = synthetic_equity.pct_change().fillna(0.0)
            null_stats[i] = self._stat(synthetic_returns, statistic)

        p_value = float((np.sum(null_stats >= observed) + 1) / (n_permutations + 1))
        return {
            "observed": observed,
            "p_value": p_value,
            "n_permutations": n_permutations,
            "null_mean": float(np.nanmean(null_stats)),
            "null_std": float(np.nanstd(null_stats)),
            # Raw per-shift statistics, not just their mean/std -- kept so
            # plotting.py can draw the actual null distribution rather than
            # a synthetic normal approximation around the mean.
            "null_stats": null_stats.tolist(),
        }

    def block_bootstrap_ci(self, returns, block_size=20, n_boot=500, ci=0.90, statistic="sharpe"):
        """Confidence interval on the statistic via moving-block
        bootstrap -- resamples contiguous blocks (preserving local
        autocorrelation) rather than individual bars."""
        r = returns.dropna().to_numpy()
        n = len(r)
        if n < block_size * 2:
            return {"lower": np.nan, "upper": np.nan, "n_boot": 0}

        n_blocks = int(np.ceil(n / block_size))
        boot_stats = np.empty(n_boot)
        starts_range = n - block_size
        for b in range(n_boot):
            starts = self.rng.integers(0, starts_range, size=n_blocks)
            sample = np.concatenate([r[s:s + block_size] for s in starts])[:n]
            boot_stats[b] = self._stat(pd.Series(sample), statistic)

        alpha = (1 - ci) / 2
        lower, upper = np.nanquantile(boot_stats, [alpha, 1 - alpha])
        return {"lower": float(lower), "upper": float(upper), "n_boot": n_boot, "ci": ci}

    def _stat(self, r, kind):
        if kind == "sharpe":
            std = r.std()
            if std == 0 or np.isnan(std):
                return 0.0
            return float(np.sqrt(self.ann_factor) * r.mean() / std)
        if kind == "mean_return":
            return float(r.mean())
        raise ValueError(f"Unknown statistic: {kind}")

    # --------------------------------------------------- multiple tests --
    def multiple_testing_correction(self, p_values, method="benjamini_hochberg", alpha=0.10):
        """Correct for testing several strategies/horizons at once -- with
        5 strategies x a handful of horizons, even a 10% per-test false
        positive rate compounds fast. Returns which tests remain
        significant after correction, and at what adjusted threshold/
        p-value."""
        names = list(p_values.keys())
        pvals = np.array([p_values[n] for n in names], dtype=float)
        m = len(pvals)
        if m == 0:
            return {}

        if method == "bonferroni":
            threshold = alpha / m
            significant = pvals <= threshold
            adjusted = np.minimum(pvals * m, 1.0)
        elif method == "benjamini_hochberg":
            order = np.argsort(pvals)
            ranked = pvals[order]
            thresholds = (np.arange(1, m + 1) / m) * alpha
            passed = ranked <= thresholds
            k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
            significant_sorted = np.zeros(m, dtype=bool)
            significant_sorted[:k] = True
            significant = np.empty(m, dtype=bool)
            significant[order] = significant_sorted
            adjusted_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
            adjusted = np.empty(m)
            adjusted[order] = np.clip(adjusted_sorted, 0, 1)
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            name: {"p_value": float(pvals[i]), "adjusted_p_value": float(adjusted[i]),
                   "significant": bool(significant[i])}
            for i, name in enumerate(names)
        }

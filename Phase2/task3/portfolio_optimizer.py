"""
PortfolioOptimizer -- the five required allocation methods, all
operating on the same input: a dict of name -> net-return Series for
the strategies being combined.

Every method returns a plain dict of {strategy_name: weight}, weights
summing to 1 and non-negative (capital allocated TO a strategy, not a
short position against one -- a strategy that itself goes short
already expresses that in its own returns).
"""

import numpy as np
import pandas as pd

import config


class PortfolioOptimizer:
    def __init__(self, annualization_factor=None):
        self.ann_factor = annualization_factor or config.ANNUALIZATION_FACTOR

    def _returns_matrix(self, returns_dict):
        return pd.DataFrame(returns_dict).dropna(how="any")

    # ------------------------------------------------------------ (1) --
    def best_individual(self, returns_dict, metric="sharpe"):
        """All capital on whichever single strategy has the best
        in-sample Sharpe -- the naive baseline every other method must
        beat to justify its own complexity."""
        scores = {}
        for name, r in returns_dict.items():
            r = r.dropna()
            std = r.std()
            scores[name] = float(np.sqrt(self.ann_factor) * r.mean() / std) if std else -np.inf
        best = max(scores, key=scores.get)
        return {name: (1.0 if name == best else 0.0) for name in returns_dict}

    # ------------------------------------------------------------ (2) --
    def equal_weight(self, returns_dict):
        n = len(returns_dict)
        return {name: 1.0 / n for name in returns_dict}

    # ------------------------------------------------------------ (3) --
    def inverse_volatility(self, returns_dict):
        """Risk parity in its simplest form: weight inversely
        proportional to each strategy's own volatility, ignoring
        cross-correlation (that's what (4) adds)."""
        vols = {name: r.dropna().std() for name, r in returns_dict.items()}
        inv = {name: (1.0 / v if v and v > 0 else 0.0) for name, v in vols.items()}
        total = sum(inv.values())
        if total == 0:
            return self.equal_weight(returns_dict)
        return {name: w / total for name, w in inv.items()}

    # ------------------------------------------------------------ (4) --
    def mean_variance_shrinkage(self, returns_dict, shrinkage=0.3):
        """Markowitz max-Sharpe direction (w proportional to
        Sigma^-1 * mu), with the sample covariance shrunk toward a
        diagonal (equal-variance, zero-correlation) target -- the
        standard fix for a sample covariance matrix estimated from only
        a few hundred return observations, which is otherwise nearly
        singular and produces wild, unstable weights. Negative raw
        weights are clipped to 0 and the result renormalised (this
        implementation allocates non-negative capital across strategies,
        it does not short one strategy to lever another)."""
        R = self._returns_matrix(returns_dict)
        names = list(R.columns)
        if len(R) < 20 or len(names) < 2:
            return self.equal_weight(returns_dict)

        mu = R.mean().to_numpy()
        sample_cov = R.cov().to_numpy()
        target = np.diag(np.diag(sample_cov))  # same variances, zero off-diagonal
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target

        try:
            inv_cov = np.linalg.pinv(shrunk_cov)
        except np.linalg.LinAlgError:
            return self.equal_weight(returns_dict)

        raw_w = inv_cov @ mu
        raw_w = np.clip(raw_w, 0, None)
        total = raw_w.sum()
        if total <= 0:
            return self.equal_weight(returns_dict)
        weights = raw_w / total
        return dict(zip(names, weights.tolist()))

    # ------------------------------------------------------------ (5) --
    def from_scores(self, scores):
        """Turn arbitrary per-strategy conviction scores (e.g. from
        AlphaMetaModel) into non-negative, sum-to-1 weights via a
        softmax over positive scores. All-non-positive scores fall back
        to equal weight rather than an undefined/degenerate allocation --
        see dynamic_allocator.py for the rebalance-time wrapper around
        this (no-trade band, walk-forward score source)."""
        names = list(scores.keys())
        vals = np.array([max(scores[n], 0.0) for n in names])
        if vals.sum() == 0:
            return {n: 1.0 / len(names) for n in names}
        exp = np.exp(vals - vals.max())
        w = exp / exp.sum()
        return dict(zip(names, w.tolist()))

    # --------------------------------------------------------- dispatch --
    def compute_weights(self, method, returns_dict, **kwargs):
        dispatch = {
            "best_individual": self.best_individual,
            "equal_weight": self.equal_weight,
            "inverse_volatility": self.inverse_volatility,
            "mean_variance_shrinkage": self.mean_variance_shrinkage,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown method: {method}")
        return dispatch[method](returns_dict, **kwargs)

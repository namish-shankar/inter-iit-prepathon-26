"""
FactorModel -- separates a strategy's return into alpha (skill/timing)
and beta (just being correlated with the underlying's own drift), via
OLS: r_i(t) = alpha_i + beta_i * F(t) + eps_i(t).

This is a single-asset dataset (one instrument's signal library), so
there is no cross-sectional "market index" to use as the factor. The
natural, well-motivated choice of F(t) here is the asset's OWN
buy-and-hold (open-to-open) return: it isolates how much of a
strategy's return is just "being long the asset's underlying drift"
(beta) versus genuine timing skill on top of that (alpha) -- exactly
the question a factor decomposition is for for a single-instrument
strategy set.
"""

import numpy as np
import pandas as pd

import config


class FactorModel:
    def __init__(self, annualization_factor=None):
        self.ann_factor = annualization_factor or config.ANNUALIZATION_FACTOR
        self.results_ = {}

    def compute_factor_returns(self, market_data):
        """The single "market" factor for this problem: buy-and-hold
        open-to-open return of the underlying asset."""
        return market_data["open"].pct_change().rename("factor_buy_and_hold")

    def _ols_alpha_beta(self, y, x):
        df = pd.DataFrame({"y": y, "x": x}).dropna()
        n = len(df)
        if n < 10 or df["x"].std() == 0:
            return {"alpha": np.nan, "beta": np.nan, "r_squared": np.nan,
                    "alpha_t_stat": np.nan, "n_obs": n}

        X = np.column_stack([np.ones(n), df["x"].to_numpy()])
        yv = df["y"].to_numpy()
        coef, residuals_ss, rank, sv = np.linalg.lstsq(X, yv, rcond=None)
        alpha, beta = coef
        fitted = X @ coef
        resid = yv - fitted

        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        dof = max(n - 2, 1)
        sigma2 = ss_res / dof
        xtx_inv = np.linalg.inv(X.T @ X)
        se_alpha = float(np.sqrt(max(sigma2 * xtx_inv[0, 0], 0)))
        alpha_t_stat = float(alpha / se_alpha) if se_alpha > 0 else np.nan

        return {
            "alpha": float(alpha), "beta": float(beta), "r_squared": float(r_squared),
            "alpha_t_stat": alpha_t_stat, "n_obs": n,
            "alpha_annualized": float(alpha) * self.ann_factor,
        }

    def fit(self, strategy_returns, factor_returns):
        """`strategy_returns`: name -> net-return Series. Fits one
        regression per strategy against the shared factor series."""
        self.results_ = {
            name: self._ols_alpha_beta(r, factor_returns)
            for name, r in strategy_returns.items()
        }
        return self.results_

    def get_report(self):
        return dict(self.results_)

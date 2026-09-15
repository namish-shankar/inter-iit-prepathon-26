"""
OrthogonalityAnalyzer -- are the strategies actually different bets, or
is one strategy's edge just a repackaging of another's?

Five strategies with positive Sharpe are worth a lot less than five
strategies with positive Sharpe AND low mutual correlation -- the first
set might be one real signal wearing five costumes. Pivoted QR gives an
honest count of how many genuinely independent return streams are
actually present, which correlation alone (pairwise, not joint) can miss.
"""

import numpy as np
import pandas as pd
from scipy.linalg import qr


class OrthogonalityAnalyzer:
    def build_return_matrix(self, returns_dict):
        """`returns_dict`: name -> net-return Series. Aligns on the common
        index and drops any row where a strategy has no observation
        (rather than filling 0, which would fabricate a flat day for a
        strategy that simply had no signal that bar)."""
        df = pd.DataFrame(returns_dict)
        return df.dropna(how="any")

    def correlation_matrix(self, return_matrix):
        return return_matrix.corr()

    def pivoted_qr(self, return_matrix, tolerance=1e-6):
        """QR decomposition with column pivoting on the (de-meaned)
        return matrix. The diagonal of R, taken in pivot order, is a
        direct read on effective dimensionality: a diagonal entry near
        zero means that column is (numerically) a linear combination of
        earlier-pivoted columns -- i.e. it adds no new information."""
        X = (return_matrix - return_matrix.mean()).to_numpy()
        # Scale each column to unit variance so the pivoting isn't just
        # picking out whichever strategy happens to have the largest raw
        # return magnitude.
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Xs = X / scale

        _, R, piv = qr(Xs, mode="economic", pivoting=True)
        diag = np.abs(np.diag(R))
        rel_diag = diag / diag[0] if diag[0] > 0 else diag
        effective_rank = int(np.sum(rel_diag > tolerance))

        cols = list(return_matrix.columns)
        pivot_order = [cols[i] for i in piv]

        return {
            "pivot_order": pivot_order,
            "diagonal_magnitudes": diag.tolist(),
            "relative_diagonal_magnitudes": rel_diag.tolist(),
            "effective_rank": effective_rank,
            "n_strategies": len(cols),
        }

    def residual_alpha(self, return_matrix):
        """For each strategy, regress its returns on all the OTHERS and
        report the residual's mean (its "alpha" beyond what the rest of
        the set already explains) and what fraction of its variance is
        NOT explained by the rest -- a strategy that is almost entirely
        explained by the others (low residual variance fraction, residual
        mean near zero) is redundant even if its own Sharpe looks fine in
        isolation."""
        out = {}
        cols = list(return_matrix.columns)
        for target in cols:
            others = [c for c in cols if c != target]
            y = return_matrix[target].to_numpy()
            X = return_matrix[others].to_numpy()
            X_design = np.column_stack([np.ones(len(X)), X])
            coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            fitted = X_design @ coef
            resid = y - fitted

            total_var = y.var()
            resid_var = resid.var()
            explained_frac = 1 - (resid_var / total_var) if total_var > 0 else np.nan

            se = resid.std() / np.sqrt(len(resid)) if len(resid) > 1 else np.nan
            resid_t_stat = (resid.mean() / se) if se and se > 0 else np.nan

            out[target] = {
                "residual_mean": float(resid.mean()),
                "residual_t_stat": float(resid_t_stat) if not np.isnan(resid_t_stat) else np.nan,
                "variance_explained_by_others": float(explained_frac) if not np.isnan(explained_frac) else np.nan,
            }
        return out

    def rolling_window_stability(self, return_matrix, window=250, step=20, tolerance=1e-6):
        """Re-run pivoted_qr over rolling windows and check whether the
        top pivot (the "most information-bearing" strategy) and the
        effective rank stay stable through time -- an orthogonality
        result that only holds in one window is not a structural
        property of the strategy set."""
        n = len(return_matrix)
        if n < window + step:
            return {"windows": [], "note": "series too short for the requested window/step"}

        results = []
        for start in range(0, n - window + 1, step):
            block = return_matrix.iloc[start:start + window]
            if block.std().min() == 0:
                continue
            qr_result = self.pivoted_qr(block, tolerance=tolerance)
            results.append({
                "window_start": str(block.index.min().date()),
                "window_end": str(block.index.max().date()),
                "top_pivot": qr_result["pivot_order"][0],
                "effective_rank": qr_result["effective_rank"],
            })

        if not results:
            return {"windows": [], "note": "no valid windows"}

        top_pivots = [r["top_pivot"] for r in results]
        most_common_top = max(set(top_pivots), key=top_pivots.count)
        stability = top_pivots.count(most_common_top) / len(top_pivots)

        return {
            "windows": results,
            "most_common_top_pivot": most_common_top,
            "top_pivot_stability": float(stability),
            "effective_rank_mean": float(np.mean([r["effective_rank"] for r in results])),
            "effective_rank_std": float(np.std([r["effective_rank"] for r in results])),
        }

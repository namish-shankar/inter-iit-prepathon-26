"""
FeatureEngine -- registers and applies derived columns, builds return
series for accounting, and provides the look-ahead checks the framework
is required to run.

Price/volume never leave this file as a *feature*: compute_returns() and
the market-return helpers below exist only to serve the two permitted
roles from the Problem Statement -- forward-return training targets and
execution/accounting inputs -- never as an input to generate_features or
generate_signal in a strategy.
"""

import numpy as np
import pandas as pd


class FeatureEngine:
    def __init__(self):
        self._registry = {}  # name -> function(data) -> Series

    # --------------------------------------------------------- registry --
    def add_feature(self, data, name, function):
        """Register a derived feature `function(data) -> Series` under
        `name`, and apply it immediately to the given frame (so callers
        can chain add_feature calls and see the column right away)."""
        self._registry[name] = function
        out = data.copy()
        out[name] = function(out)
        return out

    def transform(self, data):
        """Apply every registered feature function to `data`, in
        registration order (later features may depend on earlier ones)."""
        out = data.copy()
        for name, function in self._registry.items():
            out[name] = function(out)
        return out

    # ---------------------------------------------------------- returns --
    def compute_returns(self, data, price_col="close"):
        """Simple and log returns of `price_col`, for use ONLY as (a) an
        accounting/plotting convenience or (b) the forward-return label
        source in research -- never fed to generate_signal."""
        out = data.copy()
        out["simple_return"] = out[price_col].pct_change()
        out["log_return"] = np.log(out[price_col] / out[price_col].shift(1))
        return out

    def forward_diff(self, data, price_col="close", horizon=1):
        """Forward PRICE DIFFERENCE over `horizon` bars: diff_h(t) =
        price[t+h] - price[t]. This is the label used throughout Task 2/3
        research -- a difference, not the raw future level, and framed so
        that sign(diff_h) is the direction the strategy is actually
        scored against (see StatisticalTester.hit_rate)."""
        out = data.copy()
        out[f"fwd_diff_{horizon}"] = out[price_col].shift(-horizon) - out[price_col]
        out[f"fwd_dir_{horizon}"] = np.sign(out[f"fwd_diff_{horizon}"])
        return out

    # ------------------------------------------------------------ rolling --
    def rolling_feature(self, data, window, function, column=None):
        """Apply `function` to a rolling `window` of `column` (or the
        first numeric column if unspecified). `function` receives a numpy
        array and returns a scalar, e.g. `np.nanstd`."""
        if column is None:
            column = data.select_dtypes(include=[np.number]).columns[0]
        return data[column].rolling(window=window).apply(
            lambda arr: function(arr), raw=True
        )

    # ------------------------------------------------------- integrity --
    def validate_no_lookahead(self, data, signal_fn=None, n_checkpoints=5):
        """Two independent look-ahead checks:

        1. Truncation stability: generate_signal on data.iloc[:k] must
           agree with generate_signal on the full frame, restricted to
           the first k rows, for several k -- any rolling stat or model
           that peeks past its cutoff will disagree here.
        2. Perturbation: doubling a value strictly after a midpoint must
           never change any output at or before that midpoint.

        Returns True if no look-ahead was detected, False otherwise. If
        `signal_fn` is None, only a structural NaN-pattern sanity check
        runs (useful when validating a plain feature frame rather than a
        strategy's decision function).
        """
        if signal_fn is None:
            # Structural sanity check: warm-up NaNs should be a *prefix*
            # of each column, not scattered through the middle -- a NaN
            # reappearing after real values started would suggest a
            # feature draws on a window that can run out of data later.
            ok = True
            for col in data.select_dtypes(include=[np.number]).columns:
                s = data[col]
                first_valid = s.first_valid_index()
                if first_valid is None:
                    continue
                after = s.loc[first_valid:]
                if after.isna().any():
                    ok = False
            return ok

        n = len(data)
        checkpoints = np.linspace(n // 5, n - 1, n_checkpoints, dtype=int)
        full_signal = signal_fn(data)

        for k in checkpoints:
            truncated_signal = signal_fn(data.iloc[: k + 1])
            a = full_signal.iloc[: k + 1].fillna(0).to_numpy()
            b = truncated_signal.fillna(0).to_numpy()
            if not np.array_equal(a, b):
                return False

        mid = n // 2
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        if numeric_cols:
            perturbed = data.copy()
            col = numeric_cols[0]
            perturbed.iloc[mid, perturbed.columns.get_loc(col)] = (
                perturbed.iloc[mid][col] * 2.0 + 1.0
            )
            perturbed_signal = signal_fn(perturbed)
            a = full_signal.iloc[:mid].fillna(0).to_numpy()
            b = perturbed_signal.iloc[:mid].fillna(0).to_numpy()
            if not np.array_equal(a, b):
                return False

        return True

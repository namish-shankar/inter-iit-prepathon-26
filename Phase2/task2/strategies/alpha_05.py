"""
alpha_05 -- Learned composite of the continuous signals.

Model-choice reasoning (required to be stated explicitly, not just
implemented): the five continuous, bounded signals (PB07, PB08, BB06,
BB07, VB05) each carry a plausible directional prior on their own, but
no single one dominates -- a natural case for a model that *combines*
them rather than hand-picking one. Given ~1,000 bars total and a
development window of a few hundred bars after the 2021-01-01 cutoff,
the effective sample supports only a handful of free parameters: a
logistic classifier on 5 standardised features (6 parameters including
intercept) is the highest-capacity model defensible here. A random
forest or gradient-boosted tree, by contrast, would have hundreds of
effective degrees of freedom against a few hundred fitting rows -- a
recipe for memorising noise, not learning structure. So: a *linear*
model on *few* features, and the classifier predicts DIRECTION (sign of
the forward price difference), not the return magnitude, per the "price
difference for prediction, not price" framing this whole document uses.

Fit discipline: fit() trains only on rows strictly BEFORE
config.DEV_HOLDOUT_SPLIT_DATE, using price purely as the source of the
sign(forward-difference) label -- never as a feature. generate_features/
generate_signal never see price at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from strategy import BaseStrategy

CONTINUOUS = ["PB07", "PB08", "BB06", "BB07", "VB05"]
LABEL_HORIZON = 5  # bars ahead the model is trained to call direction over


class Alpha05LearnedComposite(BaseStrategy):
    name = "alpha_05_learned_composite"
    hypothesis = (
        "No single continuous signal dominates, but a low-capacity linear "
        "combination of the five bounded continuous signals can call "
        f"{LABEL_HORIZON}-bar-ahead price direction better than any one alone, "
        "without enough free parameters to memorise noise."
    )
    signals_used = CONTINUOUS

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.coef_ = None
        self.intercept_ = None
        self.fit_report_ = {}

    def generate_features(self, data):
        return data[CONTINUOUS].copy()

    def _standardize(self, X):
        return (X - self.mean_) / self.std_.replace(0, 1.0)

    def fit(self, data):
        """`data` is the FULL merged frame (signals + price), used only to
        build the label -- generate_features/generate_signal never receive
        price. Fits on the development period only (strictly before
        config.DEV_HOLDOUT_SPLIT_DATE); everything at/after that date is
        untouched by fitting, consistent with the holdout discipline used
        throughout this project."""
        dev = data.loc[data.index < pd.Timestamp(config.DEV_HOLDOUT_SPLIT_DATE)]
        X = dev[CONTINUOUS]
        fwd_diff = dev["close"].shift(-LABEL_HORIZON) - dev["close"]
        y = np.sign(fwd_diff)

        mask = X.notna().all(axis=1) & y.notna() & (y != 0)
        X, y = X.loc[mask], y.loc[mask]

        if len(X) < 30:
            # Not enough clean development rows to fit responsibly --
            # fall back to "always flat" rather than fit noise.
            self.fit_report_ = {"fitted": False, "reason": "insufficient dev rows", "n": len(X)}
            return self

        self.mean_ = X.mean()
        self.std_ = X.std()
        Xs = self._standardize(X).to_numpy()

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
        model.fit(Xs, y.to_numpy())

        self.coef_ = model.coef_.ravel()
        self.intercept_ = float(model.intercept_[0])
        train_acc = float(model.score(Xs, y.to_numpy()))
        self.fit_report_ = {
            "fitted": True,
            "n_train_rows": int(len(X)),
            "train_accuracy": train_acc,
            "coefficients": dict(zip(CONTINUOUS, self.coef_.tolist())),
        }
        return self

    def generate_signal(self, data):
        if self.coef_ is None:
            return pd.Series(0.0, index=data.index)

        X = data[CONTINUOUS]
        mask = X.notna().all(axis=1)
        position = pd.Series(0.0, index=data.index)

        Xs = self._standardize(X.loc[mask]).to_numpy()
        logit = Xs @ self.coef_ + self.intercept_
        position.loc[mask] = np.sign(logit)
        return position

    def get_metadata(self):
        meta = super().get_metadata()
        meta["fit_report"] = self.fit_report_
        return meta

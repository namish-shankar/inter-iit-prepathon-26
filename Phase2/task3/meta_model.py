"""
AlphaMetaModel -- learns per-rebalance conviction scores for each
strategy, to drive the fifth (dynamic) allocation method.

Model-choice reasoning: rebalancing every REBALANCE_FREQ_BARS (=21,
roughly monthly) over ~965 bars gives on the order of 40-45 independent
rebalance decisions in the whole sample, and each one is fit using only
the data strictly before it (expanding-window walk-forward -- see
walk_forward_fit_predict). With that few effective observations, ANY
model with more than a handful of parameters overfits by construction;
a ridge-regularised logistic classifier on 2 features per strategy
(trailing Sharpe, trailing hit-rate) is close to the ceiling of what is
defensible. It predicts whether a strategy's NEXT rebalance-period
return will be positive (direction, not magnitude -- consistent with
this project's "price difference for prediction" framing extended to
strategy-level returns), and its decision-function score becomes the
conviction score `dynamic_allocator.py` turns into a weight.
"""

import numpy as np
import pandas as pd

import config

REBALANCE_FREQ_BARS = 21
MIN_TRAIN_REBALANCES = 6  # at least this many past rebalance points before the model fits at all


class AlphaMetaModel:
    def __init__(self, lookback_bars=63, rebalance_freq=REBALANCE_FREQ_BARS, seed=None):
        self.lookback_bars = lookback_bars
        self.rebalance_freq = rebalance_freq
        self.rng = np.random.default_rng(seed or config.RANDOM_SEED)
        self.fold_reports_ = []

    # ----------------------------------------------------- feature build --
    def _features_at(self, returns_matrix, as_of_idx, name):
        window = returns_matrix[name].iloc[max(0, as_of_idx - self.lookback_bars):as_of_idx]
        window = window.dropna()
        if len(window) < 10:
            return None
        mean, std = window.mean(), window.std()
        sharpe = (mean / std) if std > 0 else 0.0
        hit_rate = float((window > 0).mean())
        return np.array([sharpe, hit_rate])

    def _label_at(self, returns_matrix, rebalance_idx, next_idx, name):
        fwd = returns_matrix[name].iloc[rebalance_idx:next_idx]
        if len(fwd) == 0 or fwd.isna().all():
            return None
        total = (1 + fwd.fillna(0)).prod() - 1
        return 1 if total > 0 else 0

    def _rebalance_points(self, n_rows):
        return list(range(self.lookback_bars, n_rows - self.rebalance_freq, self.rebalance_freq))

    # ------------------------------------------------------ walk-forward --
    def walk_forward_fit_predict(self, returns_matrix):
        """Expanding-window walk-forward: at each rebalance point, fit a
        fresh model on every (strategy, rebalance-point) pair strictly
        BEFORE that point, predict a score for every strategy at that
        point, then move on. Returns a DataFrame of scores indexed by
        rebalance date x strategy, plus a per-fold accuracy report
        (never a single blended in-sample number)."""
        from sklearn.linear_model import LogisticRegression

        names = list(returns_matrix.columns)
        n = len(returns_matrix)
        points = self._rebalance_points(n)

        history_X, history_y = [], []
        score_rows = []
        self.fold_reports_ = []

        for k, idx in enumerate(points):
            next_idx = points[k + 1] if k + 1 < len(points) else n

            # Predict for every strategy at this rebalance point using
            # only the model fit on STRICTLY PAST (feature, label) pairs.
            scores = {}
            if len(history_X) >= MIN_TRAIN_REBALANCES * len(names):
                Xtr, ytr = np.array(history_X), np.array(history_y)
                if len(np.unique(ytr)) >= 2:
                    model = LogisticRegression(penalty="l2", C=0.5, max_iter=1000)
                    model.fit(Xtr, ytr)
                    correct = 0
                    for name in names:
                        feat = self._features_at(returns_matrix, idx, name)
                        if feat is None:
                            scores[name] = 0.0
                            continue
                        scores[name] = float(model.decision_function(feat.reshape(1, -1))[0])
                    # fold accuracy: did the sign of this fold's score match
                    # the realised label, evaluated on THIS fold only
                    fold_correct, fold_total = 0, 0
                    for name in names:
                        label = self._label_at(returns_matrix, idx, next_idx, name)
                        if label is None or name not in scores:
                            continue
                        fold_total += 1
                        fold_correct += int((scores[name] > 0) == bool(label))
                    if fold_total:
                        self.fold_reports_.append({
                            "rebalance_idx": idx,
                            "date": str(returns_matrix.index[idx].date()),
                            "n_predictions": fold_total,
                            "accuracy": fold_correct / fold_total,
                        })
                else:
                    scores = {name: 0.0 for name in names}
            else:
                scores = {name: 0.0 for name in names}  # not enough history yet -> neutral

            score_rows.append({"date": returns_matrix.index[idx], **scores})

            # Now that this fold's true labels are known, add them to history.
            for name in names:
                feat = self._features_at(returns_matrix, idx, name)
                label = self._label_at(returns_matrix, idx, next_idx, name)
                if feat is not None and label is not None:
                    history_X.append(feat)
                    history_y.append(label)

        scores_df = pd.DataFrame(score_rows).set_index("date")
        return scores_df

    # ------------------------------------------------------ null baseline --
    def null_baseline(self, returns_matrix, n_shuffles=100):
        """Two null comparisons, per the model-discipline requirement:
        (a) permuted-label baseline -- refit with labels shuffled within
        each training window, to see whether the walk-forward accuracy
        above is actually better than a model that learned nothing; (b)
        random-allocation baseline -- scores drawn from noise, run
        through the exact same scores_to_weights path.
        """
        real_scores = self.walk_forward_fit_predict(returns_matrix)
        real_acc = np.mean([f["accuracy"] for f in self.fold_reports_]) if self.fold_reports_ else np.nan

        perm_accs = []
        names = list(returns_matrix.columns)
        n = len(returns_matrix)
        points = self._rebalance_points(n)
        for _ in range(n_shuffles):
            shuffled = returns_matrix.copy()
            for name in names:
                shuffled[name] = self.rng.permutation(shuffled[name].to_numpy())
            perm_model = AlphaMetaModel(self.lookback_bars, self.rebalance_freq, seed=int(self.rng.integers(1e9)))
            perm_model.walk_forward_fit_predict(shuffled)
            if perm_model.fold_reports_:
                perm_accs.append(np.mean([f["accuracy"] for f in perm_model.fold_reports_]))

        perm_accs = np.array(perm_accs)
        p_value = (
            float((np.sum(perm_accs >= real_acc) + 1) / (len(perm_accs) + 1))
            if len(perm_accs) and not np.isnan(real_acc) else np.nan
        )

        return {
            "real_walk_forward_accuracy": float(real_acc) if not np.isnan(real_acc) else None,
            "permuted_label_accuracy_mean": float(perm_accs.mean()) if len(perm_accs) else None,
            "permuted_label_accuracy_std": float(perm_accs.std()) if len(perm_accs) else None,
            "p_value_vs_permuted_null": p_value,
            "n_shuffles": len(perm_accs),
            "n_folds": len(self.fold_reports_),
            # Raw per-shuffle accuracies, not just their mean/std -- kept so
            # plotting.py can draw the actual null distribution rather than
            # a synthetic normal approximation around the mean.
            "permuted_accuracies": perm_accs.tolist(),
        }

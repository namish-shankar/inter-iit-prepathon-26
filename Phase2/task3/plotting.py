"""
plotting.py -- turns Task 3's factor-model and allocation results into
PNG charts under results/plots/. Purely a reporting convenience layered
on top of FactorModel / FinalEvaluator's output; nothing here changes a
single number in task3_report.json, and every function degrades to
"skip this plot" on a missing/empty input rather than raising.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _new_fig(figsize=(10, 5)):
    return plt.subplots(figsize=figsize)


def plot_allocation_equity_curves(static_methods, dynamic_method, out_path):
    """Equity curve of all 5 allocation methods on one axis -- the
    single picture that makes 'mean_variance_shrinkage edges out
    best_individual, dynamic tracks equal_weight' visible at a glance,
    rather than only readable off a table of Sharpe ratios."""
    fig, ax = _new_fig((11, 6))
    plotted = False
    for name, res in static_methods.items():
        r = res.get("returns")
        if r is None or len(r) == 0:
            continue
        eq = (1.0 + r.fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label=name, linewidth=1.2)
        plotted = True

    r = dynamic_method.get("returns")
    if r is not None and len(r) > 0:
        eq = (1.0 + r.fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label="dynamic_meta_model", color="#da3633",
                 linewidth=1.4, linestyle="--")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axhline(1.0, color="#888888", linewidth=0.5, linestyle=":")
    ax.set_title("Equity curves: all 5 allocation methods")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sharpe_comparison(comparison_table, out_path):
    """Sharpe across all 5 methods on one bar chart -- the dynamic
    method is highlighted, not hidden, since the honest result here is
    that it does NOT beat the naive baselines on this run."""
    names = list(comparison_table.keys())
    if not names:
        return
    sharpes = [comparison_table[n]["sharpe_ratio"] for n in names]
    colors = ["#1f6feb"] * len(names)
    if "dynamic_meta_model" in names:
        colors[names.index("dynamic_meta_model")] = "#da3633"

    fig, ax = _new_fig((9, 5))
    ax.bar(names, sharpes, color=colors)
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe ratio by allocation method (red = dynamic/learned)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_static_weights(static_methods, out_path):
    """Grouped bar chart: how each of the 4 static methods actually
    splits capital across strategies -- makes best_individual's
    all-or-nothing allocation visually distinct from the others'
    diversified split."""
    methods = list(static_methods.keys())
    strategies = sorted({name for res in static_methods.values() for name in res["weights"]})
    if not strategies:
        return

    x = np.arange(len(strategies))
    width = 0.8 / max(len(methods), 1)
    fig, ax = _new_fig((9, 5))
    for i, method in enumerate(methods):
        weights = [static_methods[method]["weights"].get(s, 0.0) for s in strategies]
        ax.bar(x + i * width, weights, width=width, label=method)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Weight")
    ax.set_title("Static allocation weights by method")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dynamic_weights_over_time(weights_daily, out_path):
    """Stacked area of the dynamic allocator's daily weights -- shows
    directly why its return matches equal_weight's: the 5% no-trade
    band keeps this close to a flat 1/3 split for most of the sample,
    rather than leaving that claim as a sentence in the README."""
    if weights_daily is None or len(weights_daily) == 0:
        return
    fig, ax = _new_fig((11, 5))
    ax.stackplot(weights_daily.index, *[weights_daily[c].values for c in weights_daily.columns],
                 labels=weights_daily.columns, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("Dynamic (learned) allocator: daily weights over time")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_factor_alpha_beta(factor_report, out_path):
    """Two-panel bar chart: alpha t-stat and beta per strategy against
    the buy-and-hold factor. Puts baseline_pb01's near-pure-beta profile
    and alpha_02's near-zero-beta profile side by side, instead of only
    in a table."""
    names = list(factor_report.keys())
    if not names:
        return
    alphas_t = [factor_report[n]["alpha_t_stat"] for n in names]
    betas = [factor_report[n]["beta"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(names, alphas_t, color="#2ea043")
    ax1.axhline(0, color="#444444", linewidth=0.7)
    ax1.axhline(1.96, color="#888888", linestyle="--", linewidth=0.7)
    ax1.axhline(-1.96, color="#888888", linestyle="--", linewidth=0.7)
    ax1.set_title("Alpha t-stat vs. buy-and-hold (dashed = |t|=1.96)")
    plt.setp(ax1.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    ax2.bar(names, betas, color="#1f6feb")
    ax2.axhline(0, color="#444444", linewidth=0.7)
    ax2.set_title("Beta vs. buy-and-hold")
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    fig.suptitle("Factor decomposition: each strategy vs. buy-and-hold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_meta_model_null_baseline(null_baseline, out_path):
    """Real walk-forward direction accuracy against the full
    permuted-label null distribution -- the picture behind the p=0.198
    headline, so 'plausible but not significant' is visibly true rather
    than just asserted."""
    if null_baseline is None:
        return
    perm_accs = null_baseline.get("permuted_accuracies")
    real_acc = null_baseline.get("real_walk_forward_accuracy")
    if not perm_accs or real_acc is None:
        return

    fig, ax = _new_fig((8, 5))
    ax.hist(perm_accs, bins=20, color="#8b949e", alpha=0.85, label="permuted-label null")
    ax.axvline(real_acc, color="#da3633", linewidth=1.8, label=f"real accuracy = {real_acc:.3f}")
    ax.axvline(0.5, color="#444444", linewidth=1.0, linestyle=":", label="no-information (0.5)")
    p_val = null_baseline.get("p_value_vs_permuted_null")
    p_str = f"{p_val:.3f}" if p_val is not None else "n/a"
    ax.set_title(f"Meta-model walk-forward accuracy vs. permuted-label null (p={p_str})")
    ax.set_xlabel("Direction accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all(static_methods, dynamic_method, comparison_table, factor_report, out_dir):
    """Entry point called from main.py. Returns the list of PNG
    filenames actually written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_allocation_equity_curves(static_methods, dynamic_method, out_dir / "allocation_equity_curves.png")
    plot_sharpe_comparison(comparison_table, out_dir / "allocation_sharpe_comparison.png")
    plot_static_weights(static_methods, out_dir / "static_weights.png")
    plot_dynamic_weights_over_time(dynamic_method.get("weights_daily"), out_dir / "dynamic_weights_over_time.png")
    plot_factor_alpha_beta(factor_report, out_dir / "factor_alpha_beta.png")
    plot_meta_model_null_baseline(dynamic_method.get("null_baseline"), out_dir / "meta_model_null_baseline.png")
    return sorted(p.name for p in out_dir.glob("*.png"))

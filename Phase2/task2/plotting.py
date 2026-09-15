"""
plotting.py -- turns Task 2's alpha-research results into PNG charts
under results/plots/. Purely a reporting convenience layered on top of
AlphaResearch's output; nothing here changes a single number in
task2_report.json, and every function degrades to "skip this plot"
rather than raising on a degenerate input (e.g. a strategy with too few
observations to permutation-test).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statistics import StatisticalTester


def _new_fig(figsize=(10, 5)):
    return plt.subplots(figsize=figsize)


def plot_equity_curves(strategy_results, price_data, out_path):
    """All strategies' equity curves on one axis, against buy-and-hold --
    the first honest look at whether any of them are doing something a
    passive position wouldn't."""
    fig, ax = _new_fig((11, 6))
    plotted = False
    for name, res in strategy_results.items():
        eq = res.get("equity_curve")
        if eq is None or len(eq) == 0:
            continue
        ax.plot(eq.index, eq.values, label=name, linewidth=1.1)
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    buyhold_ret = price_data["open"].pct_change().fillna(0.0)
    buyhold_equity = (1.0 + buyhold_ret).cumprod()
    ax.plot(buyhold_equity.index, buyhold_equity.values, label="buy_and_hold",
             color="#444444", linestyle="--", linewidth=1.0)

    ax.axhline(1.0, color="#888888", linewidth=0.5, linestyle=":")
    ax.set_title("Equity curves: all strategies vs. buy-and-hold")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sharpe_comparison(perf_reports, stats, out_path):
    """Sharpe per strategy, colour-coded by whether it actually survives
    the Benjamini-Hochberg correction -- a good-looking bar in grey is
    exactly the "looked good, wasn't significant" case this project is
    built to catch and report honestly."""
    names = list(perf_reports.keys())
    if not names:
        return
    sharpes = [perf_reports[n]["sharpe_ratio"] for n in names]
    significant = [stats.get(n, {}).get("significant_after_bh_correction", False) for n in names]
    colors = ["#2ea043" if s else "#8b949e" for s in significant]

    fig, ax = _new_fig((9, 5))
    ax.bar(names, sharpes, color=colors)
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe ratio per strategy (green = significant after BH correction)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_signal_ic_heatmap(signal_table, out_path):
    """Information coefficient of every raw signal against the forward
    price difference, across horizons -- the reference grid every
    strategy hypothesis in this project was checked against."""
    scored = signal_table.dropna(subset=["ic"])
    if scored.empty:
        return
    pivot = scored.pivot(index="signal", columns="horizon", values="ic")
    fig, ax = _new_fig((7, 8))
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Horizon (bars)")
    ax.set_title("Information coefficient: signal vs. forward price difference")
    fig.colorbar(im, ax=ax, label="IC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(returns_dict, out_path):
    """Pairwise correlation between strategies' net-return series -- the
    quick visual companion to OrthogonalityAnalyzer's pivoted-QR rank:
    high off-diagonal correlation is one strategy wearing another's
    clothes."""
    matrix = pd.DataFrame(returns_dict).dropna(how="any")
    if matrix.empty or matrix.shape[1] < 2:
        return
    corr = matrix.corr()
    fig, ax = _new_fig((7, 6))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(val) > 0.5 else "black")
    ax.set_title("Strategy return correlation matrix")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_permutation_tests(strategy_results, returns_dict, price_data, out_path, n_permutations=500):
    """Small multiples: the null distribution of Sharpe against the
    observed Sharpe, one panel per strategy.

    Reruns StatisticalTester.permutation_test with the project's fixed
    seed, so the p-value drawn on each panel matches task2_report.json
    exactly -- this is the picture behind that number, not a different
    computation. The null is built by circularly shifting each
    strategy's POSITION relative to the fixed, real market data and
    re-running it through the actual ExecutionEngine/Portfolio -- NOT by
    shuffling the realised return series, which is mean/std-invariant to
    any reordering and would draw a single spike instead of a real
    distribution (see statistics.py's permutation_test docstring)."""
    import config
    price_cols = [c for c in config.PRICE_COLUMNS if c in price_data.columns]
    market_data = price_data[price_cols]

    names = [
        n for n in strategy_results
        if n in returns_dict and returns_dict[n] is not None and len(returns_dict[n].dropna()) >= 20
    ]
    if not names:
        return
    tester = StatisticalTester()
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        position = strategy_results[name]["executions"]["position"]
        perm = tester.permutation_test(returns_dict[name], position, market_data, n_permutations=n_permutations)
        null_stats = np.array(perm.get("null_stats", []))
        null_stats = null_stats[np.isfinite(null_stats)]
        observed = perm.get("observed", np.nan)
        p_value = perm.get("p_value", np.nan)

        # A strategy that barely trades can still produce a near-constant
        # null -- guard the bin count so that degenerate case draws a
        # (flat) histogram instead of raising.
        spread = float(np.ptp(null_stats)) if len(null_stats) else 0.0
        n_bins = 30 if spread > 1e-9 else 1
        if len(null_stats):
            ax.hist(null_stats, bins=n_bins, color="#8b949e", alpha=0.85, label="null (shifted position)")
        if np.isfinite(observed):
            ax.axvline(observed, color="#da3633", linewidth=1.5, label=f"observed={observed:.2f}")
        p_str = f"{p_value:.2f}" if np.isfinite(p_value) else "n/a"
        ax.set_title(f"{name}\np={p_str}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper left")

    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Permutation-test null distributions (position shifted vs. fixed market)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all(strategy_results, returns_dict, perf_reports, stats, signal_table, price_data, out_dir):
    """Entry point called from main.py. Returns the list of PNG
    filenames actually written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_equity_curves(strategy_results, price_data, out_dir / "equity_curves.png")
    plot_sharpe_comparison(perf_reports, stats, out_dir / "sharpe_comparison.png")
    plot_signal_ic_heatmap(signal_table, out_dir / "signal_ic_heatmap.png")
    plot_correlation_heatmap(returns_dict, out_dir / "strategy_correlation_heatmap.png")
    plot_permutation_tests(strategy_results, returns_dict, price_data, out_dir / "permutation_tests.png")
    return sorted(p.name for p in out_dir.glob("*.png"))

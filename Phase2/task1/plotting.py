"""
plotting.py -- turns Task 1's backtest results into a handful of PNG
charts under results/plots/. Purely a reporting convenience: nothing
here feeds back into the backtest, and a missing/empty series degrades
to "skip this plot" rather than raising, so a quiet strategy (e.g. zero
trades) can never crash main.py at the very end of a run.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _new_fig(figsize=(10, 5)):
    return plt.subplots(figsize=figsize)


def plot_equity_vs_buyhold(equity_curve, price_data, out_path):
    """Strategy equity vs. a buy-and-hold-the-underlying benchmark, both
    starting at 1.0, on the same axis -- the first question any
    backtest result should answer: did this beat doing nothing clever?"""
    if equity_curve is None or len(equity_curve) == 0:
        return
    buyhold_ret = price_data["open"].pct_change().reindex(equity_curve.index).fillna(0.0)
    buyhold_equity = (1.0 + buyhold_ret).cumprod()

    fig, ax = _new_fig()
    ax.plot(equity_curve.index, equity_curve.values, label="Strategy", color="#1f6feb", linewidth=1.5)
    ax.plot(buyhold_equity.index, buyhold_equity.values, label="Buy & hold", color="#8b949e",
             linewidth=1.2, linestyle="--")
    ax.axhline(1.0, color="#444444", linewidth=0.6, linestyle=":")
    ax.set_title("Equity curve: strategy vs. buy-and-hold")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdown(equity_curve, out_path):
    """Underwater curve -- how far below its own prior peak the equity
    curve sits at every point in time, not just the single worst number."""
    if equity_curve is None or len(equity_curve) == 0:
        return
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak

    fig, ax = _new_fig()
    ax.fill_between(dd.index, dd.values * 100, 0, color="#da3633", alpha=0.35)
    ax.plot(dd.index, dd.values * 100, color="#da3633", linewidth=1.0)
    ax.set_title("Drawdown from prior equity peak")
    ax.set_ylabel("Drawdown (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_position_timeline(positions, price_data, out_path):
    """Position (-1/0/+1) as a step series, with close price on a second
    axis for context -- makes it visible at a glance whether a strategy
    was actually doing anything, and when."""
    if positions is None or len(positions) == 0:
        return
    fig, ax1 = _new_fig()
    ax2 = ax1.twinx()

    close = price_data["close"].reindex(positions.index)
    ax2.plot(close.index, close.values, color="#c9d1d9", linewidth=0.8, zorder=1)
    ax2.set_ylabel("Close price", color="#6e7681")

    ax1.step(positions.index, positions.values, where="post", color="#1f6feb", linewidth=1.2, zorder=2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_yticks([-1, 0, 1])
    ax1.set_ylabel("Position")
    ax1.set_title("Position over time (against close price)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all(data, results, out_dir):
    """Entry point called from main.py. `data`: merged price+signal
    frame. `results`: the dict returned by Backtester.run(). Returns the
    list of PNG filenames actually written, so main.py can report what
    landed rather than assuming."""
    out_dir.mkdir(parents=True, exist_ok=True)
    equity = results.get("equity_curve")
    executions = results.get("executions")
    positions = executions["position"] if executions is not None else None

    plot_equity_vs_buyhold(equity, data, out_dir / "equity_vs_buyhold.png")
    plot_drawdown(equity, out_dir / "drawdown.png")
    plot_position_timeline(positions, data, out_dir / "position_timeline.png")

    return sorted(p.name for p in out_dir.glob("*.png"))

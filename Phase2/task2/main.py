"""
Task 2 entry point: everything Task 1 does, plus signal characterisation,
five distinct alpha strategies, significance testing, robustness checks,
and an orthogonality analysis across the set.

Run with:  python main.py
"""

import json
import random

import numpy as np
import pandas as pd

import config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from alpha_research import AlphaResearch

from strategies.baseline_strategy import BaselineStrategy
from strategies.alpha_01 import Alpha01TrendAgreement
from strategies.alpha_02 import Alpha02BandReversion
from strategies.alpha_03 import Alpha03VolumeBreakout
from strategies.alpha_04 import Alpha04SqueezeExpansion
from strategies.alpha_05 import Alpha05LearnedComposite
from plotting import plot_all as plot_task2_results


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_merged_data():
    loader = DataLoader()
    price = DataCleaner(kind="price").clean(loader.load_price())
    signals = DataCleaner(kind="signals").clean(loader.load_signals())
    return price.join(signals, how="inner")


def balance_report(strategy_reports):
    """Flags a strategy whose metric profile looks spiky (great on one
    axis, broken on another) rather than good-ish across the board --
    used by AlphaResearch.select_final_set as an extra screen, and
    printed so the flag is visible even for strategies kept anyway."""
    out = {}
    for name, r in strategy_reports.items():
        flags = []
        if r["sharpe_ratio"] > 0.3 and r["max_drawdown"] < -0.30:
            flags.append("decent Sharpe but drawdown > 30%")
        if r["num_round_trips"] < 5:
            flags.append("fewer than 5 round trips -- metrics not statistically meaningful")
        calmar = r["calmar_ratio"]
        if calmar is not None and not (isinstance(calmar, float) and np.isnan(calmar)) and abs(calmar) > 5:
            flags.append("calmar ratio implausibly large -- likely a near-zero-drawdown artefact")
        out[name] = {"balanced": len(flags) == 0, "flags": flags}
    return out


def run():
    set_seed()
    print("=" * 70)
    print("TASK 2 -- Alpha Research")
    print("=" * 70)

    data = load_merged_data()
    print(f"\nMerged dataset: {len(data)} rows, {data.index.min().date()} -> {data.index.max().date()}")

    research = AlphaResearch(data)

    print("\n--- Signal characterisation (hit-rate & IC vs forward price diff) ---")
    signal_table = research.characterize_signals()
    signal_table.to_csv(config.RESULTS_DIR / "task2_signal_characterization.csv", index=False)
    scored = signal_table.dropna(subset=["ic"]).copy()
    scored["abs_ic"] = scored["ic"].abs()
    best_idx = scored.groupby("horizon")["abs_ic"].idxmax()
    best_by_horizon = scored.loc[best_idx].drop(columns="abs_ic")
    print(best_by_horizon.to_string(index=False))

    strategies = {
        "baseline_pb01": BaselineStrategy(),
        "alpha_01_trend_agreement": Alpha01TrendAgreement(),
        "alpha_02_band_reversion": Alpha02BandReversion(),
        "alpha_03_volume_breakout": Alpha03VolumeBreakout(),
        "alpha_04_squeeze_expansion": Alpha04SqueezeExpansion(),
        "alpha_05_learned_composite": Alpha05LearnedComposite(),
    }

    print("\n--- Running all strategies through the shared Backtester ---")
    research.run_all(list(strategies.values()))

    perf_reports = {name: res["report"] for name, res in research.strategy_results.items()}
    print("\n--- Performance report per strategy ---")
    print(pd.DataFrame(perf_reports).T.round(4).to_string())

    balance = balance_report(perf_reports)
    print("\n--- Balance report (good-ish everywhere check) ---")
    for name, b in balance.items():
        print(f"  {name}: balanced={b['balanced']}  {b['flags']}")

    print("\n--- Statistical significance ---")
    stats = research.run_statistics(horizon=5)
    print(pd.DataFrame(stats).T.round(4).to_string())

    print("\n--- Robustness (sub-periods, cost sensitivity, regime) ---")
    robustness = research.run_robustness(strategies)

    print("\n--- Orthogonality analysis ---")
    orthogonality = research.run_orthogonality()
    print(f"Effective rank (pivoted QR, {orthogonality['pivoted_qr']['n_strategies']} strategies): "
          f"{orthogonality['pivoted_qr']['effective_rank']}")
    print(f"Pivot order (most to least information-bearing): {orthogonality['pivoted_qr']['pivot_order']}")
    print(f"Rolling top-pivot stability: {orthogonality['rolling_stability'].get('top_pivot_stability')}")

    print("\n--- Final selection ---")
    selection = research.select_final_set(stats, orthogonality, balance)
    print(json.dumps(selection, indent=2, default=str))

    # ------------------------------------------------------------- save --
    full_report = {
        "n_rows": len(data),
        "date_range": [str(data.index.min().date()), str(data.index.max().date())],
        "performance": perf_reports,
        "balance_report": balance,
        "statistics": stats,
        "robustness": robustness,
        "orthogonality": {k: v for k, v in orthogonality.items() if k != "rolling_stability"},
        "rolling_stability_summary": {
            k: v for k, v in orthogonality["rolling_stability"].items() if k != "windows"
        },
        "selection": selection,
    }
    with open(config.RESULTS_DIR / "task2_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nSaved outputs to {config.RESULTS_DIR}/")

    plot_files = plot_task2_results(
        research.strategy_results, research.returns, perf_reports, stats, signal_table, data,
        config.RESULTS_DIR / "plots",
    )
    print(f"Saved plots to {config.RESULTS_DIR}/plots/: {plot_files}")


if __name__ == "__main__":
    run()

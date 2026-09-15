"""
Task 1 entry point: data -> clean -> merge -> baseline backtest -> report.

Run with:  python main.py
Everything is read/written relative to this file, so the project runs
unmodified from a fresh clone.
"""

import json
import random

import numpy as np
import pandas as pd

import config
from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engine import FeatureEngine
from backtester import Backtester
from strategies.baseline_strategy import BaselineStrategy
from plotting import plot_all as plot_task1_results


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_and_clean():
    loader = DataLoader()
    price_raw = loader.load_price()
    signals_raw = loader.load_signals()

    price_cleaner = DataCleaner(kind="price")
    price = price_cleaner.clean(price_raw)
    price_report = price_cleaner.get_report()

    signal_cleaner = DataCleaner(kind="signals")
    signals = signal_cleaner.clean(signals_raw)
    signal_report = signal_cleaner.get_report()

    return price, price_report, signals, signal_report


def merge(price, signals):
    """Inner join on date -- rows dropped in either direction are logged,
    not silently absorbed, per the 'check formatting properly' rule."""
    before_price, before_signals = len(price), len(signals)
    merged = price.join(signals, how="inner")
    return merged, {
        "price_rows_before_join": before_price,
        "signal_rows_before_join": before_signals,
        "merged_rows": len(merged),
        "price_rows_dropped_by_join": before_price - len(merged),
        "signal_rows_dropped_by_join": before_signals - len(merged),
    }


def run():
    set_seed()
    print("=" * 70)
    print("TASK 1 -- Data & Backtesting Infrastructure")
    print("=" * 70)

    price, price_report, signals, signal_report = load_and_clean()
    print("\n--- Data-quality report: price ---")
    print(json.dumps(price_report, indent=2, default=str))
    print("\n--- Data-quality report: signals ---")
    print(json.dumps(signal_report, indent=2, default=str))

    data, join_report = merge(price, signals)
    print("\n--- Join report ---")
    print(json.dumps(join_report, indent=2, default=str))

    fe = FeatureEngine()
    structural_ok = fe.validate_no_lookahead(data)  # NaN-pattern sanity check
    print(f"\nStructural NaN-pattern check (warm-up NaNs form a clean prefix): {structural_ok}")

    strategy = BaselineStrategy()
    backtester = Backtester()
    results = backtester.run(data, strategy, check_lookahead=True)

    print(f"\nLook-ahead check passed (truncation + perturbation): "
          f"{results['lookahead_check_passed']}")

    print("\n--- Baseline strategy performance ---")
    print(json.dumps(results["report"], indent=2, default=str))

    # ------------------------------------------------------------ save --
    results["equity_curve"].to_csv(config.RESULTS_DIR / "task1_equity_curve.csv")
    results["trade_log"].to_csv(config.RESULTS_DIR / "task1_trade_log.csv", index=False)
    with open(config.RESULTS_DIR / "task1_report.json", "w") as f:
        json.dump(
            {
                "price_data_quality": price_report,
                "signal_data_quality": signal_report,
                "join_report": join_report,
                "lookahead_check_passed": results["lookahead_check_passed"],
                "strategy": results["strategy"],
                "performance": results["report"],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved outputs to {config.RESULTS_DIR}/")

    plot_files = plot_task1_results(data, results, config.RESULTS_DIR / "plots")
    print(f"Saved plots to {config.RESULTS_DIR}/plots/: {plot_files}")


if __name__ == "__main__":
    run()

"""
Task 3 entry point: everything Task 1 + Task 2 do, plus factor
attribution and five portfolio-allocation methods evaluated on one
common harness, including a walk-forward-validated learned allocator
checked against a null baseline.

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
from factor_model import FactorModel
from final_evaluation import FinalEvaluator

from strategies.baseline_strategy import BaselineStrategy
from strategies.alpha_01 import Alpha01TrendAgreement
from strategies.alpha_02 import Alpha02BandReversion
from strategies.alpha_03 import Alpha03VolumeBreakout
from strategies.alpha_04 import Alpha04SqueezeExpansion
from strategies.alpha_05 import Alpha05LearnedComposite
from plotting import plot_all as plot_task3_results


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_merged_data():
    loader = DataLoader()
    price = DataCleaner(kind="price").clean(loader.load_price())
    signals = DataCleaner(kind="signals").clean(loader.load_signals())
    return price.join(signals, how="inner")


def run():
    set_seed()
    print("=" * 70)
    print("TASK 3 -- Portfolio Construction & Dynamic Allocation")
    print("=" * 70)

    data = load_merged_data()
    print(f"\nMerged dataset: {len(data)} rows, {data.index.min().date()} -> {data.index.max().date()}")

    # --------------------------------------------------- Task 1+2 rerun --
    research = AlphaResearch(data)
    strategies = {
        "baseline_pb01": BaselineStrategy(),
        "alpha_01_trend_agreement": Alpha01TrendAgreement(),
        "alpha_02_band_reversion": Alpha02BandReversion(),
        "alpha_03_volume_breakout": Alpha03VolumeBreakout(),
        "alpha_04_squeeze_expansion": Alpha04SqueezeExpansion(),
        "alpha_05_learned_composite": Alpha05LearnedComposite(),
    }
    research.run_all(list(strategies.values()))
    stats = research.run_statistics(horizon=5)
    orthogonality = research.run_orthogonality()

    perf_reports = {name: res["report"] for name, res in research.strategy_results.items()}
    balance = {}  # not needed again for selection here; reuse stats-only screen
    for name in perf_reports:
        balance[name] = {"balanced": True}
    selection = research.select_final_set(stats, orthogonality, balance)

    combined_set = selection["selected"] or selection["practical_set_for_task3"]
    print(f"\nStrategies carried into Task 3: {combined_set}")
    if selection["practical_set_caveat"]:
        print(f"CAVEAT (carried forward from Task 2): {selection['practical_set_caveat']}")

    returns_dict = {name: research.returns[name] for name in combined_set}

    # ------------------------------------------------------- factor model --
    print("\n--- Factor model: alpha/beta vs. buy-and-hold ---")
    fm = FactorModel()
    factor_returns = fm.compute_factor_returns(data)
    factor_report = fm.fit(returns_dict, factor_returns)
    print(pd.DataFrame(factor_report).T.round(4).to_string())

    # --------------------------------------------------------- allocation --
    print("\n--- Evaluating all 5 allocation methods on a common harness ---")
    evaluator = FinalEvaluator(returns_dict, caveat=selection["practical_set_caveat"])
    results = evaluator.evaluate_all()

    print("\n--- Comparison table (all 5 methods) ---")
    print(pd.DataFrame(results["comparison_table"]).T.round(4).to_string())

    print("\n--- Static method weights ---")
    for method, res in results["static_methods"].items():
        print(f"  {method}: {res['weights']}")

    print("\n--- Dynamic (learned) method: null-baseline check ---")
    print(json.dumps(results["dynamic_method"]["null_baseline"], indent=2, default=str))
    print(f"Average weights over time: {results['dynamic_method']['weights_avg_over_time']}")

    # ------------------------------------------------------------- save --
    full_report = {
        "strategies_combined": combined_set,
        "practical_set_caveat": selection["practical_set_caveat"],
        "factor_model": factor_report,
        "allocation_comparison": results["comparison_table"],
        "static_method_weights": {m: r["weights"] for m, r in results["static_methods"].items()},
        "dynamic_method": {
            "weights_avg_over_time": results["dynamic_method"]["weights_avg_over_time"],
            "performance": results["dynamic_method"]["performance"],
            "null_baseline": results["dynamic_method"]["null_baseline"],
        },
    }
    with open(config.RESULTS_DIR / "task3_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\nSaved outputs to {config.RESULTS_DIR}/")

    plot_files = plot_task3_results(
        results["static_methods"], results["dynamic_method"], results["comparison_table"],
        factor_report, config.RESULTS_DIR / "plots",
    )
    print(f"Saved plots to {config.RESULTS_DIR}/plots/: {plot_files}")


if __name__ == "__main__":
    run()

# Multi-Alpha Lab – Phase 2: Project Structure

Three self-contained, independently submittable folders, each a complete project:

```
Phase2/
  task1/
    ├── config.py              central constants (all fixed: costs, seed, assumptions)
    ├── data_loader.py         DataLoader (schema validation, no transformation)
    ├── data_cleaner.py        DataCleaner (date parsing, dedup, NaN handling)
    ├── feature_engine.py      FeatureEngine (derived columns, lookahead checks)
    ├── strategy.py            BaseStrategy (abstract contract)
    ├── execution_engine.py    ExecutionEngine (fills at open, cost + slippage)
    ├── portfolio.py           Portfolio (position tracking, equity curve)
    ├── performance.py         PerformanceAnalyzer (Sharpe, Sortino, drawdown, etc.)
    ├── backtester.py          Backtester (orchestrator, enforces t-1 cutoff)
    ├── plotting.py            plot generation (equity, drawdown, position timeline)
    ├── strategies/
    │   └── baseline_strategy.py
    ├── data/                  (price_train.csv, signals_train.csv)
    └── results/               (outputs: equity curve, trade log, JSON, plots/*.png)
  
  task2/
    ├── [all of task1 above]
    ├── statistics.py          StatisticalTester (permutation, Newey-West, BH correction)
    ├── robustness.py          RobustnessTester (sub-periods, cost sensitivity, regime)
    ├── orthogonality.py       OrthogonalityAnalyzer (pivoted QR, residual alpha)
    ├── alpha_research.py      AlphaResearch (orchestrator)
    ├── plotting.py            (extends task1's: equity curves, IC heatmap, perm tests, etc.)
    ├── strategies/
    │   ├── baseline_strategy.py
    │   ├── alpha_01.py   Trend agreement
    │   ├── alpha_02.py   Band-extreme mean reversion
    │   ├── alpha_03.py   Volume-confirmed breakout
    │   ├── alpha_04.py   Squeeze-expansion timing
    │   └── alpha_05.py   Learned composite (logistic regression)
    └── results/               (plus: signal characterization, plots/*.png)

  task3/
    ├── [all of task1 + task2 above]
    ├── factor_model.py        FactorModel (alpha/beta decomposition)
    ├── portfolio_optimizer.py PortfolioOptimizer (5 allocation methods)
    ├── meta_model.py          AlphaMetaModel (walk-forward learned scoring)
    ├── dynamic_allocator.py   DynamicAllocator (scores → weights, no-trade band)
    ├── final_evaluation.py    FinalEvaluator (all 5 methods on one harness)
    ├── plotting.py            (extends task2's: allocation curves, weights, factor plot, etc.)
    └── results/               (plus: plots/*.png for allocation methods)

  REPORT.md                    technical report (data, backtester, alphas, results)
  README.md                    this file (structure and interrelation)
```

## Interrelation

**Task 1** (backtester foundation):
- `DataLoader` → `DataCleaner` → merged data → `Backtester` calls `BaselineStrategy` via `ExecutionEngine` → `Portfolio` → `PerformanceAnalyzer`
- `FeatureEngine` validates no lookahead and builds derived columns
- Results: `results/task1_report.json`, equity curve CSV, trade log CSV, 3 PNG plots

**Task 2** (alpha research layer):
- Takes Task 1's clean data and backtester
- `AlphaResearch.characterize_signals()` runs every signal against forward-difference labels (all horizons)
- `AlphaResearch.run_all(strategies)` runs each alpha through the same backtester, collecting returns
- `StatisticalTester`: permutation test, Newey-West t-stat, BH correction on the set
- `RobustnessTester`: sub-period Sharpe, cost sensitivity, regime conditioning
- `OrthogonalityAnalyzer`: pivoted QR (rank), residual alpha (independence check), rolling stability
- `AlphaResearch.select_final_set()` applies selection criteria: significant after BH correction, not redundant with others, balanced metric profile
- **Result**: no strategy clears the bar; `practical_set_for_task3` (top-3 by Sharpe, caveated as unvalidated) carried forward
- Output: `results/task2_report.json`, signal characterization CSV, 5 PNG plots

**Task 3** (portfolio construction layer):
- Takes Task 2's return streams (the 3-strategy practical set)
- `FactorModel`: regresses each strategy's return on the asset's own buy-and-hold, separates alpha (timing) vs. beta (drift correlation)
- `PortfolioOptimizer`: computes static weights via 4 methods (best_individual, equal_weight, inverse_volatility, mean_variance_shrinkage)
- `AlphaMetaModel` + `DynamicAllocator`: walk-forward learned allocation (scores → weights, 5% no-trade band)
- `FinalEvaluator`: runs all 5 methods on the same return matrix, one comparable harness
- `AlphaMetaModel.null_baseline()`: checks learned model against permuted-label null (52.3% vs. 48.2%, p=0.20, not significant)
- **Result**: mean_variance_shrinkage best among static (Sharpe 0.189); dynamic doesn't beat baselines
- Output: `results/task3_report.json`, 6 PNG plots

## Key Invariants Enforced at the Code Level

1. **No lookahead**: `Backtester.generate_signals()` shifts signal columns by SIGNAL_LAG before strategy sees them
2. **Fill at open**: `ExecutionEngine.execute()` fills at candle `t`'s open, not close
3. **Cost consistent**: 0.05%/side transaction cost + slippage, applied uniformly
4. **Open-to-open accounting**: return = `position(t-1) × (open(t)/open(t-1) – 1)`; cost charged at t
5. **Price never features**: strategies read only signals, never raw OHLC (FeatureEngine enforces at interface)
6. **Price difference for labels**: all research uses `close[t+h] – close[t]` (move), not `close[t+h]` (level)


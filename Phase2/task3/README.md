# Task 3 -- Portfolio Construction & Dynamic Allocation



## New files

```
factor_model.py        FactorModel
portfolio_optimizer.py   PortfolioOptimizer (methods 1-4, static)
meta_model.py             AlphaMetaModel (method 5's scoring engine, walk-forward)
dynamic_allocator.py      DynamicAllocator (method 5's weight/no-trade-band logic)
final_evaluation.py       FinalEvaluator (runs all 5 on one harness)
plotting.py               chart generation (see "Plots" below)
```

## Plots (`plotting.py`, `results/plots/`)

Generated fresh by every `main.py` run (6 charts):

| Plot | Expected content | Why |
|---|---|---|
| `allocation_equity_curves.png` | 5 overlapping lines (blue best_individual, orange equal_weight, green inverse_volatility, red mean_variance_shrinkage solid, red dynamic_meta_model dashed). Lines mostly move together; red solid edges ahead slightly. All end slightly above 1.0 (modest overall gains). | Shows which allocation method captured the most upside. Overlap confirms all methods rode the same underlying strategy returns; the small spread between methods means allocation choice doesn't matter much when no strategy is strongly validated. |
| `allocation_sharpe_comparison.png` | 5 bars, all blue except red for dynamic. Mean_variance_shrinkage ~0.189 (highest), best_individual ~0.185, inverse_volatility ~0.031, equal_weight and dynamic ~0.009 (lowest). | Proves dynamic did not beat naive baselines on this run. Shrinkage won, but barely—no allocation method found a big edge. |
| `static_weights.png` | Grouped bars, 4 methods × 3 strategies. Best_individual concentrates all capital in one strategy (alpha_02). Others diversify 1/3 each (equal) or reweight by volatility (inverse) or Markowitz (shrinkage). | Makes allocation philosophy visible at a glance: best_individual is all-or-nothing; shrinkage is the workhorse, most balanced. |
| `dynamic_weights_over_time.png` | Stacked area chart, 3 colored bands (one per strategy). Bands stay roughly 1/3 each with minimal movement over time—occasional small wiggles but visibly flat. | Proves the 5% no-trade band worked as designed: the learned allocator moved weights only slightly despite changing data, keeping near a naive 1/3 split. This is why its return nearly matched equal_weight's. |
| `factor_alpha_beta.png` | Two bar charts side by side. **Left (alpha t-stat)**: mostly green bars in range -2 to +0.5, all below the ±1.96 significance threshold (dashed lines). **Right (beta)**: blue bars, baseline_pb01 ~0.6, alpha_02 ~0.0 (orthogonal to buy-hold), alpha_05 ~0.35. | Shows each strategy's timing skill (alpha) and correlation with market drift (beta). Baseline is pure beta (drift-riding). Alpha_02 is independent of the asset's own movement—it is truly market-neutral in its return source. |
| `meta_model_null_baseline.png` | Histogram of gray bars (permuted nulls) centered ~48%. Red vertical line at real walk-forward accuracy ~52.3%. Blue dashed line at 50% (no-information). Real is above null's mean but squarely inside the null distribution (not separated). | Visualizes why p=0.198 even though 52.3% > 50%: the learned allocator does edge the null, but the edge is small and within noise. The picture makes "plausible but not significant" tangible. |

## What gets combined, and why

Task 2 found that no strategy clears the BH-corrected significance bar
on this sample (see `task2/README.md`). Task 3 needs at least two return
streams to demonstrate portfolio construction at all, so it carries
forward `practical_set_for_task3` -- the top-3 strategies by unadjusted
Sharpe (`alpha_02_band_reversion`, `alpha_05_learned_composite`,
`baseline_pb01`) -- **with the caveat printed at the top of every run
and written into `task3_report.json`**: this combination demonstrates
the allocation machinery, it is not a claim that these three are
validated alpha.

## Factor model: alpha vs. beta against buy-and-hold

This is a single-instrument dataset, so there is no cross-sectional
market index to regress against. The natural factor here is the asset's
OWN buy-and-hold (open-to-open) return -- it isolates how much of a
strategy's return is just "being correlated with the underlying's own
drift" (beta) versus genuine timing skill (alpha). On this run:

- `baseline_pb01` (long whenever the trend flag is up) is almost
  entirely beta: R-squared 0.59 against buy-and-hold, and a *negative*
  alpha t-stat of -2.07 -- its historical return is explained by simply
  being correlated with the asset's own upward drift, with a
  statistically negative timing contribution on top.
- `alpha_02_band_reversion` carries almost no beta at all (R-squared
  0.0015) -- whatever its return is, it is not a repackaged version of
  "the asset went up."
- `alpha_05_learned_composite` sits in between (R-squared 0.33,
  negative but insignificant alpha).

## The five allocation methods, and the honest result

| Method | Mechanism | This run's Sharpe |
|---|---|---|
| 1. best_individual | all capital on the single best in-sample Sharpe | 0.185 |
| 2. equal_weight | 1/n each | 0.009 |
| 3. inverse_volatility | weight $\propto 1/\sigma$, ignores correlation | 0.031 |
| 4. mean_variance_shrinkage | Markowitz direction, covariance shrunk 30% toward diagonal | **0.189** |
| 5. dynamic_meta_model | walk-forward learned scores -> softmax weights, no-trade band | 0.009 |

`mean_variance_shrinkage` is the best of the five here, modestly ahead
of the naive best-individual baseline it has to justify its extra
complexity against. The dynamic method's *average* weights over time
came out at exactly 1/3 each -- the 5% no-trade band, combined with a
cold-start equal-weight fallback before the model has enough walk-
forward history, meant it rarely moved far from equal weight in
practice, so its portfolio return matches `equal_weight`'s almost
exactly. This is reported plainly rather than dressed up: **the dynamic
allocator did not demonstrably beat the naive baselines on this run.**

## The meta-model's own validation, reported honestly

`AlphaMetaModel.null_baseline()` compares its walk-forward direction
accuracy against a permuted-label null:

- Real walk-forward accuracy: **52.3%** (37 folds)
- Permuted-label null: mean 48.2%, std 4.1%
- **p-value vs. null: 0.198**

This is a plausible edge (above the 50% no-information baseline, above
the null's mean) but **not statistically significant at conventional
thresholds** -- exactly the model-discipline requirement this project
is built around: report the null-baseline comparison honestly, rather
than present the 52.3% headline number alone and imply it is validated.
A smaller, validated improvement would be worth more than this
unvalidated one; here, there isn't a validated improvement to report,
and the report says so.

## Design notes

- **Rebalance frequency**: 21 bars (~monthly), giving 37 walk-forward
  folds over the sample -- few enough that `AlphaMetaModel` is
  deliberately kept to 2 features (trailing Sharpe, trailing hit-rate)
  per strategy per fold, the same capacity-discipline reasoning as
  `alpha_05` in Task 2.
- **Rebalancing cost**: reallocating capital between strategies is
  charged at the same 0.05%/side rate as any other trade in the
  underlying (documented assumption -- moving capital between two
  already-running strategies still means unwinding one position and
  entering another in the same instrument).
- **Portfolio-level trade statistics read as zero** in the comparison
  table (`num_round_trips`, `win_rate`, etc.) -- this is expected, not a
  bug: `PerformanceAnalyzer` computes round-trip statistics from a
  single discrete position series, which a blended multi-strategy
  portfolio doesn't have. Per-strategy trade statistics are in
  `task2/results/task2_report.json`.
- **Static methods (1-4) are fit once** on the full return matrix and
  held fixed -- they are baselines for the dynamic method to beat, not
  themselves adaptive. Only method 5 is walk-forward validated, which is
  exactly why it is the only one checked against a null baseline.



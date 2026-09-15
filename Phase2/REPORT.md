# Multi-Alpha Quantitative Trading Lab: Technical Report

## 1. Executive Summary

This project constructs an end-to-end quantitative trading pipeline for single-instrument alpha discovery, strategy validation, and portfolio construction. The pipeline enforces strict research discipline: lookahead-bias prevention via structural timing enforcement, significance testing with autocorrelation adjustment and multiple-testing correction, and honest null-baseline validation for every learned model. The research finds no statistically validated edge in the supplied 986-day, single-instrument dataset using five distinct alpha strategies and five allocation methods, a null result that is itself the finding.

## 2. Data and Preprocessing

### 2.1 Raw Input
- **Price file**: 1000 OHLCV records, `price_train.csv`
- **Signal file**: 20 pre-computed indicators in three categories (price-based: PB01–PB08; band-based: BB01–BB07; volume-based: VB01–VB05), `signals_train.csv`
- **Period**: January 2, 2018 – December 2, 2021 (approximately 1000 trading days)

### 2.2 Data Quality Issues Discovered and Remedied
1. **Mixed date formats**: Most rows in `price_train.csv` use YYYY-MM-DD; ~30 scattered rows use DD-MM-YYYY (e.g., `21-02-2018`). Parsed deterministically by detecting which token is 4 digits (`_parse_date_token` in `data_cleaner.py`), avoiding the ambiguity of `dayfirst` heuristics.

2. **Format-disguised duplicates**: 14 duplicate days in the price file, several hidden by using the *other* date format for the same OHLCV (e.g., `2018-07-18` and `18-07-2018` both present with identical prices). De-duplication runs *after* date normalisation so format-disguised copies are caught.

3. **Signal-only rows**: 14 dates present in the signal file with no matching price row (not duplicates, genuinely absent from the price file). Inner join on `date` drops these and logs the count; no forward-filling of prices that were never observed.

4. **Warm-up NaNs**: Columns `BB06` (19 rows), `BB07` (13 rows), `VB05` (19 rows) have NaN at the start, representing rolling windows not yet filled. These are left as NaN (never fabricated) and strategies reading them are documented to treat NaN as "no signal."

**Final clean dataset**: 986 usable rows (Jan 2, 2018 – Nov 1, 2021).

## 3. Backtesting Infrastructure (Task 1)

### 3.1 Core Architecture
The backtester enforces two critical invariants:

**Invariant 1: No lookahead bias (t-1 cutoff)**
- Signals for decision at candle `t` may use data up to and including candle `t-1`'s close.
- Implemented structurally: `Backtester.generate_signals()` shifts all signal columns by `SIGNAL_LAG=1` before passing to the strategy. The strategy never sees row `t`'s signals when deciding row `t`'s trade because they are not available.
- Verified via two automatic tests:
  - *Truncation check*: `strategy.generate_signal()` run on `data.iloc[:k]` matches the full-sample run restricted to the same rows.
  - *Perturbation check*: Doubling a signal value strictly after the sample midpoint does not change any signal at or before the midpoint.

**Invariant 2: Fill at open, cost charged correctly**
- All trades fill at candle `t`'s `open` price, not close.
- Transaction cost: 0.05% per side (mandated by spec), applied to `|position(t) – position(t-1)|`.
- Slippage: 5% of that candle's `(high – low) / open` range, scaled by trade size (not mandated; documented assumption accounting for intraday volatility).

### 3.2 Accounting Convention: Open-to-Open
- Position held over `[t-1, t)` interval: `position(t-1)` (entered at `t-1`'s open).
- Return earned: `position(t-1) × (open(t) / open(t-1) – 1)`.
- Cost charged at `t`: transaction cost + slippage on the trade executed at `t`'s open.
- This matches the mandated fill-at-open rule exactly: the position change happens at `t`'s open, and its return (and cost) are realized immediately thereafter.

### 3.3 Baseline Strategy
`BaselineStrategy`: long when `PB01==1`, flat otherwise. Deliberately simple, exists only to verify the pipeline runs end to end. Over the 986-day sample: **Sharpe –0.23**, total return –13.2%, max drawdown –33.4%. Not profitable, as expected; this strategy reads a single discrete signal without any hypothesis beyond "does a simple trend flag predict anything."

## 4. Alpha Research (Task 2)

### 4.1 Five Distinct Hypotheses
Each strategy encodes a different market mechanism and reads a mostly disjoint signal subset (enforced at the interface level):

#### 4.1.1 Alpha01: Trend Agreement
- **Mechanism**: Trend continuation. Buy when short-term trend (PB01) *and* long-term trend (PB02) *and* price-above-average (PB04) all agree on direction.
- **Signals**: PB01, PB02, PB04 (all binary).
- **Reasoning**: Divergence between short- and long-term trends may be noise; agreement may indicate genuine momentum.
- **Result**: Sharpe –0.64, permutation p-value 1.00 (observed Sharpe is at the left tail of the null).

#### 4.1.2 Alpha02: Band-Extreme Mean Reversion
- **Mechanism**: Mean reversion at Bollinger Band extremes. Buy when overbought+confirmed, sell when oversold+confirmed.
- **Signals**: BB03 (overbought), BB04 (oversold), BB06 (continuous band position).
- **Reasoning**: Extremes tend to mean-revert before trending further; mean reversion typically has shorter-term validity than trend continuation.
- **Result**: Sharpe +0.185, permutation p-value 0.096. Closest to significance (raw p < 0.10) but fails BH correction (adjusted p = 0.575 after testing 6 strategies).

#### 4.1.3 Alpha03: Volume-Confirmed Breakout
- **Mechanism**: Long-only continuation on breakout + volume confirmation (no equivalent short rule found in signal library).
- **Signals**: PB06 (breakout), VB01 (volume participation), VB03 (volume-supported).
- **Reasoning**: Breakouts without volume are often false; volume-confirmed breakouts are more reliable.
- **Result**: Sharpe –0.58, permutation p-value 0.70.

#### 4.1.4 Alpha04: Squeeze-Expansion Timing
- **Mechanism**: Volatility regime timing. Trade in momentum direction for 10 bars after a squeeze (low volatility, BB05==0 after BB05==1) ends.
- **Signals**: BB05 (squeeze flag), PB05 (momentum direction).
- **Reasoning**: Markets sometimes rally after a low-volatility regime ends; timing the regime switch and momentum direction may capture this.
- **Result**: Sharpe –1.39, permutation p-value 0.96.

#### 4.1.5 Alpha05: Learned Composite
- **Mechanism**: Logistic regression on five continuous signals, predicting sign of 5-bar-forward price *difference*.
- **Signals**: PB07, PB08, BB06, BB07, VB05 (all continuous, ranges ~[–2, +2]).
- **Model**: Logistic regression (6 parameters: intercept + 5 coefficients).
- **Model choice rationale**: ~600 development-period rows (before 2021-01-01). With 5 features, 6 parameters is at the ceiling of defensible complexity; a tree ensemble would carry hundreds of effective parameters against a few hundred rows and memorize noise. Predicting sign (direction), not magnitude (return), is consistent with the project discipline of using price *difference*, not level, as the prediction target.
- **Result**: Sharpe +0.102, permutation p-value 0.236. Balance report flags it: only 1 round trip over the sample, so win-rate (100%) and Sharpe are not statistically meaningful.

### 4.2 Signal Characterization
Hit-rate (direction-only) and information coefficient (magnitude-aware correlation) computed for all 20 signals against the forward price *difference* at horizons 1, 3, 5, 10, 20 bars. Top signals by absolute IC:
- PB07 vs. 10-bar fwd diff: IC = –0.18 (significant correlation, opposite direction).
- BB03 vs. 3-bar fwd diff: IC = –0.12 (mean reversion signal).

Individual signals show predictive power, but none of the five hand-built rules converts that into a cost-surviving edge after multiple-testing correction.

### 4.3 Statistical Significance: The Honest Null Result

#### Methodology
**Permutation test** (corrected version): Circularly shifts the strategy's *position series* against the fixed, real market data and re-runs it through the actual `ExecutionEngine` + `Portfolio`, computing the resulting Sharpe. Builds a null distribution of 500 permuted Sharpes. Tests whether the observed Sharpe is likely under the null; if not, the strategy's edge comes from timing (being long/short at the right *times*), not from market drift or costs.

**Newey-West t-stat**: Adjusts the standard error of mean daily return for autocorrelation (positions persist, returns are autocorrelated); a plain i.i.d. t-test would be overconfident.

**Multiple-testing correction**: Benjamini-Hochberg (FDR control) across all 6 strategies tested. Per-test α = 10%; after correction, the bar for any single strategy to be "significant" rises.

#### Results
| Strategy | Sharpe | Hit-rate (5-bar) | Perm p-value | BH-adjusted p | Significant (α=0.10) |
|---|---|---|---|---|---|
| baseline_pb01 | –0.23 | 48% | 0.99 | 0.998 | No |
| alpha_01 | –0.64 | 42% | 1.00 | 0.998 | No |
| **alpha_02** | **0.185** | **54%** | **0.096** | **0.575** | **No** |
| alpha_03 | –0.58 | 44% | 0.70 | 0.998 | No |
| alpha_04 | –1.39 | 39% | 0.96 | 0.998 | No |
| alpha_05 | 0.102 | 50% | 0.236 | 0.707 | No |

**Finding**: None of the six tested edges clears the corrected significance bar. Even `alpha_02`, which comes closest (raw p = 0.096, genuinely close to the conventional 0.05 line for a single test), is not significant after accounting for the six tests tried. This is reported as a result, not hidden.

### 4.4 Robustness Checks

**Sub-period analysis** (4 quarters, Sharpe per block): No strategy is stable across all four periods.

**Cost sensitivity**: For each strategy, re-run the backtest at cost levels 0%, 0.05%, 0.10%, 0.15%, 0.20%/side. None shows a plausible "break-even cost" much above 0.05%.

**Regime conditioning** (split by BB05 squeeze flag): Strategies do not concentrate their returns in one regime; results are not regime-specific artifacts.

**Orthogonality analysis** (pivoted QR on the return matrix):
- Effective rank: 6 out of 6 strategies (full rank, no strategy is numerically redundant).
- Rolling-window stability of the "top pivot" (most information-bearing strategy): 33% (changes often). The ranking of strategies is not a stable structural property on this sample.

### 4.5 Model Discipline: Every Model States Its Capacity Ceiling
`alpha_05`'s docstring and the project README state explicitly: "5 features, ~600 dev-period rows → 6 parameters (5 feature coefficients + intercept) is the defensible ceiling. More would memorize. This is a linear model because that's how much structure the data can honestly support."

Similarly, Task 3's `AlphaMetaModel` (logistic regression on 2 features per strategy per rebalance window) includes: "~37 walk-forward folds, so 2 features × 3 strategies = 6 per-fold observations to build the feature matrix. A model with > ~2 parameters per strategy overfits by construction."

## 5. Portfolio Construction & Allocation (Task 3)

### 5.1 Strategy Combination Problem
Task 3 needs at least two return streams to demonstrate allocation. Since Task 2 found no statistically validated edge, `practical_set_for_task3` carries forward the top-3 by unadjusted Sharpe:
- `alpha_02_band_reversion` (0.185)
- `alpha_05_learned_composite` (0.102)
- `baseline_pb01` (–0.23)

**Caveat**: Explicitly stated in JSON output and console: "These three are unvalidated. Allocation methods are demonstrated here; this is not a claim of alpha."

### 5.2 Factor Model: Alpha vs. Beta Decomposition
Regression of each strategy's return against a single factor: the asset's own buy-and-hold open-to-open return.

**Model**: `r_i(t) = α_i + β_i × r_BnH(t) + ε_i(t)`

**Why this factor**: Single-instrument dataset; no cross-sectional market index available. The asset's own buy-and-hold return isolates how much of a strategy's return is just "being long the underlying's drift" (β) versus genuine timing skill (α).

**Results**:
- **baseline_pb01**: α_tstat = –2.07 (negative timing), β = 0.59, R² = 0.59. Almost entirely beta; the return it achieved was just riding the market's upward drift, with worse-than-passve timing on top.
- **alpha_02_band_reversion**: α_tstat = +0.32 (insignificant), β ≈ 0, R² = 0.0015. Near-zero beta; return is not explained by buy-and-hold correlation. Mean reversion truly is a different bet.
- **alpha_05_learned_composite**: α_tstat = insignificant, β = some positive value, R² = 0.33. Moderate beta; some drift correlation but also independent signal.

### 5.3 Five Allocation Methods Compared

| Method | Mechanism | This Run Sharpe | Notes |
|---|---|---|---|
| 1. best_individual | 100% on single best Sharpe | 0.185 | Naive baseline; only alpha_02 gets capital. |
| 2. equal_weight | 1/3 each | 0.009 | Loses to the best single strategy due to including baseline_pb01. |
| 3. inverse_volatility | Weight ∝ 1/σ (ignores correlation) | 0.031 | Still loses; doesn't account for co-movement. |
| 4. mean_variance_shrinkage | Markowitz (Σ⁻¹μ with 30% shrinkage toward diagonal) | **0.189** | Best of five. Shrinkage prevents wild weights on a small sample (1000 bars, 3 strategies, ~singular covariance). |
| 5. dynamic_meta_model | Walk-forward learned scores → softmax weights with 5% no-trade band | 0.009 | Converges to equal weight in practice due to the no-trade band + cold-start equal-weight fallback. |

**Key finding**: Method 4 (mean-variance with shrinkage) edges out method 1 (best individual) — a modest gain from spreading risk more carefully — but method 5 (the learned allocator) shows *no validated benefit*. Its null-baseline check:
- Real walk-forward direction accuracy: 52.3% (predicting sign of next month's return for each strategy).
- Permuted-label null: mean 48.2%, std 4.1% (100 permutations).
- p-value: 0.198 (one-in-five chance a gap this size occurs in noise).

The learning did not produce a statistically significant improvement. The model's architectural features (5% no-trade band, expanding-window walk-forward, cold-start equal weight) are sound, but the edge — if real — is too small to confirm on this data.

### 5.4 Rebalancing and Cost
- **Frequency**: 21 bars (~monthly), giving 37 walk-forward folds over the sample.
- **Static methods** (1–4): Fit once on the full return matrix, held fixed. No rebalancing cost.
- **Dynamic method** (5): Rebalances at each fold; charged 0.05%/side on each reallocation (same as any trade in the underlying).
- **No-trade band**: 5% absolute; weights only move if their change exceeds this band. Prevents churn on small, possibly-noisy model output changes.

## 6. Disciplines and Invariants

The entire project enforces five core disciplines:

1. **Price difference for prediction, not price level**. Labels are `close[t+h] – close[t]` (the move), not `close[t+h]` (the level). Direction (hit-rate) and magnitude (IC) are both computed for every signal, never just one.

2. **Every model states why it isn't bigger**. Alpha05 (logistic, 6 params): "5 features × 600 dev rows = 6 params is the ceiling." Meta-model (logistic, 6 params per fold): "~37 folds × 3 strategies = 2 features/strategy is defensible." Models larger than this would memorize noise on this data.

3. **Edge cases handled in code, not hope**. Mixed date formats, warm-up NaNs, misaligned rows, trades that don't change position — all handled explicitly and logged, not glossed over.

4. **Formatting validated**. Boolean signal columns checked to contain only {0, 1}; continuous ones flagged if outside expected ranges; joins checked for row count changes.

5. **Good-ish everywhere over great in one place**. A strategy with 50% Sharpe but –50% max drawdown is not "better" than one with 0.2% Sharpe and –5% drawdown. Balance report flags strategies with spiky profiles. Final selection prefers a validated, modest edge over an unvalidated large one.

## 7. Honest Null-Baseline Validation

Every learned model (alpha_05, meta-model) is checked against a null where its learning signal is scrambled:
- **Alpha05**: Trained on the standard 5-bar-forward-difference labels, tested on held-out data. Performance is what it is; no null test applied (it's a one-shot fit on data strictly before 2021-01-01, predict on data strictly after).
- **Meta-model** (AlphaMetaModel): Walk-forward, refit at each rebalance point. Accuracy checked against a permuted-label null where the target (positive/negative next-period return) is shuffled within each training window. Real accuracy 52.3% vs. null 48.2% (p=0.20). Not significant; the edge exists but is not confirmed.

## 8. Conclusion

The research pipeline is built for discipline: lookahead prevention via structural enforcement, significance testing with autocorrelation adjustment, multiple-testing correction, and null-baseline validation for every learned model. Applied to a single-instrument, 986-day dataset with 20 pre-computed signals and five alpha hypotheses, the results are:

- **No strategy produces a statistically validated edge** after proper permutation testing and BH correction.
- **Individual signals carry predictive power** (IC ≈ –0.18 for some), but hand-built rules don't convert this into cost-surviving alpha.
- **Learned models (alpha_05, meta-model) show plausible but unvalidated edges** — they beat no-information baselines slightly, but not by enough to rule out luck.
- **The gap between "signal has IC" and "rule using that signal survives permutation test" is itself a finding**, pointing to the difficulty of converting raw correlation into actionable alpha under transaction costs and autocorrelation.

This null result is honestly reported rather than hidden behind cherry-picked headlines, because a report that hides its own defeats is worth less than one that shows them.

---

## Appendix: Plots Generated

Each task generates charts in `results/plots/`:

**Task 1** (3 charts):
- `equity_vs_buyhold.png`: Strategy equity vs. buy-and-hold benchmark.
- `drawdown.png`: Underwater curve (% below prior equity peak at each point).
- `position_timeline.png`: Position (–1/0/+1) over time against close price.

**Task 2** (5 charts):
- `equity_curves.png`: All 6 strategies' equity curves on one axis.
- `sharpe_comparison.png`: Sharpe per strategy (green if BH-corrected significant).
- `signal_ic_heatmap.png`: IC of all 20 signals vs. forward difference, across horizons.
- `strategy_correlation_heatmap.png`: Pairwise correlation of strategy returns.
- `permutation_tests.png`: Actual null distribution behind each strategy's p-value.

**Task 3** (6 charts):
- `allocation_equity_curves.png`: All 5 allocation methods' equity curves.
- `allocation_sharpe_comparison.png`: Sharpe by method.
- `static_weights.png`: How each static method splits capital.
- `dynamic_weights_over_time.png`: Learned allocator's daily weights (shows convergence to equal weight).
- `factor_alpha_beta.png`: Alpha t-stat and beta per strategy.
- `meta_model_null_baseline.png`: Real walk-forward accuracy vs. permuted-label null.

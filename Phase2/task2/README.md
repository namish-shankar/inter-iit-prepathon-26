# Task 2 -- Alpha Research


## New files

```
strategies/
  alpha_01.py   Trend agreement (PB01, PB02, PB04)
  alpha_02.py   Band-extreme mean reversion (BB03, BB04, BB06)
  alpha_03.py   Participation-confirmed breakout (PB06, VB01, VB03) -- long only
  alpha_04.py   Volatility squeeze -> expansion (BB05, PB05)
  alpha_05.py   Learned composite of the 5 continuous signals (logistic regression)
statistics.py       StatisticalTester
robustness.py        RobustnessTester
orthogonality.py      OrthogonalityAnalyzer
alpha_research.py     AlphaResearch (orchestrator main.py calls)
```

## The five strategies, and why each is genuinely different

Each strategy reads a **disjoint (or near-disjoint) subset of the 20
signals** and encodes a **different market mechanism**, not five
parameter variants of one idea:

| Strategy | Mechanism | Signals |
|---|---|---|
| alpha_01 | trend continuation | PB01, PB02, PB04 |
| alpha_02 | mean reversion at band extremes | BB03, BB04, BB06 |
| alpha_03 | breakout continuation, volume-confirmed | PB06, VB01, VB03 |
| alpha_04 | volatility-regime timing | BB05, PB05 |
| alpha_05 | learned linear composite | PB07, PB08, BB06, BB07, VB05 |

alpha_02 and alpha_05 both touch BB06, but for opposite reasons -- one
reads it as a discrete band-extreme confirmation, the other as one of
five continuous inputs to a fitted model -- and the orthogonality
analysis (below) checks that overlap empirically rather than assuming
it away.

## alpha_05's model, and why logistic regression specifically

Documented in the strategy's own docstring, summarised here: five
continuous signals, none individually dominant, ~600 development-period
bars to fit on -> a linear model on 5 standardised features (6 free
parameters) is the highest-capacity choice defensible at this sample
size. A tree ensemble would carry hundreds of effective degrees of
freedom against a few hundred rows -- memorising noise, not learning
structure. The model predicts **sign of the 5-bar-forward price
difference** (a classifier on direction), not the return magnitude,
consistent with the "price difference for prediction, not price" rule
used throughout this project. It is fit ONCE on data strictly before
`config.DEV_HOLDOUT_SPLIT_DATE` (2021-01-01); `generate_features` /
`generate_signal` never see price, only the five signal columns.

## The honest result on this sample

Running `main.py` end to end on the supplied training data:

- **No strategy clears the Benjamini-Hochberg-corrected significance
  bar** at the 10% level (`results/task2_report.json` ->
  `statistics.*.significant_after_bh_correction`, all `false`). Several
  individual signals *do* show a non-trivial information coefficient in
  isolation (PB07 vs. 10-bar-forward difference: IC = -0.18; BB03 vs.
  3-bar-forward difference: IC = -0.12 -- see
  `task2_signal_characterization.csv`), but none of the five hand-built
  strategies converts that into a return series that survives a
  permutation test once transaction costs and the correction for
  testing five strategies at once are applied.
- This is reported as a **finding**, not hidden or argued around --
  it is exactly the outcome the permutation test and multiple-testing
  correction exist to catch. The strategy that looks best on raw Sharpe,
  `alpha_02_band_reversion` (0.185), has the smallest *raw* permutation
  p-value of the five (0.096 -- genuinely close to the conventional 0.05
  line), but that raw p-value is a single one of six tests being run at
  once; after the Benjamini-Hochberg correction for testing all six
  together it becomes 0.575, well short of significant. Every other
  strategy sits further out (raw p-values 0.24-1.00; see
  `results/plots/permutation_tests.png` for the null distribution behind
  each one).
- Because Task 3 needs at least two return streams to demonstrate its
  allocation machinery, `AlphaResearch.select_final_set` also returns a
  `practical_set_for_task3` (top-3 by unadjusted Sharpe:
  `alpha_02_band_reversion`, `alpha_05_learned_composite`,
  `baseline_pb01`) with an explicit caveat attached in the JSON output:
  this set is for demonstrating the *allocation mechanics*, not a claim
  of validated alpha. Task 3 carries that caveat forward rather than
  quietly dropping it.

## Discipline 5 in practice: the balance report

`balance_report()` in `main.py` flags any strategy whose metric profile
is spiky rather than good-ish across the board (a decent Sharpe next to
a >30% drawdown, or fewer than 5 round trips making every other metric
unreliable). On this run only `alpha_05_learned_composite` is flagged
(a single round trip over the sample -- its win-rate of 100% and
Sharpe are not meaningful on n=1, and the selection logic treats it
accordingly rather than taking the headline number at face value).

## Statistical machinery (statistics.py)

- **hit_rate / information_coefficient**: direction-only vs.
  magnitude-aware read of a signal's edge against the forward price
  *difference* (never the level) -- see "final digest" discipline 1 in
  the project-wide notes.
- **t_stat_with_newey_west**: corrects the standard error for the serial
  correlation a position-holding return series has; a plain i.i.d.
  t-test would be overconfident here.
- **permutation_test**: builds a null distribution for Sharpe by
  circularly shifting each strategy's *position* series against the
  market's own, fixed OHLC data and re-running it through the real
  ExecutionEngine + Portfolio -- tests whether the edge comes from being
  long/short at the right *times*, not just from market drift plus
  favourable costs. (Shifting or shuffling the already-realised return
  series directly does NOT work for a Sharpe-based statistic -- mean and
  std depend only on the set of values, not their order, so any
  reordering of the same returns reproduces the identical Sharpe. An
  earlier version of this method did exactly that; the bug was caught
  visually, because its "null distributions" were a single spike on the
  observed value every time -- see `results/plots/permutation_tests.png`
  for the corrected version, and `statistics.py`'s docstring for the
  full explanation.)
- **block_bootstrap_ci**: moving-block resampling for a confidence
  interval on Sharpe, for the same autocorrelation reason.
- **multiple_testing_correction**: Benjamini-Hochberg (default) or
  Bonferroni across the whole strategy set at once.

## Robustness machinery (robustness.py)

- **sub_period_analysis**: 4-way split, Sharpe per block -- catches an
  edge that is really one lucky quarter.
- **cost_sensitivity**: reruns the backtest at cost levels from 0 to
  0.20%/side and estimates the approximate break-even cost, i.e. how
  much cushion the edge has over the mandated 0.05%/side.
- **regime_conditioning**: splits returns by the BB05 squeeze flag --
  catches an edge that quietly only works in one volatility regime.

## Orthogonality machinery (orthogonality.py)

`pivoted_qr` on the (variance-scaled) return matrix gives an **effective
rank** for the strategy set -- on this run, rank 6 out of 6 strategies,
i.e. no strategy is numerically redundant with the rest at the
tolerance used. `residual_alpha` regresses each strategy on all the
others and reports how much of its variance the rest of the set already
explains. `rolling_window_stability` reruns the pivoted QR over rolling
250-bar windows; the "most information-bearing" strategy (top pivot)
changes across windows on this sample (33% stability), which is itself
informative -- it says the *ranking* of which strategy carries the most
independent signal is not a stable structural property here, consistent
with the significance-testing result above.

## Plots (`plotting.py`, `results/plots/`)

Generated fresh by every `main.py` run (5 charts):

| Plot | Expected content | Why |
|---|---|---|
| `equity_curves.png` | 6 colored lines (one per strategy) + gray dashed buy-and-hold. B&H ends ~1.4 (40% gain); most strategies end below 1.0 (negative returns); alpha_02 (green) ~1.05 (5% return), alpha_05 (brown) ~1.00 (flat). All curves are noisy (frequent rebalancing). | Shows each strategy's full return path. Most underwater confirms the result: no strategy beats the passive benchmark, and most add negative alpha on top of transaction costs. |
| `sharpe_comparison.png` | 6 bars, all gray (none green—no green bar means none survive BH-corrected significance). Alpha_02 highest at ~0.185, baseline negative, others scattered -0.2 to 0.15. No bar reaches the implicit green threshold. | The single most important result: visually proves no strategy passes the significance bar after multiple-testing correction. If a bar were green, it would be the validated edge; here there is none. |
| `signal_ic_heatmap.png` | 20 rows (signals) × 5 columns (forward horizons: 1/3/5/10/20 bars). Mix of colors: dark blue cells (negative IC, mean-reversion signal), light colors (weak IC), occasional reds (trend-following). Dark blue concentrated in PB07–PB08 rows and longer horizons. | Shows each signal's predictive content across different time horizons. No signal is uniformly strong at all horizons; some (PB07) show real mean-reversion IC (–0.18) at certain horizons but fail at others. Guides alpha design. |
| `strategy_correlation_heatmap.png` | 6×6 symmetric matrix, diagonal all 1.0. Baseline and alpha_01 correlated ~0.59 (both trend-following). Alpha_02 anticorrelated with baseline (–0.21, mean reversion vs. trend). Alpha_05 occupies middle ground. Off-diagonals scattered –0.2 to +0.6. | Proves the strategies are not redundant copies. The negative correlations (alpha_02 vs. baseline) show diversification benefit if both were validated, which they aren't. Diagonal dominance with off-diagonal scatter matches the QR rank-6 result. |
| `permutation_tests.png` | 6 subplots (one per strategy), each a histogram of gray bars + red vertical line marking observed Sharpe. Each histogram centered near 0 (null mean) and observed Sharpe sits mostly inside or on the right tail of null, never far outside. Red lines for alpha_02 closest to right tail (~p=0.096 raw), others further in. | The visual proof that no strategy's edge is extreme. Most have p > 0.2 raw, meaning there's >20% chance a shuffled position would score as well just by luck. Alpha_02 edges closest (p=0.096), but that raw p becomes 0.575 after BH correction across the six strategies. |

## Assumptions carried from Task 1

0% risk-free rate, 252-day annualisation,
the documented slippage model, 2021-01-01 dev/holdout split, seed 42.
`alpha_05`'s label horizon (5 bars) and `alpha_04`'s post-squeeze
expansion window (10 bars) are additional documented assumptions, not
mandated by the spec.

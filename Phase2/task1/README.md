# Task 1 -- Data & Backtesting Infrastructure


## Project structure

```
task1/
  config.py            fixed execution rules + documented assumptions, in one place
  data_loader.py        DataLoader
  data_cleaner.py        DataCleaner
  feature_engine.py      FeatureEngine
  strategy.py             BaseStrategy (abstract)
  execution_engine.py    ExecutionEngine
  portfolio.py            Portfolio
  performance.py          PerformanceAnalyzer
  backtester.py           Backtester
  plotting.py             chart generation (see "Plots" below)
  strategies/
    baseline_strategy.py  BaselineStrategy(BaseStrategy)
  data/                   price_train.csv, signals_train.csv (as supplied)
  results/                equity curve, trade log, JSON report, plots/*.png (written by main.py)
```

## Plots (`plotting.py`, `results/plots/`)

Generated fresh by every `main.py` run (3 charts):

| Plot | Expected content | Why this matters |
|---|---|---|
| `equity_vs_buyhold.png` | Two lines: blue solid (strategy equity) and gray dashed (buy-and-hold). Strategy ends ~0.87 (13% loss); B&H ends ~1.4 (40% gain). Strategy is noisy with visible rebalancing; B&H is smooth uptrend. | Proves the backtester works end-to-end. Shows correct cost deduction (baseline is dumb, so it should lose to passive). Noisiness confirms frequent trading. |
| `drawdown.png` | Red underwater curve with peak at ~–33% below prior equity high. Multiple drawdown episodes visible (not a single cliff)—periods of months where strategy is underwater. | More informative than a single "max drawdown" number. Shows the *duration* and *path* of losses. A realistic strategy has multiple valleys as trades succeed and fail. |
| `position_timeline.png` | Blue stepped line (position +1/0/–1) overlaid on gray close-price line. ~17 distinct trade blocks (steps) scattered across 4 years. Long positions appear as +1 plateaus lasting weeks to months. | Proves the strategy is actually trading (not flat or broken). Steps should align with trend-flag transitions, not random—confirming the signal logic works. |

## Design decisions

**Timing convention is enforced structurally, not by convention.**
`Backtester.generate_signals` shifts the signal-library columns by one
row *before* they are ever passed to `strategy.generate_features` /
`generate_signal` -- a strategy physically cannot see row `t`'s signal
values when deciding row `t`'s trade, because it never receives them.
`ExecutionEngine.execute` fills every trade at that row's `open`, never
its close.

**Accounting convention.** The position held over the interval ending at
candle `t`'s open is `position(t-1)`; the return earned is
`position(t-1) * (open(t)/open(t-1) - 1)`, and cost is charged at `t` on
`|position(t) - position(t-1)|` (the trade that fills at `t`'s open).
This is open-to-open accounting, matching the mandated fill price exactly
-- see `portfolio.py`'s docstring for the full derivation.

**Transaction cost:** fixed at 0.05% per side (`config.py`), applied
uniformly. **Slippage** is not mandated by the spec, so it is a
documented assumption: `5% x (high-low)/open x |trade size|` -- it scales
with a bar's own realised range rather than a flat bp figure that would
be too tight on calm days and too loose on wild ones.

**Baseline strategy is deliberately dumb** (`long when PB01==1, else
flat`) -- it exists to prove the pipeline runs end to end, not to be
profitable. Task 2's `alpha_*` strategies are where the research
happens.

## Research-integrity checks actually run by `main.py`

1. **Structural NaN-pattern check** (`FeatureEngine.validate_no_lookahead`,
   no `signal_fn`): every numeric column's missing values form a clean
   *prefix* (warm-up), never NaNs reappearing after real values started.
2. **Truncation-stability check**: `strategy.generate_signal` run on
   `data.iloc[:k]` must agree with the full-sample run restricted to the
   same rows, for five checkpoints spanning the sample.
3. **Perturbation check**: doubling a signal value strictly after the
   sample midpoint must never change any signal at or before the
   midpoint.

Both (2) and (3) passed for the baseline strategy on this data.

## Real data-quality issues found and handled (not hypothetical)

Running `main.py` against the supplied `price_train.csv` /
`signals_train.csv` surfaced genuine issues, all logged in
`results/task1_report.json`:

- **Mixed date formats in `price_train.csv`.** Most rows are
  `YYYY-MM-DD`; a scattered minority (e.g. `21-02-2018`, `01-03-2019`)
  are `DD-MM-YYYY`. Parsed deterministically by which token is 4 digits
  (`data_cleaner.py:_parse_date_token`) -- no `dayfirst` guessing, which
  would have mis-parsed the genuinely ambiguous rows like `01-03-2019`.
- **14 duplicate dates in the price file**, several disguised by using
  the *other* date format for an already-seen day (e.g. `2018-07-18` and
  `18-07-2018` both present with identical OHLCV). De-duplication runs
  strictly *after* date normalisation for exactly this reason; format-
  disguised duplicates are still caught.
- **14 dates present in the signal file with no matching price row at
  all** (not duplicates -- genuinely absent), e.g. `2018-02-05`,
  `2021-01-05`. The inner join on `date` drops these from both sides and
  the count is reported explicitly (`join_report`), rather than silently
  forward-filling a price that was never observed.
- **Warm-up NaNs in `BB06` (19 rows), `BB07` (13 rows), `VB05` (19
  rows)** at the start of the signal file -- a rolling-window not yet
  filled, not a data error. Left as NaN (never fabricated), and any
  strategy reading them before they exist is documented to treat NaN as
  "no signal" (flat).

## Assumptions (Sec. 7 of the Problem Statement)

- Risk-free rate = 0 throughout.
- Annualisation factor = 252 trading days.
- Slippage model as described above (cost is fixed by spec, slippage is not).
- Development/holdout split date = 2021-01-01 (used from Task 2 onward).
- Duplicate-date conflicts (had there been any -- none were found) would
  be resolved by keeping the first chronological occurrence.

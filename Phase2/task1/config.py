"""
Central configuration for the Multi-Alpha Lab.

Every constant that the Problem Statement / Technical Documentation fixes
(the transaction cost, the timing convention, the annualisation factor)
lives here, and *only* here, so it can never silently drift between
modules. Anything not fixed by the spec (slippage model, split date,
seed) is a documented assumption -- see README.md for the reasoning.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths --
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PRICE_FILE = DATA_DIR / "price_train.csv"
SIGNALS_FILE = DATA_DIR / "signals_train.csv"

# ------------------------------------------------------- execution rules --
# Mandatory, fixed by the Technical Documentation Sec. 1.2 -- never a free
# parameter. 0.05% per side -> 0.0010 round trip.
TRANSACTION_COST_PER_SIDE = 0.0005

# Slippage is NOT mandated by the spec. We model it as a fraction of the
# candle's own (high-low) range at the fill price, which scales naturally
# with a name's realised volatility instead of a single constant bp figure
# that would be too tight on quiet bars and too loose on wild ones.
# Documented assumption -- see README.md "Execution & Cost Model".
SLIPPAGE_RANGE_FRACTION = 0.05  # 5% of that bar's (high-low)/open

# Timing convention (Technical Documentation Sec. 1.2): a decision for
# candle t may use signal rows up to and including t-1's close; the trade
# executes at candle t's open. This is enforced structurally in
# Backtester.generate_signals / ExecutionEngine.execute, not just assumed.
SIGNAL_LAG = 1

# ------------------------------------------------------------- research --
ANNUALIZATION_FACTOR = 252
RISK_FREE_RATE = 0.0  # documented assumption: no risk-free adjustment
RANDOM_SEED = 42

# Chronological development / holdout split. Everything at or after this
# date is held out and touched only for final evaluation -- never for
# threshold or parameter selection. ~80/20 split over the supplied sample.
DEV_HOLDOUT_SPLIT_DATE = "2021-01-01"

# Boolean signal columns must contain only {0, 1}; continuous ones are
# expected to fall roughly in this range (used only to *flag*, not clip,
# outliers in the data-quality report).
BOOLEAN_SIGNALS = [
    "PB01", "PB02", "PB03", "PB04", "PB05", "PB06",
    "BB01", "BB02", "BB03", "BB04", "BB05",
    "VB01", "VB02", "VB03", "VB04",
]
CONTINUOUS_SIGNALS = ["PB07", "PB08", "BB06", "BB07", "VB05"]
ALL_SIGNALS = BOOLEAN_SIGNALS + CONTINUOUS_SIGNALS

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]

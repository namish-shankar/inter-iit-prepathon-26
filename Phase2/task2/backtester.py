"""
Backtester -- orchestrates the whole per-candle loop from the Technical
Documentation:

    for t in candles:
        inputs_t  = signal_library rows up to and including t-1
        decision_t = strategy.generate_signal(inputs_t)
        execute decision_t at price = open[t]

generate_signals() enforces the t-1 cutoff structurally, by shifting the
signal-library columns one row *before* they ever reach
strategy.generate_features / generate_signal -- so a strategy cannot see
row t's signals when deciding row t's trade, no matter how it is written
internally (it never receives them).
"""

import numpy as np
import pandas as pd

import config
from execution_engine import ExecutionEngine
from portfolio import Portfolio
from performance import PerformanceAnalyzer
from feature_engine import FeatureEngine


class Backtester:
    def __init__(self, cost_per_side=None, slippage_range_fraction=None,
                 initial_capital=1.0):
        self.execution_engine = ExecutionEngine(cost_per_side, slippage_range_fraction)
        self.portfolio = Portfolio(initial_capital)
        self._results = {}

    def generate_signals(self, data, strategy):
        """Shift the signal-library columns by SIGNAL_LAG before handing
        them to the strategy -- this is the structural enforcement of the
        t-1 cutoff, not a convention the strategy has to remember."""
        signal_cols = [c for c in config.ALL_SIGNALS if c in data.columns]
        lagged_inputs = data[signal_cols].shift(config.SIGNAL_LAG)

        features = strategy.generate_features(lagged_inputs)
        raw_signal = strategy.generate_signal(features)
        return raw_signal.reindex(data.index).fillna(0.0)

    def execute_signals(self, signals, data):
        price_cols = [c for c in config.PRICE_COLUMNS if c in data.columns]
        return self.execution_engine.execute(signals, data[price_cols])

    def update_portfolio(self, executions, data):
        price_cols = [c for c in config.PRICE_COLUMNS if c in data.columns]
        return self.portfolio.update(executions, data[price_cols])

    def analyze(self, portfolio):
        analyzer = PerformanceAnalyzer(
            returns=portfolio.get_pnl(),
            trade_log=self.execution_engine.get_trade_log(),
            positions=portfolio.get_positions(),
        )
        return analyzer.report()

    def run(self, data, strategy, check_lookahead=True):
        """Run the full pipeline end to end and stash every intermediate
        artefact in self._results (see get_results())."""
        lookahead_ok = None
        if check_lookahead:
            fe = FeatureEngine()
            lookahead_ok = fe.validate_no_lookahead(
                data, signal_fn=lambda d: self.generate_signals(d, strategy)
            )

        signals = self.generate_signals(data, strategy)
        executions = self.execute_signals(signals, data)
        equity_curve = self.update_portfolio(executions, data)
        report = self.analyze(self.portfolio)

        self._results = {
            "strategy": strategy.get_metadata(),
            "signals": signals,
            "executions": executions,
            "equity_curve": equity_curve,
            "trade_log": self.execution_engine.get_trade_log(),
            "report": report,
            "lookahead_check_passed": lookahead_ok,
        }
        return self._results

    def get_results(self):
        return dict(self._results)

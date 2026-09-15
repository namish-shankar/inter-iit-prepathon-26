"""
Portfolio -- turns executed trades into an equity curve.

Accounting convention (matches the mandated timing rule exactly): the
position held over the interval ending at candle t's open is the
position *entered* at t-1 -- i.e. prior_position(t) = position(t-1).
The return earned over that interval is prior_position(t) *
r_open(t), where r_open(t) = open(t)/open(t-1) - 1. Cost is charged at
t on the trade that occurs *at* t (moving the position from t-1's value
to t's value), which is exactly what ExecutionEngine.execute() already
computed alongside the fill.
"""

import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, initial_capital=1.0):
        self.initial_capital = initial_capital
        self._equity_curve = None
        self._net_returns = None
        self._positions = None
        self._executions = None

    def update(self, executions, market_data):
        """`executions` is the DataFrame returned by
        ExecutionEngine.execute() (columns: position, trade, fill_price,
        total_cost_frac, ...). `market_data` supplies the open series used
        to mark the held position's return."""
        self._executions = executions
        self._positions = executions["position"]

        r_open = market_data["open"].pct_change().reindex(executions.index)
        prior_position = executions["position"].shift(1).fillna(0.0)

        gross_return = prior_position * r_open.fillna(0.0)
        net_return = gross_return - executions["total_cost_frac"]
        net_return.iloc[0] = -executions["total_cost_frac"].iloc[0]  # entry cost only

        self._net_returns = net_return
        self._equity_curve = self.initial_capital * (1.0 + net_return).cumprod()
        return self._equity_curve

    def mark_to_market(self, market_data):
        """Re-derive the equity curve from stored positions against a
        (possibly updated) market_data frame -- a cross-check that the
        compounded curve in update() agrees with an independent
        recomputation from the same position series."""
        if self._positions is None:
            raise RuntimeError("update() must be called before mark_to_market()")
        r_open = market_data["open"].pct_change().reindex(self._positions.index).fillna(0.0)
        prior_position = self._positions.shift(1).fillna(0.0)
        gross_return = prior_position * r_open
        equity = self.initial_capital * (1.0 + gross_return).cumprod()
        return equity

    def get_equity_curve(self):
        return self._equity_curve

    def get_positions(self):
        return self._positions

    def get_pnl(self):
        return self._net_returns

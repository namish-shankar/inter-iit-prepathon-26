"""
ExecutionEngine -- turns a target-position series into fills at the
mandated candle-open price, with the mandated transaction cost and a
documented slippage model.

Timing convention (Technical Documentation Sec. 1.2): the `signal`
Series handed to execute() is assumed to already be t-1-safe (the
Backtester enforces this before calling in); execute() itself enforces
the *other* half of the rule -- fills happen at that same candle's OPEN,
never its close.
"""

import numpy as np
import pandas as pd

import config


class ExecutionEngine:
    def __init__(self, cost_per_side=None, slippage_range_fraction=None):
        self.cost_per_side = (
            config.TRANSACTION_COST_PER_SIDE if cost_per_side is None else cost_per_side
        )
        self.slippage_range_fraction = (
            config.SLIPPAGE_RANGE_FRACTION
            if slippage_range_fraction is None
            else slippage_range_fraction
        )
        self._trade_log = []

    # ------------------------------------------------------------ costs --
    def apply_transaction_cost(self, trades):
        """Mandated 0.05% per side, charged on |trade size| (never on the
        held position itself -- a position you don't change costs nothing
        to hold)."""
        return trades.abs() * self.cost_per_side

    def apply_slippage(self, trades, market_data):
        """Slippage modelled as a fraction of that bar's own (high-low)
        range relative to its open, scaled by trade size. This lets
        slippage breathe with realised intraday volatility instead of a
        single flat bp figure that would be too loose on calm bars and
        too tight on wild ones (documented assumption -- see README)."""
        bar_range_frac = (market_data["high"] - market_data["low"]) / market_data["open"]
        bar_range_frac = bar_range_frac.reindex(trades.index).fillna(0.0)
        return trades.abs() * self.slippage_range_fraction * bar_range_frac

    # --------------------------------------------------------------- log --
    def record_trade(self, trade):
        self._trade_log.append(trade)

    def get_trade_log(self):
        if not self._trade_log:
            return pd.DataFrame(
                columns=["date", "trade_delta", "fill_price", "transaction_cost",
                         "slippage_cost", "total_cost_frac"]
            )
        return pd.DataFrame(self._trade_log)

    # ----------------------------------------------------------- execute --
    def execute(self, signal, market_data):
        """`signal` is the (already t-1-safe) target position indexed by
        the candle it trades ON. Returns a DataFrame aligned to
        `market_data.index` with the realised position, the trade delta,
        the fill price (candle open), and the per-candle cost fraction
        (transaction cost + slippage, as a fraction of trade notional).

        A no-op (target position unchanged) is never logged or charged --
        "trading" only happens where the position actually moves.
        """
        position = signal.reindex(market_data.index).fillna(0.0)
        trade = position.diff()
        trade.iloc[0] = position.iloc[0] - 0.0  # entering from flat
        trade = trade.fillna(0.0)

        fill_price = market_data["open"]
        txn_cost = self.apply_transaction_cost(trade)
        slip_cost = self.apply_slippage(trade, market_data)
        total_cost_frac = txn_cost + slip_cost

        out = pd.DataFrame(
            {
                "position": position,
                "trade": trade,
                "fill_price": fill_price,
                "transaction_cost": txn_cost,
                "slippage_cost": slip_cost,
                "total_cost_frac": total_cost_frac,
            },
            index=market_data.index,
        )

        nonzero = out.index[out["trade"] != 0]
        for ts in nonzero:
            self.record_trade(
                {
                    "date": ts,
                    "trade_delta": out.loc[ts, "trade"],
                    "fill_price": out.loc[ts, "fill_price"],
                    "transaction_cost": out.loc[ts, "transaction_cost"],
                    "slippage_cost": out.loc[ts, "slippage_cost"],
                    "total_cost_frac": out.loc[ts, "total_cost_frac"],
                }
            )

        return out

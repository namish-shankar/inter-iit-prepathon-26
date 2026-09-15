"""
PerformanceAnalyzer -- turns a net-return series (plus optional trade
log / position series) into the required performance report.

Every ratio here is guarded against the degenerate cases a real backtest
actually hits: a strategy that never trades in a sub-period, a return
series with zero variance, an empty window after a split. Each guard
returns NaN/0 rather than raising or silently dividing by zero, so a
robustness sweep across sub-periods or cost levels never crashes on the
one fold where a strategy went quiet.
"""

import numpy as np
import pandas as pd

import config


class PerformanceAnalyzer:
    def __init__(self, returns, trade_log=None, positions=None,
                 annualization_factor=None, risk_free_rate=None):
        self.returns = returns.dropna() if returns is not None else pd.Series(dtype=float)
        self.trade_log = trade_log if trade_log is not None else pd.DataFrame()
        self.positions = positions
        self.ann_factor = annualization_factor or config.ANNUALIZATION_FACTOR
        self.rf = config.RISK_FREE_RATE if risk_free_rate is None else risk_free_rate

    # ------------------------------------------------------------- core --
    def _equity_curve(self):
        if len(self.returns) == 0:
            return pd.Series(dtype=float)
        return (1.0 + self.returns).cumprod()

    def total_return(self):
        if len(self.returns) == 0:
            return 0.0
        return float((1.0 + self.returns).prod() - 1.0)

    def annualized_return(self):
        if len(self.returns) == 0:
            return 0.0
        total = self.total_return()
        n = len(self.returns)
        return float((1.0 + total) ** (self.ann_factor / n) - 1.0)

    def volatility(self):
        if len(self.returns) < 2:
            return 0.0
        return float(self.returns.std() * np.sqrt(self.ann_factor))

    def sharpe_ratio(self):
        if len(self.returns) < 2:
            return 0.0
        std = self.returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        excess = self.returns.mean() - self.rf / self.ann_factor
        return float(np.sqrt(self.ann_factor) * excess / std)

    def sortino_ratio(self):
        if len(self.returns) < 2:
            return 0.0
        downside = self.returns[self.returns < 0]
        if len(downside) == 0 or downside.std() == 0 or np.isnan(downside.std()):
            return 0.0
        excess = self.returns.mean() - self.rf / self.ann_factor
        return float(np.sqrt(self.ann_factor) * excess / downside.std())

    def max_drawdown(self):
        eq = self._equity_curve()
        if len(eq) == 0:
            return 0.0
        peak = eq.cummax()
        dd = (eq - peak) / peak
        return float(dd.min()) if len(dd) else 0.0

    def drawdown_duration(self):
        """Longest run (in bars) spent strictly below a prior equity
        peak. Returns 0 for a strategy that never drew down (e.g. a flat
        strategy with zero trades in a sub-period) instead of raising."""
        eq = self._equity_curve()
        if len(eq) == 0:
            return 0
        peak = eq.cummax()
        underwater = eq < peak
        if not underwater.any():
            return 0
        run = 0
        longest = 0
        for flag in underwater:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return int(longest)

    def calmar_ratio(self):
        mdd = self.max_drawdown()
        if mdd == 0:
            return np.nan
        return float(self.annualized_return() / abs(mdd))

    # ------------------------------------------------------------ trades --
    def trade_statistics(self):
        """Round-trip trade stats built from the position series: a round
        trip is a maximal contiguous run of non-flat position. Guarded to
        return zeros (not raise) when there were no trades in the window
        under analysis."""
        empty = {
            "num_round_trips": 0, "win_rate": 0.0, "avg_win": 0.0,
            "avg_loss": 0.0, "avg_holding_period_bars": 0.0,
            "turnover_per_bar": 0.0, "total_cost_paid": 0.0,
        }
        if self.positions is None or len(self.positions) == 0:
            return empty

        turnover = float(self.positions.diff().abs().fillna(self.positions.abs()).mean())
        total_cost = float(self.trade_log["total_cost_frac"].sum()) if len(self.trade_log) else 0.0

        in_trade = self.positions != 0
        if not in_trade.any():
            empty["turnover_per_bar"] = turnover
            empty["total_cost_paid"] = total_cost
            return empty

        block_id = (in_trade != in_trade.shift()).cumsum()
        rets = self.returns.reindex(self.positions.index).fillna(0.0)

        round_trip_returns, holding_periods = [], []
        for _, grp in pd.DataFrame({"in_trade": in_trade, "block": block_id}).groupby("block"):
            if not grp["in_trade"].iloc[0]:
                continue
            idx = grp.index
            rt_return = float((1.0 + rets.loc[idx]).prod() - 1.0)
            round_trip_returns.append(rt_return)
            holding_periods.append(len(idx))

        if not round_trip_returns:
            empty["turnover_per_bar"] = turnover
            empty["total_cost_paid"] = total_cost
            return empty

        rr = np.array(round_trip_returns)
        wins = rr[rr > 0]
        losses = rr[rr < 0]
        return {
            "num_round_trips": len(rr),
            "win_rate": float(len(wins) / len(rr)),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "avg_holding_period_bars": float(np.mean(holding_periods)),
            "turnover_per_bar": turnover,
            "total_cost_paid": total_cost,
        }

    # ------------------------------------------------------------ report --
    def report(self):
        return {
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "volatility": self.volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown(),
            "drawdown_duration_bars": self.drawdown_duration(),
            "calmar_ratio": self.calmar_ratio(),
            **self.trade_statistics(),
        }

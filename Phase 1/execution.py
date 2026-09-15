import pandas as pd
import numpy as np

class OHLCVEngine:
    def __init__(self, fee=0.0002):
        self.fee = fee  # 0.02% per side (0.04% round-trip)

    def backtest(self, data, signals):
        # Shift signals by 1 to execute on the next bar's open (avoids look-ahead bias)
        position = signals.shift(1).fillna(0)
        trades = position.diff().fillna(0)
        
        # P&L calculation: return based on close-to-close change
        market_return = data['close'].pct_change()
        gross_return = position * market_return
        
        # Deduct 0.02% per side on position changes
        costs = np.abs(trades) * self.fee 
        net_return = gross_return - costs
        
        equity = (1 + net_return).cumprod()
        return equity, net_return, trades, signals

class OrderBookEngine:
    def __init__(self, fee=0.0002):
        self.fee = fee  # 0.02% per side

    def walk_book(self, order_qty, row):
        """Walks the 3-level LOB depth to compute exact VWAP execution price."""
        if order_qty == 0:
            return row['mid_price']
            
        is_buy = order_qty > 0
        qty_left = abs(order_qty)
        total_cost = 0.0
        
        prefix = 'ask' if is_buy else 'bid'
        
        for i in range(1, 4):
            price = row[f'{prefix}_price_{i}']
            vol = row[f'{prefix}_volume_{i}']
            
            fill_qty = min(qty_left, vol)
            total_cost += fill_qty * price
            qty_left -= fill_qty
            
            if qty_left <= 0:
                break
                
        # If order size exceeds level 3 depth, apply dynamic slippage penalty
        if qty_left > 0:
            penalty_price = price * 1.05 if is_buy else price * 0.95
            total_cost += qty_left * penalty_price
            
        return total_cost / abs(order_qty)

    def backtest(self, data, signals, order_size=100):
        position = signals.shift(1).fillna(0)
        trades = position.diff().fillna(0)
        
        exec_prices = pd.Series(index=data.index, dtype=float)
        
        # Compute VWAP execution for timestamps where position changes occur
        trade_indices = trades[trades != 0].index
        for idx in trade_indices:
            trade_dir = trades.loc[idx]
            exec_prices.loc[idx] = self.walk_book(trade_dir * order_size, data.loc[idx])
            
        market_return = data['mid_price'].pct_change()
        gross_return = position * market_return
        costs = np.abs(trades) * self.fee
        
        net_return = gross_return - costs
        equity = (1 + net_return).cumprod()
        
        return equity, net_return, trades, signals
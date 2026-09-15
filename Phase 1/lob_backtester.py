import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Metrics:
    @staticmethod
    def compute(returns, trades):
        total_ret = (1 + returns).prod() - 1
        ann_ret = (1 + total_ret) ** (252 / max(len(returns), 1)) - 1
        roll_vol = returns.rolling(252).std() * np.sqrt(252)
        sharpe = (np.sqrt(252) * returns.mean() / returns.std()) if returns.std() != 0 else 0.0
        downside = returns[returns < 0]
        sortino = (np.sqrt(252) * returns.mean() / downside.std()) if len(downside) > 0 and downside.std() != 0 else 0.0
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()
        calmar = (ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
        active_returns = returns[returns != 0]
        win_rate = (len(returns[returns > 0]) / len(active_returns)) if len(active_returns) > 0 else 0.0
        num_trades = len(trades[trades != 0])

        return {
            "Total Return": round(total_ret, 4),
            "Annualized Return": round(ann_ret, 4),
            "Sharpe Ratio": round(sharpe, 4),
            "Sortino Ratio": round(sortino, 4),
            "Max Drawdown": round(max_dd, 4),
            "Calmar Ratio": round(calmar, 4),
            "Win Rate": round(win_rate, 4),
            "Total Trades": num_trades
        }

    @staticmethod
    def plot_performance(returns, title="Backtest"):
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(cum_ret.index, cum_ret, color='blue', label='Equity Curve')
        ax1.set_title(f'{title} Performance')
        ax1.set_ylabel('Cumulative Portfolio Multiplier')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        filename = f"{title.replace(' ', '_').lower()}_chart.png"
        plt.savefig(filename)
        print(f"Saved visualization to: {filename}")
        plt.close()

def check_lookahead_bias_robust(signal_func, data, *args):
    orig_signals = signal_func(data, *args)
    mid_idx = int(len(data) * 0.7)
    data_pert = data.copy()
    target_col = 'close' if 'close' in data_pert.columns else 'mid_price'
    data_pert.iloc[mid_idx, data_pert.columns.get_loc(target_col)] *= 2.0
    pert_signals = signal_func(data_pert, *args)
    past_orig = orig_signals.iloc[:mid_idx].fillna(0).values
    past_pert = pert_signals.iloc[:mid_idx].fillna(0).values
    bias_detected = not np.array_equal(past_orig, past_pert)
    return bias_detected

def check_execution_lookahead_bias(signals: pd.Series, trades: pd.Series) -> bool:
    expected_positions = signals.shift(1).fillna(0)
    expected_trades = expected_positions.diff().fillna(0).to_numpy()
    actual_trades = trades.fillna(0).to_numpy()
    bias_detected = not np.array_equal(expected_trades, actual_trades)
    return bias_detected

class DataProcessor:
    def process_lob(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.ffill()
        if 'mid_price' not in df.columns:
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2.0
        df['simple_return'] = df['mid_price'].pct_change()
        df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
        return df.dropna()

class MovingAverageCrossover:
    def __init__(self, short_window=20, long_window=50, target_col='close'):
        self.short = short_window
        self.long = long_window
        self.col = target_col

    def generate_signals(self, df):
        ma_short = df[self.col].rolling(window=self.short).mean()
        ma_long = df[self.col].rolling(window=self.long).mean()
        signals = pd.Series(0, index=df.index)
        signals[ma_short > ma_long] = 1
        signals[ma_short < ma_long] = -1
        return signals

class OrderBookEngine:
    def __init__(self, fee=0.0002):
        self.fee = fee

    def walk_book(self, order_qty, row):
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

        if qty_left > 0:
            penalty_price = price * 1.05 if is_buy else price * 0.95
            total_cost += qty_left * penalty_price

        return total_cost / abs(order_qty)

    def backtest(self, data, signals, order_size=100):
        position = signals.shift(1).fillna(0)
        trades = position.diff().fillna(0)
        exec_prices = pd.Series(index=data.index, dtype=float)
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

if __name__ == "__main__":
    print("--- Running LOB Backtest ---")
    processor = DataProcessor()
    data = processor.process_lob('orderbook_train.csv')
    
    strategy = MovingAverageCrossover(target_col='mid_price')
    signals = strategy.generate_signals(data)
    
    print(f"Strategy Bias: {check_lookahead_bias_robust(strategy.generate_signals, data)}")
    
    engine = OrderBookEngine()
    equity, net_return, trades, signals = engine.backtest(data, signals)
    
    print(f"Execution Bias: {check_execution_lookahead_bias(signals, trades)}")
    print(Metrics.compute(net_return, trades))
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
    def process_ohlcv(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.sort_values('date').drop_duplicates(subset=['date'])
        df.set_index('date', inplace=True)
        df = df.ffill()
        df['simple_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
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

class OHLCVEngine:
    def __init__(self, fee=0.0002):
        self.fee = fee

    def backtest(self, data, signals):
        position = signals.shift(1).fillna(0)
        trades = position.diff().fillna(0)
        market_return = data['close'].pct_change()
        gross_return = position * market_return
        costs = np.abs(trades) * self.fee
        net_return = gross_return - costs
        equity = (1 + net_return).cumprod()
        return equity, net_return, trades, signals

if __name__ == "__main__":
    print("--- Running OHLCV Backtest ---")
    processor = DataProcessor()
    data = processor.process_ohlcv('ohlcv_train.csv')
    
    strategy = MovingAverageCrossover(target_col='close')
    signals = strategy.generate_signals(data)
    f
    print(f"Strategy Bias: {check_lookahead_bias_robust(strategy.generate_signals, data)}")
    
    engine = OHLCVEngine()
    equity, net_return, trades, signals = engine.backtest(data, signals)
    
    print(f"Execution Bias: {check_execution_lookahead_bias(signals, trades)}")
    print(Metrics.compute(net_return, trades))
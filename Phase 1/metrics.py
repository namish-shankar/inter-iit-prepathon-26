import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        """Generates and saves the equity curve and drawdown visualizations."""
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity Curve
        ax1.plot(cum_ret.index, cum_ret, color='blue', label='Equity Curve')
        ax1.set_title(f'{title} Performance')
        ax1.set_ylabel('Cumulative Portfolio Multiplier')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Drawdown
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

def check_lookahead_bias(signal_func, data, *args):
    """
    Robust check: Perturbs a random point in the MIDDLES of the dataset 
    to prevent local window leaks at row N-1 from slipping through.
    """
    orig_signals = signal_func(data, *args)
    
    # Pick a random midpoint index (e.g., 70% through the dataset)
    mid_idx = int(len(data) * 0.7)
    
    data_pert = data.copy()
    target_col = 'close' if 'close' in data_pert.columns else 'mid_price'
    
    # Perturb the midpoint price
    data_pert.iloc[mid_idx, data_pert.columns.get_loc(target_col)] *= 2.0
    
    pert_signals = signal_func(data_pert, *args)
    
    # Check if ANY signal strictly BEFORE mid_idx changed
    past_orig = orig_signals.iloc[:mid_idx].fillna(0).values
    past_pert = pert_signals.iloc[:mid_idx].fillna(0).values
    
    bias_detected = not np.array_equal(past_orig, past_pert)
    return bias_detected
def check_execution_lookahead_bias(signals: pd.Series, trades: pd.Series) -> bool:
    """
    Checks if trade execution violates temporal causality.
    Handles high-frequency whipsaws by strictly comparing against expected t+1 deltas.
    """
    # 1. Mathematically construct what a perfectly unbiased trade array MUST be:
    # Shift signals by 1 (t+1 delay), then take the diff to find the trade triggers.
    expected_positions = signals.shift(1).fillna(0)
    expected_trades = expected_positions.diff().fillna(0).to_numpy()
    
    # 2. Get the actual trades your engine executed
    actual_trades = trades.fillna(0).to_numpy()
    
    # 3. If they don't match perfectly, the engine altered the execution timing
    bias_detected = not np.array_equal(expected_trades, actual_trades)
    
    return bias_detected

'''
The `check_lookahead_bias()` function checks whether changing a price at a later point in the dataset unexpectedly 
changes signals generated before that point, which would indicate that the strategy is using future information. 
It runs the strategy on the original and perturbed data, compares only the signals before the modified index, 
and returns `True` if any of them differ. The `check_execution_lookahead_bias()` function separately checks whether 
trades are executed with the correct timing by shifting signals by one period to create the expected trades, then
comparing them with the actual trades produced by the backtester. If the expected and actual trades differ, 
the execution timing may contain look-ahead bias.
'''
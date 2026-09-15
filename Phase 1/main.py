from data_processor import DataProcessor
from strategy import MovingAverageCrossover
from execution import OHLCVEngine, OrderBookEngine
from metrics import Metrics, check_lookahead_bias, check_execution_lookahead_bias

def run():
    processor = DataProcessor()
    
    # --- 1. OHLCV Pipeline ---
    print("--- Running OHLCV Engine ---")
    ohlcv_data = processor.process_ohlcv('ohlcv_train.csv')
    ohlcv_strat = MovingAverageCrossover(target_col='close')
    ohlcv_signals = ohlcv_strat.generate_signals(ohlcv_data)
    
    # Strategy Bias Check
    strat_bias = check_lookahead_bias(ohlcv_strat.generate_signals, ohlcv_data)
    print(f"Strategy Signal Bias: {strat_bias}")
    
    # Run Backtest
    ohlcv_engine = OHLCVEngine()
    ohlcv_equity, ohlcv_ret, ohlcv_trades, ohlcv_signals = ohlcv_engine.backtest(ohlcv_data, ohlcv_signals)
    
    # Execution Timing Bias Check
    exec_bias = check_execution_lookahead_bias(ohlcv_signals, ohlcv_trades)
    print(f"Execution Timing Bias: {exec_bias}")
    
    print(Metrics.compute(ohlcv_ret, ohlcv_trades))

    # --- 2. Order Book Pipeline ---
    print("\n--- Running LOB Engine ---")
    lob_data = processor.process_lob('orderbook_train.csv')
    lob_strat = MovingAverageCrossover(target_col='mid_price')
    lob_signals = lob_strat.generate_signals(lob_data)
    
    # Strategy Bias Check
    lob_strat_bias = check_lookahead_bias(lob_strat.generate_signals, lob_data)
    print(f"Strategy Signal Bias: {lob_strat_bias}")
    
    # Run Backtest
    lob_engine = OrderBookEngine()
    lob_equity, lob_ret, lob_trades, lob_signals = lob_engine.backtest(lob_data, lob_signals)
    
    # Execution Timing Bias Check
    lob_exec_bias = check_execution_lookahead_bias(lob_signals, lob_trades)
    print(f"Execution Timing Bias: {lob_exec_bias}")
    
    print(Metrics.compute(lob_ret, lob_trades))

if __name__ == "__main__":
    run()
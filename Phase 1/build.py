import os
import re

# The modular files that contain your classes
CORE_FILES = ['metrics.py', 'data_processor.py', 'strategy.py', 'execution.py']
LOCAL_MODULES = ['metrics', 'data_processor', 'strategy', 'execution']

def is_local_import(line: str) -> bool:
    """Checks if a line is importing one of your local files."""
    for mod in LOCAL_MODULES:
        # Matches 'import metrics' or 'from metrics import ...'
        if re.match(rf'^\s*(from|import)\s+{mod}\b', line):
            return True
    return False

def extract_code():
    external_imports = set()
    body_lines = []

    for file in CORE_FILES:
        if not os.path.exists(file):
            print(f"Warning: {file} not found. Skipping.")
            continue
            
        with open(file, 'r') as f:
            body_lines.append(f"\n# {'='*40}\n# From: {file}\n# {'='*40}\n")
            for line in f:
                # Catch external imports to hoist to the top
                if (line.startswith('import ') or line.startswith('from ')) and not is_local_import(line):
                    external_imports.add(line.strip())
                # Ignore local imports and empty lines at the top
                elif not is_local_import(line):
                    body_lines.append(line.rstrip())

    return "\n".join(sorted(external_imports)) + "\n" + "\n".join(body_lines)

# --- Define the specific run blocks for each file ---

OHLCV_MAIN = """
# ==========================================
# MAIN EXECUTION BLOCK (OHLCV)
# ==========================================
if __name__ == "__main__":
    print("--- Running OHLCV Backtest ---")
    processor = DataProcessor()
    data = processor.process_ohlcv('ohlcv_train.csv')
    
    strategy = MovingAverageCrossover(target_col='close')
    signals = strategy.generate_signals(data)
    
    print(f"Strategy Bias: {check_lookahead_bias_robust(strategy.generate_signals, data)}")
    
    engine = OHLCVEngine()
    equity, net_return, trades, signals = engine.backtest(data, signals)
    
    print(f"Execution Bias: {check_execution_lookahead_bias(signals, trades)}")
    print(Metrics.compute(net_return, trades))
"""

LOB_MAIN = """
# ==========================================
# MAIN EXECUTION BLOCK (LIMIT ORDER BOOK)
# ==========================================
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
"""

def generate_files():
    base_code = extract_code()
    
    # 1. Build OHLCV File
    with open('ohlcv_backtester.py', 'w') as f:
        f.write(base_code + "\n" + OHLCV_MAIN)
    print("Successfully generated ohlcv_backtester.py")
    
    # 2. Build LOB File
    with open('lob_backtester.py', 'w') as f:
        f.write(base_code + "\n" + LOB_MAIN)
    print("Successfully generated lob_backtester.py")

if __name__ == "__main__":
    generate_files()
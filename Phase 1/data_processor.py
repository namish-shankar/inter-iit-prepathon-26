import pandas as pd
import numpy as np

class DataProcessor:
    def process_ohlcv(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower()
        
        # Parse dates and sort chronologically
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.sort_values('date').drop_duplicates(subset=['date'])
        df.set_index('date', inplace=True)
        
        # Forward fill missing bars
        df = df.ffill()
        
        # Calculate simple and log returns
        df['simple_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        return df.dropna()

    def process_lob(self, filepath):
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower()
        
        # Parse timestamps and sort chronologically
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        df = df.ffill()
        
        # Ensure mid_price exists or compute if missing
        if 'mid_price' not in df.columns:
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2.0
            
        df['simple_return'] = df['mid_price'].pct_change()
        df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))
        return df.dropna()
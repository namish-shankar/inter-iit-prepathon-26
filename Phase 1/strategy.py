import pandas as pd

class MovingAverageCrossover:
    def __init__(self, short_window=20, long_window=50, target_col='close'):
        self.short = short_window
        self.long = long_window
        self.col = target_col

    def generate_signals(self, df):
        ma_short = df[self.col].rolling(window=self.short).mean()
        ma_long = df[self.col].rolling(window=self.long).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[ma_short > ma_long] = 1  # Long
        signals[ma_short < ma_long] = -1 # Short
        
        return signals
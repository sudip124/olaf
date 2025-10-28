import vectorbt as vbt
import pandas as pd

def generate_signals(price: pd.Series, fast_window=10, slow_window=20) -> tuple:
    """
    Trend following: Buy on fast MA > slow MA, sell on cross below.
    Returns entries (buy signals), exits (sell signals) as boolean Series.
    """
    fast_ma = vbt.MA.run(price, fast_window)
    slow_ma = vbt.MA.run(price, slow_window)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return entries, exits
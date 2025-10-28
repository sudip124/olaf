import vectorbt as vbt
import pandas as pd

def generate_signals(price: pd.Series, window=20, std_dev=2) -> tuple:
    """
    Mean reversion: Buy when price < lower band, sell when > upper band.
    Returns entries (buy), exits (sell) as boolean Series.
    """
    bbands = vbt.BBANDS.run(price, window=window, nstd=std_dev)
    entries = price < bbands.lower  # Oversold: buy
    exits = price > bbands.upper    # Overbought: sell
    return entries, exits
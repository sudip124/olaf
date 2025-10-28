import vectorbt as vbt
import pandas as pd
from backtest_config import SYMBOL, INTERVAL, INIT_CASH, FEES, EXCHANGE
from data_fetcher import fetch_historical_data
from strategies.trend_following import generate_signals as trend_signals
# from strategies.mean_reversion import generate_signals as mr_signals  # Swap for mean reversion

# Fetch data (adjust dates for longer periods if needed; current is ~2 months)
df = fetch_historical_data(SYMBOL, INTERVAL, '2025-08-01', '2025-10-09', exchange=EXCHANGE)
print(df.head())  # Verify data
price = df['close']

# Single-run backtest (fixed params; this is working)
entries, exits = trend_signals(price, fast_window=5, slow_window=15)  # Or mr_signals(price, window=20, std_dev=2)
pf = vbt.Portfolio.from_signals(
    price,
    entries,
    exits,
    init_cash=INIT_CASH,
    fees=FEES,
    freq=INTERVAL.replace('m', 'min'),  # Use 'min' to fix FutureWarning (e.g., '15min')
    size=1.0,  # Full cash allocation
    direction='longonly'  # For intraday swings
)
print(pf.stats())  # Your stats output

# Parameter optimization (free VectorBT workaround: lists + run_combs)
# Define param lists for sweeps (tune for intraday: shorter for swings)
fast_windows = list(range(5, 30, 5))  # [5, 10, 15, 20, 25]
slow_windows = list(range(15, 40, 5))  # [15, 20, 25, 30, 35]; ensure slow > fast in logic

# Run MAs with combos (run_combs returns tuple of MAs)
fast_ma, slow_ma = vbt.MA.run_combs(price, fast_windows, slow_windows, short_names=['fast', 'slow'])

# Generate signals across all param combos
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

# Create optimized portfolio
pf_opt = vbt.Portfolio.from_signals(
    price,
    entries,
    exits,
    init_cash=INIT_CASH,
    fees=FEES,
    freq=INTERVAL.replace('m', 'min'),  # Fix warning
    size=1.0,
    direction='longonly'
)

# total_return() is now a MultiIndex Series (fast x slow)
print(pf_opt.total_return())  # Debug: See returns matrix

# Heatmap visualization (works on MultiIndex)
fig = pf_opt.total_return().vbt.heatmap(
    x_level='slow_window',  # Columns
    y_level='fast_window'   # Rows
)
fig.show()  # Interactive Plotly; or fig.write_image('logs/heatmap.png')

# Optional: Best params (e.g., max return)
best_params = pf_opt.total_return().idxmax()
print(f"Best params: {best_params} with return {pf_opt.total_return().max()}")
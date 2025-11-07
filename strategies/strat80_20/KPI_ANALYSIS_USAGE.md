# Backtest KPI Analysis for Live Trading Readiness

## Overview
After running a backtest, use the KPI analysis functions to evaluate which symbols are ready for live trading based on risk-adjusted performance metrics.

## KPIs Calculated

1. **Sharpe Ratio** - Annualized risk-adjusted return (higher is better, >1.0 is good)
2. **Sortino Ratio** - Risk-adjusted return using only downside deviation (focuses on bad volatility)
3. **Total Trades** - Number of completed trades in the backtest period
4. **Win Rate (%)** - Percentage of winning trades
5. **Median Win Return** - Median PnL of winning trades
6. **Median Loss Return** - Median PnL of losing trades  
7. **Top Weekday** - Best performing weekday with win count and total PnL
8. **Win Rate Last 30D (%)** - Recent performance indicator
9. **Trades Last 30D** - Number of trades in the last 30 days

## Usage

### Method 1: Command Line (Recommended)

After running a backtest, analyze the results:

```bash
# Basic analysis
python strategies/strat80_20/backtest_analysis.py --backtest-id 42

# With custom filtering criteria
python strategies/strat80_20/backtest_analysis.py --backtest-id 42 \
    --min-sharpe 1.5 \
    --min-winrate 65 \
    --min-trades 25
```

### Method 2: Python Code

```python
from strategies.strat80_20.backtest_analysis import analyze_backtest_for_live_readiness

# Run complete analysis
backtest_id = 42
kpis = analyze_backtest_for_live_readiness(backtest_id)

# Filter for live-ready symbols
live_ready = kpis[
    (kpis['Sharpe Ratio'] > 1.0) & 
    (kpis['Win Rate (%)'] > 60) & 
    (kpis['Total Trades'] >= 20)
]

print(f"Symbols ready for live: {live_ready['Symbol'].tolist()}")
```

### Method 3: Separate Steps

```python
from strategies.strat80_20.backtest_analysis import (
    build_backtest_journal, 
    calculate_symbol_kpis
)

# Step 1: Build the journal
backtest_id = 42
trades = build_backtest_journal(backtest_id)
print(f"Processed {trades} trades")

# Step 2: Calculate KPIs
kpis = calculate_symbol_kpis(backtest_id)
print(kpis)
```

## Example Output

```
================================================================================
KPI ANALYSIS FOR BACKTEST ID: 42
================================================================================

 Symbol  Total Trades  Win Rate (%)  Sharpe Ratio  Sortino Ratio  Median Win Return  Median Loss Return         Top Weekday  Win Rate Last 30D (%)  Trades Last 30D
SBIN           45          75.56          2.34           3.12              125.50             -45.30  Monday (15W, ₹1850.0)                  72.22               18
HDFC           38          68.42          1.89           2.45               98.75             -52.10   Friday (12W, ₹1450.0)                  66.67               15
ICICI          42          71.43          1.67           2.18              110.20             -48.90  Tuesday (14W, ₹1620.0)                  70.00               20

================================================================================
```

## Recommended Filtering Criteria

For **conservative live trading**:
- Sharpe Ratio >= 1.5
- Sortino Ratio >= 2.0
- Win Rate >= 65%
- Total Trades >= 30
- Win Rate Last 30D >= 60%

For **moderate risk**:
- Sharpe Ratio >= 1.0
- Sortino Ratio >= 1.5
- Win Rate >= 60%
- Total Trades >= 20
- Win Rate Last 30D >= 55%

For **aggressive** (not recommended):
- Sharpe Ratio >= 0.5
- Win Rate >= 55%
- Total Trades >= 15

## Notes

- **Sharpe Ratio** is annualized assuming 252 trading days
- **Sortino Ratio** is preferred over Sharpe as it only penalizes downside volatility
- **Median Returns** are more robust than mean returns (less affected by outliers)
- **Top Weekday** helps identify patterns for position sizing or timing
- **Last 30D metrics** help identify if recent performance is deteriorating

## Integration with Live Trading

After identifying live-ready symbols, update your `scanner_setups.json` or trading configuration:

```json
{
  "live_symbols": ["SBIN", "HDFC", "ICICI"],
  "filters_applied": {
    "min_sharpe": 1.0,
    "min_winrate": 60,
    "min_trades": 20
  },
  "analysis_date": "2025-01-15",
  "backtest_id": 42
}
```

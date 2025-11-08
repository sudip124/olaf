# SQL Query Examples for Centralized Analysis Database

This file contains useful SQL queries for analyzing your trading strategies using the centralized database.

## Prerequisites

```python
import sqlite3
import pandas as pd

# Connect to centralized database
conn = sqlite3.connect('db/nse.db')
```

---

## 📊 Basic Queries

### View all strategies in database
```sql
SELECT DISTINCT strategy_name 
FROM backtest_journal 
ORDER BY strategy_name;
```

### Latest backtest IDs per strategy
```sql
SELECT strategy_name, MAX(backtest_id) as latest_backtest_id
FROM backtest_journal
GROUP BY strategy_name;
```

### Trade count per strategy
```sql
SELECT 
    strategy_name,
    COUNT(*) as total_trades,
    SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN status = 'lose' THEN 1 ELSE 0 END) as losses
FROM backtest_journal
GROUP BY strategy_name;
```

---

## 🏆 Cross-Strategy Comparison

### Top symbols across all strategies
```sql
SELECT 
    k.strategy_name,
    k.symbol,
    k.composite_score,
    k.sharpe_ratio,
    k.win_rate,
    k.total_trades
FROM kpi_scores k
WHERE k.backtest_id IN (
    SELECT MAX(backtest_id) 
    FROM kpi_scores 
    GROUP BY strategy_name
)
ORDER BY k.composite_score DESC
LIMIT 20;
```

### Best strategy per symbol
```sql
WITH LatestRuns AS (
    SELECT strategy_name, MAX(backtest_id) as backtest_id
    FROM kpi_scores
    GROUP BY strategy_name
)
SELECT 
    k.symbol,
    k.strategy_name as best_strategy,
    k.composite_score,
    k.sharpe_ratio,
    k.win_rate
FROM kpi_scores k
INNER JOIN LatestRuns lr 
    ON k.strategy_name = lr.strategy_name 
    AND k.backtest_id = lr.backtest_id
WHERE k.symbol IN (
    -- Symbols that appear in multiple strategies
    SELECT symbol 
    FROM kpi_scores 
    GROUP BY symbol 
    HAVING COUNT(DISTINCT strategy_name) > 1
)
ORDER BY k.symbol, k.composite_score DESC;
```

### Strategy performance summary
```sql
SELECT 
    strategy_name,
    AVG(composite_score) as avg_score,
    AVG(sharpe_ratio) as avg_sharpe,
    AVG(win_rate) as avg_win_rate,
    AVG(total_trades) as avg_trades,
    COUNT(DISTINCT symbol) as symbols_traded
FROM kpi_scores
WHERE backtest_id IN (
    SELECT MAX(backtest_id) 
    FROM kpi_scores 
    GROUP BY strategy_name
)
GROUP BY strategy_name
ORDER BY avg_score DESC;
```

---

## 📈 Time-Based Analysis

### Daily PnL across all strategies
```sql
SELECT 
    date,
    strategy_name,
    SUM(pnl_absolute) as daily_pnl,
    COUNT(*) as trades,
    AVG(pnl_r) as avg_r
FROM backtest_journal
WHERE status IN ('win', 'lose')
GROUP BY date, strategy_name
ORDER BY date DESC, strategy_name;
```

### Monthly performance comparison
```sql
SELECT 
    strftime('%Y-%m', date) as month,
    strategy_name,
    SUM(pnl_absolute) as monthly_pnl,
    COUNT(*) as trades,
    SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
FROM backtest_journal
WHERE status IN ('win', 'lose')
GROUP BY month, strategy_name
ORDER BY month DESC, strategy_name;
```

### Weekday performance by strategy
```sql
SELECT 
    strategy_name,
    weekday,
    COUNT(*) as trades,
    AVG(pnl_r) as avg_pnl_r,
    SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
FROM backtest_journal
WHERE status IN ('win', 'lose')
GROUP BY strategy_name, weekday
ORDER BY strategy_name, avg_pnl_r DESC;
```

---

## 💰 Risk and Return Analysis

### Risk-adjusted returns by strategy
```sql
SELECT 
    strategy_name,
    AVG(pnl_r) as avg_return_r,
    STDEV(pnl_r) as volatility_r,
    AVG(pnl_r) / NULLIF(STDEV(pnl_r), 0) as sharpe_estimate,
    AVG(risk) as avg_risk_amount
FROM backtest_journal
WHERE status IN ('win', 'lose')
GROUP BY strategy_name
ORDER BY sharpe_estimate DESC;
```

### Drawdown analysis
```sql
SELECT 
    strategy_name,
    symbol,
    MIN(cumulative_pnl) as max_drawdown
FROM (
    SELECT 
        strategy_name,
        symbol,
        date,
        SUM(pnl_r) OVER (
            PARTITION BY strategy_name, symbol 
            ORDER BY date
        ) - MAX(SUM(pnl_r)) OVER (
            PARTITION BY strategy_name, symbol 
            ORDER BY date
        ) as cumulative_pnl
    FROM backtest_journal
    WHERE status IN ('win', 'lose')
)
GROUP BY strategy_name, symbol
HAVING COUNT(*) > 5  -- Only symbols with enough trades
ORDER BY max_drawdown;
```

### Win/Loss distribution
```sql
SELECT 
    strategy_name,
    status,
    COUNT(*) as count,
    AVG(pnl_r) as avg_pnl_r,
    MIN(pnl_r) as min_pnl_r,
    MAX(pnl_r) as max_pnl_r,
    PERCENTILE(pnl_r, 0.5) as median_pnl_r
FROM backtest_journal
WHERE status IN ('win', 'lose')
GROUP BY strategy_name, status
ORDER BY strategy_name, status;
```

---

## 🎯 Strategy Selection Queries

### Symbols where strategy A outperforms strategy B
```python
# Using pandas for easier comparison
query = """
SELECT 
    a.symbol,
    a.composite_score as strat_a_score,
    b.composite_score as strat_b_score,
    a.composite_score - b.composite_score as score_diff
FROM kpi_scores a
INNER JOIN kpi_scores b 
    ON a.symbol = b.symbol
WHERE a.strategy_name = 'strat80_20'
  AND b.strategy_name = 'other_strategy'
  AND a.backtest_id = (SELECT MAX(backtest_id) FROM kpi_scores WHERE strategy_name = 'strat80_20')
  AND b.backtest_id = (SELECT MAX(backtest_id) FROM kpi_scores WHERE strategy_name = 'other_strategy')
  AND a.composite_score > b.composite_score
ORDER BY score_diff DESC;
"""
df = pd.read_sql_query(query, conn)
```

### Best strategy recommendation per symbol
```sql
WITH RankedStrategies AS (
    SELECT 
        k.symbol,
        k.strategy_name,
        k.composite_score,
        k.sharpe_ratio,
        k.win_rate,
        ROW_NUMBER() OVER (PARTITION BY k.symbol ORDER BY k.composite_score DESC) as rank
    FROM kpi_scores k
    WHERE k.backtest_id IN (
        SELECT MAX(backtest_id) 
        FROM kpi_scores 
        GROUP BY strategy_name
    )
)
SELECT 
    symbol,
    strategy_name as recommended_strategy,
    composite_score,
    sharpe_ratio,
    win_rate
FROM RankedStrategies
WHERE rank = 1
ORDER BY composite_score DESC;
```

---

## 🔍 Detailed Trade Analysis

### Recent trades across all strategies
```sql
SELECT 
    date,
    time,
    strategy_name,
    symbol,
    entry_price,
    exit_price,
    pnl_r,
    pnl_absolute,
    status,
    exit_event
FROM backtest_journal
WHERE status IN ('win', 'lose')
ORDER BY date DESC, time DESC
LIMIT 100;
```

### Largest wins and losses
```sql
-- Top 10 wins
SELECT 
    'WIN' as type,
    date,
    strategy_name,
    symbol,
    pnl_r,
    pnl_absolute,
    exit_event
FROM backtest_journal
WHERE status = 'win'
ORDER BY pnl_r DESC
LIMIT 10

UNION ALL

-- Top 10 losses
SELECT 
    'LOSS' as type,
    date,
    strategy_name,
    symbol,
    pnl_r,
    pnl_absolute,
    exit_event
FROM backtest_journal
WHERE status = 'lose'
ORDER BY pnl_r
LIMIT 10;
```

### Strategy vs Symbol correlation
```sql
SELECT 
    j.strategy_name,
    j.symbol,
    COUNT(*) as trades,
    AVG(j.pnl_r) as avg_return,
    SUM(CASE WHEN j.status = 'win' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    k.composite_score
FROM backtest_journal j
LEFT JOIN kpi_scores k 
    ON j.strategy_name = k.strategy_name 
    AND j.symbol = k.symbol
    AND k.backtest_id = (
        SELECT MAX(backtest_id) 
        FROM kpi_scores 
        WHERE strategy_name = j.strategy_name
    )
WHERE j.status IN ('win', 'lose')
GROUP BY j.strategy_name, j.symbol
HAVING COUNT(*) >= 5  -- Minimum trades
ORDER BY j.strategy_name, avg_return DESC;
```

---

## 📊 Python Helper Functions

```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def get_strategy_comparison(db_path='db/nse.db'):
    """Compare all strategies side by side."""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        strategy_name,
        AVG(composite_score) as avg_score,
        AVG(sharpe_ratio) as avg_sharpe,
        AVG(win_rate) as avg_win_rate,
        COUNT(DISTINCT symbol) as symbols
    FROM kpi_scores
    WHERE backtest_id IN (
        SELECT MAX(backtest_id) FROM kpi_scores GROUP BY strategy_name
    )
    GROUP BY strategy_name
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_top_symbols_all_strategies(db_path='db/nse.db', limit=20):
    """Get top performing symbols across all strategies."""
    conn = sqlite3.connect(db_path)
    
    query = f"""
    SELECT 
        strategy_name,
        symbol,
        composite_score,
        sharpe_ratio,
        win_rate,
        total_trades
    FROM kpi_scores
    WHERE backtest_id IN (
        SELECT MAX(backtest_id) FROM kpi_scores GROUP BY strategy_name
    )
    ORDER BY composite_score DESC
    LIMIT {limit}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def plot_strategy_equity_curves(db_path='db/nse.db'):
    """Plot cumulative PnL for each strategy."""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        date,
        strategy_name,
        SUM(pnl_absolute) OVER (
            PARTITION BY strategy_name 
            ORDER BY date
        ) as cumulative_pnl
    FROM backtest_journal
    WHERE status IN ('win', 'lose')
    ORDER BY date, strategy_name
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Plot
    plt.figure(figsize=(12, 6))
    for strategy in df['strategy_name'].unique():
        strategy_df = df[df['strategy_name'] == strategy]
        plt.plot(strategy_df['date'], strategy_df['cumulative_pnl'], 
                label=strategy, marker='o')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.title('Strategy Equity Curves')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Usage examples:
if __name__ == '__main__':
    # Compare strategies
    comparison = get_strategy_comparison()
    print("Strategy Comparison:")
    print(comparison)
    
    # Top symbols
    top_symbols = get_top_symbols_all_strategies(limit=10)
    print("\nTop 10 Symbols:")
    print(top_symbols)
    
    # Plot equity curves
    plot_strategy_equity_curves()
```

---

## 💡 Tips

1. **Use indexes**: The database has indexes on `strategy_name`, `backtest_id`, `symbol`, and `date` for fast queries

2. **Latest runs**: Always filter by latest backtest_id when comparing current performance:
   ```sql
   WHERE backtest_id IN (SELECT MAX(backtest_id) FROM kpi_scores GROUP BY strategy_name)
   ```

3. **Pandas integration**: Use `pd.read_sql_query()` for complex analysis:
   ```python
   df = pd.read_sql_query(query, conn)
   ```

4. **Export results**: Save query results for reports:
   ```python
   df.to_csv('strategy_comparison.csv', index=False)
   ```

---

## 🎯 Next Steps

- Customize these queries for your specific needs
- Create dashboards using tools like Streamlit or Plotly Dash
- Build automated reports that run after each backtest
- Use these queries in portfolio optimization algorithms

# Analysis Module - Generic Backtest Analysis Tools

This module contains strategy-agnostic classes for backtest journal creation and KPI calculation.

## Overview

The analysis module provides three main components:

1. **`backtest_journal_base.py`** - Abstract base class for journal builders
2. **`backtest_journal_model.py`** - Factory function for creating BacktestJournal models
3. **`backtest_kpi_calculator.py`** - Generic KPI calculator for backtest analysis

## Usage

### 1. Creating a Journal Builder for Your Strategy

```python
from analysis.backtest_journal_base import BacktestJournalBuilder
from analysis.backtest_journal_model import create_backtest_journal_model
from strategies.your_strategy import db_models

class YourJournalBuilder(BacktestJournalBuilder):
    def __init__(self, strategy_name: str = 'your_strategy'):
        super().__init__(strategy_name)
        self.db_models = db_models
        # Create BacktestJournal model using your strategy's Base
        self.BacktestJournal = create_backtest_journal_model(db_models.Base)
        self.engine = db_models.get_db_engine(strategy_name)
        db_models.Base.metadata.create_all(self.engine)
    
    def build(self, backtest_run_id: int | None = None) -> int:
        # Implement your journal building logic
        pass
```

### 2. Using the KPI Calculator

```python
from analysis.backtest_kpi_calculator import BacktestCalculator
from analysis.backtest_journal_model import create_backtest_journal_model
from strategies.your_strategy import db_models

# Create the BacktestJournal model for your strategy
BacktestJournal = create_backtest_journal_model(db_models.Base)

# Initialize the calculator
calculator = BacktestCalculator(
    db_models_module=db_models,
    journal_model_class=BacktestJournal,
    strategy_name='your_strategy'
)

# Calculate KPIs
kpi_df = calculator.calculate_symbol_kpis(backtest_run_id=42)

# Save to database
calculator.save_kpi_scores_to_db(backtest_run_id=42, kpi_df=kpi_df)

```

### 3. Integration with run.py

The `run.py` script automatically uses the generic `BacktestJournalBuilder` from the analysis module:

```python
from analysis.backtest_journal_base import BacktestJournalBuilder
from strategies.your_strategy.journal_builder import YourJournalBuilder

# In do_backtest():
journal_builder: BacktestJournalBuilder = YourJournalBuilder(strategy_name='your_strategy')
inserted = journal_builder.build(backtest_run_id)
```

## Requirements for Strategy db_models Module

Your strategy's `db_models` module must provide:

- **`Base`** - SQLAlchemy declarative base
- **`get_db_engine(strategy_name)`** - Function to get database engine
- **`BacktestRun`** - Model for backtest run metadata
- **`TradeLog`** - Model for trade log entries
- **`KPIScore`** - Model for KPI scores (with specific columns)

## KPI Metrics Calculated

The `BacktestCalculator` computes the following metrics:

- **Sharpe Ratio** - Risk-adjusted return (annualized)
- **Sortino Ratio** - Risk-adjusted return using downside deviation (annualized)
- **Win Rate** - Percentage of winning trades
- **Win/Loss Ratio** - Ratio of winning to losing trades
- **Success Rate** - Percentage of setups that led to entries
- **Miss Rate** - Percentage of missed opportunities
- **Max Drawdown (R)** - Maximum peak-to-trough drawdown in R multiples
- **Median Win/Loss (R)** - Median PnL for wins and losses
- **Composite Score** - Weighted score combining multiple metrics
- **Top Weekday** - Best performing day of the week
- **Recent Performance** - Win rate and trade count for last 30 days

## Example: strat80_20

See `strategies/strat80_20/journal_builder_8020.py` for a complete implementation example.

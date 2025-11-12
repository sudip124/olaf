"""
Base ABC class for backtest strategies.

All backtest strategies should inherit from BacktestStrategy and implement the run_backtest method.
This provides a standard interface for run.py to execute any strategy's backtest.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class BacktestStrategy(ABC):
    """
    Abstract base class for backtest strategies.
    
    Each strategy should implement run_backtest() which returns a tuple of (journal_df, backtest_run_id).
    The journal_df should conform to the schema expected by AnalysisService.
    """
    
    def __init__(self, strategy_name: str):
        """
        Initialize the backtest strategy.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'strat80_20')
        """
        self.strategy_name = strategy_name
    
    @abstractmethod
    def run_backtest(
        self,
        symbols: List[str],
        from_date: str,
        to_date: str,
        scan_interval: str = 'D',
        backtest_interval: str = '15m',
        optimize: bool = False,
        max_attempts: Optional[int] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Optional[int]]:
        """
        Run backtest for the strategy.
        
        Args:
            symbols: List of symbols to backtest
            from_date: Start date for backtest (ISO format)
            to_date: End date for backtest (ISO format)
            scan_interval: Interval for setup day detection (default: 'D' for daily)
            backtest_interval: Interval for actual backtest (default: '15m')
            optimize: Whether to run parameter optimization (default: False)
            max_attempts: Maximum number of attempts per day (default: None for unlimited)
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            Tuple of (journal_df, backtest_run_id)
            - journal_df: DataFrame with trade journal conforming to AnalysisService schema
            - backtest_run_id: Database ID for the backtest run (or None if not saved)
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return self.strategy_name

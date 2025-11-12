"""
Base ABC class for live trading strategies.

All live trading strategies should inherit from LiveStrategy and implement the start_trading method.
This provides a standard interface for run.py to execute any strategy's live trading.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class LiveStrategy(ABC):
    """
    Abstract base class for live trading strategies.
    
    Each strategy should implement start_trading() which manages the live trading session.
    """
    
    def __init__(self, strategy_name: str):
        """
        Initialize the live trading strategy.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'strat80_20')
        """
        self.strategy_name = strategy_name
    
    @abstractmethod
    def start_trading(
        self,
        setups_df: pd.DataFrame,
        timezone_str: str = 'Asia/Kolkata',
        market_open_hour: int = 9,
        market_open_minute: int = 15,
        market_close_hour: int = 15,
        market_close_minute: int = 30,
        fixed_qty: int = 1,
        product_type: str = 'MIS',
        order_validity: str = 'DAY',
        max_attempts: Optional[int] = None,
        max_order_retries: Optional[int] = 3,
        **kwargs
    ) -> None:
        """
        Start live trading with given setups and parameters.
        
        Args:
            setups_df: DataFrame with setup information (columns depend on strategy)
            timezone_str: Timezone string (default: 'Asia/Kolkata')
            market_open_hour: Market open hour (default: 9)
            market_open_minute: Market open minute (default: 15)
            market_close_hour: Market close hour (default: 15)
            market_close_minute: Market close minute (default: 30)
            fixed_qty: Fixed quantity per trade (default: 1)
            product_type: Product type for orders (default: 'MIS')
            order_validity: Order validity (default: 'DAY')
            max_attempts: Maximum number of entry attempts per symbol per day (default: None)
            max_order_retries: Maximum retry attempts for failed orders (default: 3)
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            None (runs until market close or interrupted)
        """
        pass
    
    @abstractmethod
    def get_setup_days(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = 'D',
        **kwargs
    ) -> pd.DataFrame:
        """
        Scan for setup days for a given symbol.
        
        Args:
            symbol: Symbol to scan
            from_date: Start date for scan (ISO format)
            to_date: End date for scan (ISO format)
            interval: Scan interval (default: 'D' for daily)
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            DataFrame with setup information (schema depends on strategy)
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Get the strategy name."""
        return self.strategy_name

from abc import ABC, abstractmethod
import pandas as pd


class BacktestJournalBuilder(ABC):
    """
    Abstract base class for backtest journal builders.
    
    This defines the interface that all journal builders must implement.
    Subclasses should implement the build() method to process trade logs
    and create a journal DataFrame specific to their strategy.
    """
    
    def __init__(self, strategy_name: str):
        """
        Initialize the journal builder.
        
        Args:
            strategy_name: Name of the strategy
        """
        self.strategy_name = strategy_name
    
    @abstractmethod
    def build(self, backtest_run_id: int | None = None) -> pd.DataFrame:
        """
        Build the backtest journal DataFrame from trade logs.
        
        Args:
            backtest_run_id: ID of the backtest run to process. If None, uses the latest run.
            
        Returns:
            DataFrame with journal entries containing required columns:
            - date, time, weekday, signal_date
            - symbol, qty
            - initial_stoploss, final_stoploss, entry_price, exit_price
            - risk, pnl_r, pnl_absolute
            - status, exit_event
            - total_setups, total_entries, total_missed (setup metrics per symbol)
            
        Note:
            The journal DataFrame must be self-contained with all data required
            by AnalysisService for KPI calculation. This includes setup-related
            metrics that enable calculation of success rates and miss rates.
        """
        pass

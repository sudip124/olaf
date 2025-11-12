from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Scanner(ABC):
    """
    Abstract base class for strategy scanners.

    Contract: implement get_setup_days() that returns a pandas DataFrame with
    columns expected by downstream components, e.g.:
      ['symbol','setup_date','entry_date','entry_price','trigger_price','tick_size','true_range']
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

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
        Return a DataFrame of setup-day rows for the given symbol and date window.
        """
        raise NotImplementedError

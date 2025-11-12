import os
import json
import numpy as np
import pandas as pd
from strategies.base_scanner import Scanner
from strategies.strat80_20.backtest_config import EXCHANGE
from data_manager.data_fetcher import fetch_historical_data, fetch_instrument_info

# Load config params from strat_config.json
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'strat_config.json')
with open(_CONFIG_PATH, 'r') as _f:
    _params = json.load(_f)
_OPEN_POS_THRESHOLD = _params.get('open_pos_threshold', 0.8)
_CLOSE_POS_THRESHOLD = _params.get('close_pos_threshold', 0.2)
_TRIGGER_TICK_MULT = _params.get('trigger_tick_mult', 10)

class Scanner8020(Scanner):
    """Scanner implementation for strat80_20."""
    def __init__(self):
        super().__init__(strategy_name='strat80_20')

    def get_setup_days(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = 'D',
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute all historical 80-20 setup days for a symbol within [from_date, to_date].
        Always returns a DataFrame with columns:
          ['symbol','setup_date','entry_date','entry_price','trigger_price','tick_size','true_range']
        """
        volume_threshold = int(kwargs.get('volume_threshold', 100000))

        df = fetch_historical_data(symbol, interval, from_date, to_date)
        if df is None or df.empty or len(df) < 2:
            return pd.DataFrame(columns=[
                'symbol','setup_date','entry_date','entry_price','trigger_price','tick_size','true_range'
            ])

        df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df['prev_close'] = df['close'].shift(1)
        df['true_high'] = np.maximum(df['high'], df['prev_close'].fillna(df['open']))
        df['true_low'] = np.minimum(df['low'], df['prev_close'].fillna(df['open']))
        df['tr'] = df['true_high'] - df['true_low']
        df['open_pos'] = (df['open'] - df['true_low']) / df['tr'].replace(0, np.nan)
        df['close_pos'] = (df['close'] - df['true_low']) / df['tr'].replace(0, np.nan)

        liquid = df['volume'] >= volume_threshold
        setup_mask = (df['open_pos'] >= _OPEN_POS_THRESHOLD) & (df['close_pos'] <= _CLOSE_POS_THRESHOLD) & liquid

        # Resolve tick size; default to 0.05 if missing
        try:
            instrument_info = fetch_instrument_info(symbol, exchange=EXCHANGE)
            tick_size = instrument_info.get('tick_size') if instrument_info is not None else None
            if tick_size is None:
                tick_size = 0.05
        except Exception:
            tick_size = 0.05

        records = []
        for ts, row in df.loc[setup_mask].iterrows():
            entry_price = float(row['low'])
            trigger_price = entry_price - _TRIGGER_TICK_MULT * tick_size
            records.append({
                'symbol': symbol,
                'setup_date': ts.date(),
                'entry_date': (ts + pd.Timedelta(days=1)).date(),
                'entry_price': round(entry_price, 2),
                'trigger_price': round(trigger_price, 2),
                'tick_size': tick_size,
                'true_range': round(float(row['tr']), 2),
            })
        return pd.DataFrame.from_records(records)


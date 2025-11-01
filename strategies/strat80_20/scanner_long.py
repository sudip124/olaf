# strat80_20_l_scanner.py (refactored to output JSON with entry_price, trigger_price, tick_size, true_range)
# scanner.py: Daily pre-market scanner for 80-20 setups
# NEW FILE MARKER - db_models.py should be created
import pandas as pd
import numpy as np
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .backtest_config import EXCHANGE
from data_manager.data_fetcher import fetch_historical_data, fetch_instrument_info

# Load config params from strat_config.json
config_path = os.path.join(os.path.dirname(__file__), 'strat_config.json')
with open(config_path, 'r') as f:
    params = json.load(f)
open_pos_threshold = params.get('open_pos_threshold', 0.8)
close_pos_threshold = params.get('close_pos_threshold', 0.2)
trigger_tick_mult = params.get('trigger_tick_mult', 10)


def detect_setup(symbol, interval, from_date, to_date, volume_threshold, open_pos_threshold, close_pos_threshold, trigger_tick_mult):
    """
    Fetch daily data and check if the latest day is a setup.
    Returns dict with setup details (JSON-ready) or None if no setup/no data.
    Includes: symbol, date, entry_price (prev day low), trigger_price, tick_size, true_range, volume.
    """
    try:
        df = fetch_historical_data(symbol, interval, from_date, to_date)
        if df.empty or len(df) < 2:
            return None
        
        # Ensure daily aggregation (robustness for any input interval)
        if interval != 'D':
            df = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Compute true range and positions (core 80-20 logic)
        df['prev_close'] = df['close'].shift(1)
        df['true_high'] = np.maximum(df['high'], df['prev_close'].fillna(df['open']))
        df['true_low'] = np.minimum(df['low'], df['prev_close'].fillna(df['open']))
        df['tr'] = df['true_high'] - df['true_low']
        df['open_pos'] = (df['open'] - df['true_low']) / df['tr'].replace(0, np.nan)
        df['close_pos'] = (df['close'] - df['true_low']) / df['tr'].replace(0, np.nan)
        
        # Latest (setup) day
        latest = df.iloc[-1]
        if latest['volume'] < volume_threshold:
            return None  # Skip illiquid symbols
        
        is_setup = (latest['open_pos'] >= open_pos_threshold) and (latest['close_pos'] <= close_pos_threshold)
        if not is_setup:
            return None
        
        # Setup confirmed: Compute trading params for next day
        entry_price = latest['low']
        true_range = latest['tr']
        
        # Fetch dynamic tick_size
        try:
            instrument_info = fetch_instrument_info(symbol, exchange=EXCHANGE)
            tick_size = instrument_info.get('tick_size', 0.05)
        except Exception as info_err:
            print(f"Warning: Could not fetch tick_size for {symbol}, defaulting to 0.05 ({info_err})")
            tick_size = 0.05
        
        trigger_price = entry_price - trigger_tick_mult * tick_size
        
        return {
            'symbol': symbol,
            'date': df.index[-1].date().isoformat(),
            'entry_price': round(entry_price, 2),
            'trigger_price': round(trigger_price, 2),
            'tick_size': tick_size,
            'true_range': round(true_range, 2),
            'volume': int(latest['volume']),  # Retained for liquidity reference
            'open_pos': round(latest['open_pos'], 4),
            'close_pos': round(latest['close_pos'], 4)
        }
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None


def get_setup_days(symbol: str, from_date: str, to_date: str, interval: str = 'D', volume_threshold: int = 100000,
                    open_pos_threshold: float = open_pos_threshold, close_pos_threshold: float = close_pos_threshold,
                    return_details: bool = False):
    """
    Compute all historical 80-20 setup days for a symbol within [from_date, to_date].
    - If return_details is False (default): returns a list[date] of setup days (backward compatible).
    - If return_details is True: returns a pandas.DataFrame with one row per setup day and columns:
      ['symbol','setup_date','entry_date','entry_price','trigger_price','tick_size','true_range','volume','open_pos','close_pos']
    """
    df = fetch_historical_data(symbol, interval, from_date, to_date)
    if df is None or df.empty or len(df) < 2:
        return [] if not return_details else pd.DataFrame(columns=[
            'symbol','setup_date','entry_date','entry_price','trigger_price','tick_size','true_range','volume','open_pos','close_pos'
        ])

    # Always resample to daily to ensure unique dates (handles duplicate DB rows)
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
    setup_mask = (df['open_pos'] >= open_pos_threshold) & (df['close_pos'] <= close_pos_threshold) & liquid

    if not return_details:
        return [ts.date() for ts in df.index[setup_mask]]

    # Detailed rows similar to detect_setup, but for all setup days
    try:
        instrument_info = fetch_instrument_info(symbol, exchange=EXCHANGE)
        tick_size = instrument_info.get('tick_size', 0.05)
    except Exception as info_err:
        print(f"Warning: Could not fetch tick_size for {symbol}, defaulting to 0.05 ({info_err})")
        tick_size = 0.05

    records = []
    for ts, row in df.loc[setup_mask].iterrows():
        entry_price = float(row['low'])
        trigger_price = entry_price - trigger_tick_mult * tick_size
        records.append({
            'symbol': symbol,
            'setup_date': ts.date(),
            'entry_date': (ts + pd.Timedelta(days=1)).date(),
            'entry_price': round(entry_price, 2),
            'trigger_price': round(trigger_price, 2),
            'tick_size': tick_size,
            'true_range': round(float(row['tr']), 2),
            'volume': int(row['volume']) if not pd.isna(row['volume']) else None,
            'open_pos': round(float(row['open_pos']), 4) if not pd.isna(row['open_pos']) else None,
            'close_pos': round(float(row['close_pos']), 4) if not pd.isna(row['close_pos']) else None,
        })

    return pd.DataFrame.from_records(records)


def run_scanner(symbols, interval, from_date, to_date, max_workers=10, volume_threshold=100000, sort_by_volume=True):  # Thread count tuned for API rate limits
    """
    Parallel scanner across symbols. Outputs JSON for next-day live trading integration
    (e.g., auto-place buy-stop orders at entry_price after trigger).
    """
    # Load symbols
    if not symbols:
        print("No symbols provided.")
        return pd.DataFrame(columns=['symbol','date','entry_price','trigger_price','tick_size','true_range','volume','open_pos','close_pos'])
    
    print(f"Scanning {len(symbols)} symbols for 80-20 setups...")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(
            detect_setup,
            symbol,
            interval,
            from_date,
            to_date,
            volume_threshold,
            open_pos_threshold,
            close_pos_threshold,
            trigger_tick_mult
        ): symbol for symbol in symbols}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    if not results:
        print("No setups found today.")
        return pd.DataFrame(columns=['symbol','date','entry_price','trigger_price','tick_size','true_range','volume','open_pos','close_pos'])
    
    # Sort by volume descending (best liquidity first)
    if sort_by_volume:
        results.sort(key=lambda x: x['volume'], reverse=True)
    
    # Build DataFrame for caller
    results_df = pd.DataFrame(results)

    # Table preview
    print(f"\nSetups found: {len(results_df)}")
    if not results_df.empty:
        print(results_df[['symbol', 'entry_price', 'trigger_price', 'true_range', 'volume']].round(2).to_string(index=False))
        print("\nSample row:")
        print(results_df.iloc[0].to_dict())

    # Next steps hint for production
    print("\nNext-day live: Load JSON, place buy-stop @ entry_price (trigger on drop to trigger_price).")
    
    return results_df


if __name__ == "__main__":
    symbols = []  # Provide symbols via arguments when calling from run.py
    to_date = datetime.date.today().isoformat()
    from_date = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
    run_scanner(symbols, 'D', from_date, to_date)
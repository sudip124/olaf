# strat80_20_l_scanner.py (refactored to output JSON with entry_price, trigger_price, tick_size, true_range)
# scanner.py: Daily pre-market scanner for 80-20 setups
import pandas as pd
import numpy as np
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from backtest_config import EXCHANGE
from data_fetcher import fetch_historical_data, fetch_instrument_info

# Hardcoded scanner-specific values
SCANNER_SYMBOLS_FILE = 'symbols.csv'  # Adjust to your actual file path/name
SCANNER_INTERVAL = 'D'  # Standardized to '1d' for OpenAlgo compatibility
SCANNER_DAYS_BACK = 5  # Days back to ensure sufficient data
SCANNER_VOLUME_THRESHOLD = 100000  # Minimum volume threshold

# Load config params from strat80_20_config.json
config_path = os.path.join(os.path.dirname(__file__), 'strategies', 'strat80_20_config.json')
with open(config_path, 'r') as f:
    params = json.load(f)
open_pos_threshold = params.get('open_pos_threshold', 0.8)
close_pos_threshold = params.get('close_pos_threshold', 0.2)
trigger_tick_mult = params.get('trigger_tick_mult', 10)


def detect_setup(symbol):
    """
    Fetch daily data and check if the latest day is a setup.
    Returns dict with setup details (JSON-ready) or None if no setup/no data.
    Includes: symbol, date, entry_price (prev day low), trigger_price, tick_size, true_range, volume.
    """
    to_date = datetime.date.today().isoformat()
    from_date = (datetime.date.today() - datetime.timedelta(days=SCANNER_DAYS_BACK)).isoformat()
    
    try:
        df = fetch_historical_data(symbol, SCANNER_INTERVAL, from_date, to_date, exchange=EXCHANGE)
        if df.empty or len(df) < 2:
            return None
        
        # Ensure daily aggregation (robustness for any input interval)
        if SCANNER_INTERVAL != '1d':
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
        if latest['volume'] < SCANNER_VOLUME_THRESHOLD:
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


def run_scanner(max_workers=10):  # Thread count tuned for API rate limits
    """
    Parallel scanner across symbols. Outputs JSON for next-day live trading integration
    (e.g., auto-place buy-stop orders at entry_price after trigger).
    """
    # Load symbols
    if not os.path.exists(SCANNER_SYMBOLS_FILE):
        print(f"Error: {SCANNER_SYMBOLS_FILE} not found. Create it with 'Symbol' column.")
        return
    
    symbols_df = pd.read_csv(SCANNER_SYMBOLS_FILE)
    symbols = symbols_df['Symbol'].tolist()
    print(f"Scanning {len(symbols)} symbols for 80-20 setups...")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(detect_setup, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    if not results:
        print("No setups found today.")
        return
    
    # Sort by volume descending (best liquidity first)
    results.sort(key=lambda x: x['volume'], reverse=True)
    
    # Save as JSON (list of dicts for easy parsing in live scripts)
    os.makedirs('logs', exist_ok=True)
    today = datetime.date.today().isoformat()
    logfile = f"logs/scanner_setups_{today}.json"
    with open(logfile, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Table preview
    results_df = pd.DataFrame(results)
    print(f"\nSetups found: {len(results_df)}")
    print(results_df[['symbol', 'entry_price', 'trigger_price', 'true_range', 'volume']].round(2).to_string(index=False))
    
    print(f"\nFull results saved to: {logfile}")
    print("\nSample JSON structure:")
    print(json.dumps(results[0], indent=2))
    
    # Next steps hint for production
    print("\nNext-day live: Load JSON, place buy-stop @ entry_price (trigger on drop to trigger_price).")


if __name__ == "__main__":
    run_scanner()
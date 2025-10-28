# scanner.py: Daily pre-market scanner for 80-20 short setups
import pandas as pd
import numpy as np
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from backtest_config import EXCHANGE, SCANNER_SYMBOLS_FILE, SCANNER_INTERVAL, SCANNER_DAYS_BACK, SCANNER_VOLUME_THRESHOLD
from data_fetcher import fetch_historical_data
from openalgo import api  # For client init if needed


# Load config params from strat80_20_s_config.json
config_path = os.path.join(os.path.dirname(__file__), 'strategies', 'strat80_20_s_config.json')
with open(config_path, 'r') as f:
    params = json.load(f)
open_pos_threshold = params.get('open_pos_threshold', 0.2)
close_pos_threshold = params.get('close_pos_threshold', 0.8)

def detect_setup(symbol):
    """
    Fetch daily data and check if the latest day is a short setup.
    Returns dict with results or None if no setup/no data.
    """
    to_date = datetime.date.today().isoformat()
    from_date = (datetime.date.today() - datetime.timedelta(days=SCANNER_DAYS_BACK)).isoformat()
    
    try:
        df = fetch_historical_data(symbol, SCANNER_INTERVAL, from_date, to_date, exchange=EXCHANGE)
        if df.empty or len(df) < 2:
            return None
        
        # Resample/ensure daily (if fetched intraday)
        if SCANNER_INTERVAL != '1d':
            df = df.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Compute true range and positions (from strat80_20.py logic, adapted for short)
        df['prev_close'] = df['close'].shift(1)
        df['true_high'] = np.maximum(df['high'], df['prev_close'].fillna(df['open']))
        df['true_low'] = np.minimum(df['low'], df['prev_close'].fillna(df['open']))
        df['tr'] = df['true_high'] - df['true_low']
        df['open_pos'] = (df['open'] - df['true_low']) / df['tr'].replace(0, np.nan)
        df['close_pos'] = (df['close'] - df['true_low']) / df['tr'].replace(0, np.nan)
        
        # Latest day
        latest = df.iloc[-1]
        if latest['volume'] < SCANNER_VOLUME_THRESHOLD:
            return None  # Skip illiquid
        
        is_setup = (latest['open_pos'] <= open_pos_threshold) and (latest['close_pos'] >= close_pos_threshold)
        if not is_setup:
            return None
        
        return {
            'symbol': symbol,
            'date': df.index[-1].date().isoformat(),
            'open_pos': round(latest['open_pos'], 4),
            'close_pos': round(latest['close_pos'], 4),
            'tr': round(latest['tr'], 2),
            'volume': latest['volume']
        }
    except Exception as e:
        print(f"Error for {symbol}: {e}")
        return None

def run_scanner(max_workers=10):  # Adjust threads based on API limits
    """
    Run scanner on all symbols in parallel.
    """
    # Load symbols from CSV
    symbols_df = pd.read_csv(SCANNER_SYMBOLS_FILE)
    symbols = symbols_df['Symbol'].tolist()
    print(f"Scanning {len(symbols)} symbols for short setups...")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(detect_setup, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    if not results:
        print("No short setups found today.")
        return
    
    # Save/output results
    results_df = pd.DataFrame(results)
    today = datetime.date.today().isoformat()
    logfile = f"logs/scanner_short_setups_{today}.csv"
    results_df.to_csv(logfile, index=False)
    print(f"Short setups found: {len(results_df)}")
    print(results_df)
    
    # Optional: Email/Slack integration (add your code here, e.g., using smtplib or slack-sdk)

if __name__ == "__main__":
    run_scanner()
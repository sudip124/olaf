"""
Utility script to download historical data with rate limiting and automatic chunking.

This script downloads historical data using data_fetcher.fetch_historical_data().
For intraday data, it automatically breaks the time interval into chunks of 120 days
or less to avoid exceeding broker API rate limits (max 3 calls per second).

Usage:
    python utilities/download_historical_data.py
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path to import data_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager.data_fetcher import fetch_historical_data

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================================================

# Symbols to download (can be a single symbol or list of symbols)
SYMBOLS = [
            "ADANIPOWER",
            "ATGL",
            "AFCONS",
            "ASTRAL",
            "BAJAJHFL",
            "BATAINDIA",
            "CHOLAFIN",
            "DABUR",
            "IGIL",
            "JUBLFOOD",
            "KAYNES",
            "LLOYDSME",
            "MANKIND",
            "MPHASIS",
            "NEWGEN",
            "RAINBOW",
            "SCI",
            "TORNTPHARM",
            "VEDL",
            "VIJAYA",
            "WAAREEENER",
            "WELCORP",
            "ZYDUSLIFE"
        ]
# Or use a single symbol: SYMBOLS = ["RELIANCE"]

# Interval: '1m', '3m', '5m', '15m', '30m', '1h', '1d', etc.
# For intraday (minute/hour), use format like '15 minute' or '15m'
INTERVAL = "15m"

# Date range (format: 'YYYY-MM-DD')
FROM_DATE = "2024-04-01"
TO_DATE = "2025-03-31"

# Rate limiting: max calls per second (broker allows 3)
MAX_CALLS_PER_SECOND = 3

# Chunk size for intraday data (in days)
CHUNK_SIZE_DAYS = 120

# Continue on error: if True, continues with next symbol if one fails
CONTINUE_ON_ERROR = True

# ============================================================================
# SCRIPT LOGIC - DO NOT EDIT BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
# ============================================================================

def is_intraday(interval: str) -> bool:
    """Check if interval is intraday (minute or hour based)."""
    interval_lower = interval.lower()
    return 'minute' in interval_lower or 'm' in interval_lower or 'h' in interval_lower


def generate_date_chunks(from_date: str, to_date: str, chunk_days: int):
    """
    Generate date chunks for downloading data.
    
    Args:
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        chunk_days: Maximum days per chunk
        
    Yields:
        Tuples of (chunk_from_date, chunk_to_date) as strings
    """
    start = pd.to_datetime(from_date).date()
    end = pd.to_datetime(to_date).date()
    
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        yield (current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d'))
        current = chunk_end + timedelta(days=1)


def download_with_rate_limit(symbol: str, interval: str, from_date: str, to_date: str) -> dict:
    """
    Download historical data with automatic chunking and rate limiting.
    
    Args:
        symbol: Trading symbol
        interval: Data interval
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        
    Returns:
        dict: Summary with 'success', 'total_records', 'chunks_processed', 'error'
    """
    result = {
        'success': False,
        'total_records': 0,
        'chunks_processed': 0,
        'error': None
    }
    
    print(f"\n{'='*80}")
    print(f"DOWNLOADING: {symbol}")
    print(f"{'='*80}")
    print(f"Interval:     {interval}")
    print(f"Date Range:   {from_date} to {to_date}")
    print(f"{'='*80}\n")
    
    # Check if intraday and needs chunking
    if is_intraday(interval):
        # Calculate total days
        start = pd.to_datetime(from_date).date()
        end = pd.to_datetime(to_date).date()
        total_days = (end - start).days + 1
        
        if total_days > CHUNK_SIZE_DAYS:
            print(f"⚠️  Intraday data request spans {total_days} days.")
            print(f"📦 Breaking into chunks of {CHUNK_SIZE_DAYS} days to avoid API limits.\n")
            
            chunks = list(generate_date_chunks(from_date, to_date, CHUNK_SIZE_DAYS))
            total_chunks = len(chunks)
            
            print(f"Total chunks to download: {total_chunks}\n")
            
            for idx, (chunk_from, chunk_to) in enumerate(chunks, 1):
                print(f"[{idx}/{total_chunks}] Downloading {chunk_from} to {chunk_to}...", end=" ")
                
                start_time = time.time()
                try:
                    df = fetch_historical_data(symbol, interval, chunk_from, chunk_to)
                    elapsed = time.time() - start_time
                    
                    if df.empty:
                        print(f"⚠️  No data (elapsed: {elapsed:.2f}s)")
                    else:
                        result['total_records'] += len(df)
                        print(f"✓ {len(df)} records (elapsed: {elapsed:.2f}s)")
                    
                    result['chunks_processed'] += 1
                    
                    # Rate limiting: ensure we don't exceed MAX_CALLS_PER_SECOND
                    sleep_time = (1.0 / MAX_CALLS_PER_SECOND) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                except Exception as e:
                    print(f"✗ Error: {e}")
                    result['error'] = str(e)
                    print(f"Stopping download for {symbol} due to error.")
                    return result
            
            result['success'] = True
            print(f"\n✅ Download complete! All {total_chunks} chunks processed.")
        else:
            # Single request for intraday data within chunk size
            print(f"Downloading intraday data (single request)...", end=" ")
            start_time = time.time()
            try:
                df = fetch_historical_data(symbol, interval, from_date, to_date)
                elapsed = time.time() - start_time
                
                if df.empty:
                    print(f"⚠️  No data (elapsed: {elapsed:.2f}s)")
                else:
                    result['total_records'] = len(df)
                    print(f"✓ {len(df)} records (elapsed: {elapsed:.2f}s)")
                
                result['chunks_processed'] = 1
                result['success'] = True
                print(f"\n✅ Download complete!")
            except Exception as e:
                print(f"✗ Error: {e}")
                result['error'] = str(e)
    else:
        # Daily data - single request (no chunking needed)
        print(f"Downloading daily data (single request)...", end=" ")
        start_time = time.time()
        try:
            df = fetch_historical_data(symbol, interval, from_date, to_date)
            elapsed = time.time() - start_time
            
            if df.empty:
                print(f"⚠️  No data (elapsed: {elapsed:.2f}s)")
            else:
                result['total_records'] = len(df)
                print(f"✓ {len(df)} records (elapsed: {elapsed:.2f}s)")
            
            result['chunks_processed'] = 1
            result['success'] = True
            print(f"\n✅ Download complete!")
        except Exception as e:
            print(f"✗ Error: {e}")
            result['error'] = str(e)
    
    return result


def main():
    """Main entry point."""
    # Ensure SYMBOLS is a list
    symbols = SYMBOLS if isinstance(SYMBOLS, list) else [SYMBOLS]
    
    total_symbols = len(symbols)
    results = {}
    
    print(f"\n{'#'*80}")
    print(f"# BATCH DOWNLOAD - {total_symbols} SYMBOL(S)")
    print(f"{'#'*80}")
    print(f"Symbols:      {', '.join(symbols)}")
    print(f"Interval:     {INTERVAL}")
    print(f"Date Range:   {FROM_DATE} to {TO_DATE}")
    print(f"Rate Limit:   {MAX_CALLS_PER_SECOND} calls/second")
    print(f"Continue on Error: {CONTINUE_ON_ERROR}")
    print(f"{'#'*80}")
    
    try:
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[SYMBOL {idx}/{total_symbols}]")
            
            try:
                result = download_with_rate_limit(symbol, INTERVAL, FROM_DATE, TO_DATE)
                results[symbol] = result
                
                if not result['success'] and not CONTINUE_ON_ERROR:
                    print(f"\n⚠️  Stopping batch download due to error in {symbol}.")
                    break
                    
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Download interrupted by user at symbol {symbol}.")
                results[symbol] = {'success': False, 'error': 'Interrupted by user'}
                raise
            except Exception as e:
                print(f"\n❌ Unexpected error for {symbol}: {e}")
                results[symbol] = {'success': False, 'error': str(e)}
                
                if not CONTINUE_ON_ERROR:
                    print(f"\n⚠️  Stopping batch download due to error.")
                    break
        
        # Print summary
        print(f"\n\n{'#'*80}")
        print(f"# DOWNLOAD SUMMARY")
        print(f"{'#'*80}")
        
        successful = [s for s, r in results.items() if r.get('success', False)]
        failed = [s for s, r in results.items() if not r.get('success', False)]
        
        print(f"\n✅ Successful: {len(successful)}/{total_symbols}")
        for symbol in successful:
            r = results[symbol]
            print(f"   • {symbol}: {r['total_records']} records, {r['chunks_processed']} chunk(s)")
        
        if failed:
            print(f"\n❌ Failed: {len(failed)}/{total_symbols}")
            for symbol in failed:
                r = results[symbol]
                error_msg = r.get('error', 'Unknown error')
                print(f"   • {symbol}: {error_msg}")
        
        print(f"\n{'#'*80}\n")
        
        if failed and not CONTINUE_ON_ERROR:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

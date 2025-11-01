# populate_db.py
import os
from pathlib import Path
from data_manager.data_fetcher import fetch_historical_data
from data_manager.config import DB_PATH, ensure_directories
from strategies.strat80_20.backtest_config import SYMBOLS, INTERVAL, FROM_DATE, TO_DATE, EXCHANGE

def populate_database():
    """
    Script to fetch and store historical data for all configured symbols into the local SQLite DB.
    Hardcoded: Uses SYMBOLS, INTERVAL ('15m'), FROM_DATE (200 days back), TO_DATE (today), EXCHANGE ('NSE').
    Since the DB is deleted/redone from scratch, this will refetch everything from OpenAlgo and populate tables.
    """
    ensure_directories()  # Ensure db/ and logs/ directories exist
    # Additionally ensure DB folder
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    print(f"Populating DB with data from {FROM_DATE} to {TO_DATE} at {INTERVAL} interval for {len(SYMBOLS)} symbols...")

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            df = fetch_historical_data(symbol, INTERVAL, FROM_DATE, TO_DATE)
            if df.empty:
                print(f"No data available for {symbol} in the range.")
            else:
                print(f"Stored {len(df)} bars for {symbol}.")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    print("DB population complete. Tables created/updated in db/historify.db.")

if __name__ == "__main__":
    populate_database()
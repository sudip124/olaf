# populate_db.py
import os
from data_fetcher import fetch_historical_data
from backtest_config import SYMBOLS, INTERVAL, FROM_DATE, TO_DATE, EXCHANGE

def populate_database():
    """
    Script to fetch and store historical data for all configured symbols into the local SQLite DB.
    Hardcoded: Uses SYMBOLS, INTERVAL ('15m'), FROM_DATE (200 days back), TO_DATE (today), EXCHANGE ('NSE').
    Since the DB is deleted/redone from scratch, this will refetch everything from OpenAlgo and populate tables.
    """
    os.makedirs('db', exist_ok=True)  # Ensure db/ directory exists
    print(f"Populating DB with data from {FROM_DATE} to {TO_DATE} at {INTERVAL} interval for {len(SYMBOLS)} symbols...")

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            df = fetch_historical_data(symbol, INTERVAL, FROM_DATE, TO_DATE, exchange=EXCHANGE)
            if df.empty:
                print(f"No data available for {symbol} in the range.")
            else:
                print(f"Stored {len(df)} bars for {symbol}.")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    print("DB population complete. Tables created/updated in db/historify.db.")

if __name__ == "__main__":
    populate_database()
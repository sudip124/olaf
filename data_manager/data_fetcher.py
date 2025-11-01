import pandas as pd
import datetime
from pathlib import Path
from openalgo import api
from data_manager.config import OPENALGO_URL, API_KEY, DB_PATH, SAVE_TO_LOGS, EXCHANGE
from sqlalchemy import create_engine, text
import os  # Added for os.makedirs and file paths

# Initialize OpenAlgo client once
client = api(api_key=API_KEY, host=OPENALGO_URL)

# Initialize SQLAlchemy engine once
engine = create_engine(f"sqlite:///{DB_PATH}")

def _load_range_from_db(conn, table_name, from_date, to_date):
    """Internal helper to load OHLCV data from DB using a SQLAlchemy Connection or Engine."""
    # Convert input dates to strings for SQLite
    from_date_str = pd.to_datetime(from_date).strftime('%Y-%m-%d')
    to_date_str = pd.to_datetime(to_date).strftime('%Y-%m-%d')

    sql = f'''
    SELECT date, time, open, high, low, close, volume
    FROM "{table_name}"
    WHERE date BETWEEN :from_d AND :to_d
    ORDER BY date ASC, time ASC
    '''
    df = pd.read_sql_query(text(sql), conn, params={"from_d": from_date_str, "to_d": to_date_str})

    if df.empty:
        return df  # Empty DF

    # Handle NULL times (common in daily tables) by filling '00:00:00'
    df['time'] = df['time'].fillna('00:00:00')

    # Combine date + time into timestamp (handles str/object from SQLite)
    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')

    # Drop any invalid timestamps (safety)
    df = df.dropna(subset=['timestamp'])

    df = df.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']]
    return df

def fetch_historical_data(symbol, interval, from_date, to_date):
    """
    Fetch OHLCV data: prefer local nse.db if it fully covers the requested date range
    (based on MIN/MAX date check). If not (table missing, partial coverage, or gaps assumed absent),
    fetch from OpenAlgo, cache into DB (INSERT OR IGNORE to avoid duplicates), then return
    the data for the exact range from DB for consistency.
    
    - Normalizes interval for API (e.g., '15 minute' -> '15m').
    - Uses 'd' for daily tables in DB for consistency (compatible with existing '_D' via fallback).
    - Creates table + indexes if needed .
    - Returns pandas DF with DatetimeIndex, columns: open, high, low, close, volume.
    - If SAVE_TO_LOGS is True, saves newly downloaded data (from API) to logs/ as CSV.
    """
    # Normalize interval for OpenAlgo API
    if 'minute' in interval.lower():
        api_interval = interval.lower().replace('minute', 'm')
    else:
        api_interval = interval
    
    conn = engine.connect()

    symbol_lower = symbol.lower()
    exchange_lower = EXCHANGE.lower()

    # DB interval normalization: intraday as-is, daily -> 'd'
    if api_interval[-1] in ('m', 'h'):
        db_interval = api_interval
    else:
        db_interval = 'd'  # Standardize daily to 'd'
    
    # Candidates for daily (handle existing '_D' tables if present)
    candidates = [f"data_{symbol_lower}_{exchange_lower}_{db_interval}"]
    if db_interval == 'd':
        candidates.append(f"data_{symbol_lower}_{exchange_lower}_D")
    
    table_name = None
    existing_min_max = None
    for cand in candidates:
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": cand}
        )
        if result.fetchone():
            table_name = cand
            result2 = conn.execute(text(f"SELECT MIN(date), MAX(date) FROM \"{table_name}\""))
            row = result2.fetchone()
            if row and row[0] and row[1]:
                min_str, max_str = row[0], row[1]
                existing_min_max = (pd.to_datetime(min_str).date(), pd.to_datetime(max_str).date())
            break
    
    # Parse request dates
    from_d = pd.to_datetime(from_date).date()
    to_d = pd.to_datetime(to_date).date()
    
    # Check if local fully covers (MIN <= from AND MAX >= to); assumes no internal gaps if downloaded
    if existing_min_max:
        local_min, local_max = existing_min_max
        if local_min <= from_d and local_max >= to_d:
            df = _load_range_from_db(conn, table_name, from_date, to_date)
            conn.close()
            if not df.empty:
                return df  # Full coverage -> use local
    
    # Not fully covered or no table -> fetch from OpenAlgo
    df = client.history(
        symbol=symbol,
        exchange=EXCHANGE,
        interval=api_interval,
        start_date=from_date,
        end_date=to_date
    )
    
    if not isinstance(df, pd.DataFrame):
        conn.close()
        raise ValueError(f"Unexpected response from client.history: {df}")
    
    if df.empty:
        conn.close()
        return df  # Empty (e.g., no data in range)
    
    # Standardize columns
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df.index.name = 'timestamp'
    
    # Ensure volume is int (SQLite INTEGER)
    df['volume'] = df['volume'].round(0).astype(int)
    
    # NEW: Save to logs/ if configured (only for newly downloaded data)
    if SAVE_TO_LOGS:
        os.makedirs('logs', exist_ok=True)  # Create logs/ if it doesn't exist
        csv_path = f'logs/{symbol}_{api_interval}_{from_date}_{to_date}.csv'
        df.to_csv(csv_path)
        print(f"Saved downloaded OHLC data to {csv_path}")

    # Determine final table (prefer existing if found, else first candidate)
    if table_name is None:
        table_name = candidates[0]
    
    # Create table and insert data within a separate write transaction/connection
    uix_name = f"uix_{table_name}_date_time"
    create_sql = f'''
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        "id" INTEGER NOT NULL,
        "date" DATE NOT NULL,
        "time" TIME,
        "open" FLOAT NOT NULL,
        "high" FLOAT NOT NULL,
        "low" FLOAT NOT NULL,
        "close" FLOAT NOT NULL,
        "volume" INTEGER NOT NULL,
        "created_at" DATETIME,
        PRIMARY KEY("id"),
        CONSTRAINT "{uix_name}" UNIQUE("date", "time")
    );
    '''

    # Prepare for INSERT OR IGNORE
    df_insert = df.reset_index()
    df_insert['date'] = df_insert['timestamp'].dt.date.apply(lambda x: x.strftime('%Y-%m-%d'))

    if db_interval != 'd':
        df_insert['time'] = df_insert['timestamp'].dt.time.apply(
            lambda x: x.strftime('%H:%M:%S') if x else None
        )
    else:
        df_insert['time'] = None

    created_at = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    # Ensure numeric types are float for SQLite
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        df_insert[col] = df_insert[col].astype(float)

    # Ensure volume is integer
    df_insert['volume'] = df_insert['volume'].astype(int)

    insert_sql = f'''
    INSERT OR IGNORE INTO "{table_name}"
    (date, time, open, high, low, close, volume, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''

    data_tuples = list(df_insert[
        ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    ].itertuples(index=False, name=None))

    # Add created_at to each tuple
    data_tuples = [(d, t, o, h, l, c, v, created_at)
                   for d, t, o, h, l, c, v in data_tuples]

    # Use engine.begin() to handle transaction separately from the read-connection
    with engine.begin() as write_conn:
        write_conn.execute(text(create_sql))
        write_conn.execute(text(f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_date" ON "{table_name}" ("date");'))
        write_conn.execute(text(f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_time" ON "{table_name}" ("time");'))
        write_conn.exec_driver_sql(insert_sql, data_tuples)

    # Now load the (now-complete) range from DB for consistency
    df = _load_range_from_db(engine, table_name, from_date, to_date)
    conn.close()

    return df

def fetch_instrument_info(symbol, exchange='NSE'):
    """
    Fetch instrument info (e.g., tick size) for a symbol from OpenAlgo.
    Returns a dict with instrument details.
    """
    quote = client.quotes(symbol=symbol, exchange=exchange)
    # Defensive: handle missing keys
    info = quote.get('data', {})
    return info

# Example usage (unchanged):
# info = fetch_instrument_info('RELIANCE')
# print("Instrument Info:", info)
# print("Tick Size:", info.get('tick_size'))

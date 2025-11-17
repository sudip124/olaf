"""
Utility to check for duplicate records in the database.
This helps verify if INSERT OR IGNORE is working correctly.
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager.config import DB_PATH

def check_duplicates(symbol="ADANIPOWER", interval="d"):
    """Check for duplicate (date, time) entries in a table."""
    
    symbol_lower = symbol.lower()
    exchange_lower = "nse"
    table_name = f"data_{symbol_lower}_{exchange_lower}_{interval}"
    
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    print(f"\n{'='*80}")
    print(f"Checking table: {table_name}")
    print(f"{'='*80}\n")
    
    # Check if table exists
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table_name}
        )
        if not result.fetchone():
            print(f"❌ Table {table_name} does not exist!")
            return
    
    # Query for duplicates
    query = f'''
    SELECT date, time, COUNT(*) as count
    FROM "{table_name}"
    GROUP BY date, time
    HAVING COUNT(*) > 1
    ORDER BY date DESC, time DESC
    '''
    
    df_duplicates = pd.read_sql_query(text(query), engine)
    
    if df_duplicates.empty:
        print("✅ No duplicates found! INSERT OR IGNORE is working correctly.")
    else:
        print(f"❌ Found {len(df_duplicates)} duplicate (date, time) combinations:")
        print(df_duplicates.to_string(index=False))
        
        # Show actual records for first duplicate
        if len(df_duplicates) > 0:
            first_dup = df_duplicates.iloc[0]
            date_val = first_dup['date']
            time_val = first_dup['time']
            
            print(f"\n\nShowing all records for date={date_val}, time={time_val}:")
            detail_query = f'''
            SELECT * FROM "{table_name}"
            WHERE date = :date_val AND time {'IS NULL' if pd.isna(time_val) else '= :time_val'}
            '''
            params = {"date_val": date_val}
            if not pd.isna(time_val):
                params["time_val"] = time_val
                
            df_detail = pd.read_sql_query(text(detail_query), engine, params=params)
            print(df_detail.to_string(index=False))
    
    # Show total record count
    count_query = f'SELECT COUNT(*) as total FROM "{table_name}"'
    total = pd.read_sql_query(text(count_query), engine).iloc[0]['total']
    print(f"\n\nTotal records in table: {total}")
    
    # Show date range
    range_query = f'SELECT MIN(date) as min_date, MAX(date) as max_date FROM "{table_name}"'
    date_range = pd.read_sql_query(text(range_query), engine).iloc[0]
    print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Check ADANIPOWER daily data (the example you showed)
    check_duplicates("ADANIPOWER", "d")
    
    # Optionally check other symbols/intervals
    # check_duplicates("ADANIPOWER", "15m")

"""
Utility to check the actual table schema in the database.
"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager.config import DB_PATH

def check_schema(symbol="ADANIPOWER", interval="d"):
    """Check the actual schema of a table."""
    
    symbol_lower = symbol.lower()
    exchange_lower = "nse"
    table_name = f"data_{symbol_lower}_{exchange_lower}_{interval}"
    
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    print(f"\n{'='*80}")
    print(f"Schema for table: {table_name}")
    print(f"{'='*80}\n")
    
    with engine.connect() as conn:
        # Get table schema
        result = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table_name}
        )
        row = result.fetchone()
        if row:
            print("CREATE TABLE statement:")
            print(row[0])
        else:
            print(f"❌ Table {table_name} does not exist!")
            return
        
        print("\n" + "="*80 + "\n")
        
        # Get indexes
        result = conn.execute(
            text("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=:name"),
            {"name": table_name}
        )
        indexes = result.fetchall()
        
        if indexes:
            print("Indexes:")
            for idx_name, idx_sql in indexes:
                if idx_sql:  # Some indexes (like PRIMARY KEY) don't have SQL
                    print(f"\n{idx_name}:")
                    print(idx_sql)
                else:
                    print(f"\n{idx_name}: (auto-created, no SQL)")
        else:
            print("No indexes found.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    check_schema("ADANIPOWER", "d")

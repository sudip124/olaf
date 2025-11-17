"""
One-time cleanup script to remove duplicate records from the database.

This script:
1. Identifies duplicate (date, time) entries in each table
2. Keeps the record with the highest id (most recent insert)
3. Deletes older duplicate records
4. Reports statistics for each symbol

Run this ONCE after updating data_fetcher.py to use UPSERT.
"""

import sys
from pathlib import Path
from sqlalchemy import create_engine, text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_manager.config import DB_PATH

# List of symbols to clean up
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

# Intervals to check (both daily and intraday)
INTERVALS = ['d', '15m']

def cleanup_duplicates(symbol, interval, engine, dry_run=False):
    """
    Remove duplicate records from a table, keeping the one with highest id.
    
    Args:
        symbol: Trading symbol
        interval: Data interval ('d', '15m', etc.)
        engine: SQLAlchemy engine
        dry_run: If True, only report duplicates without deleting
        
    Returns:
        dict: Statistics about the cleanup
    """
    symbol_lower = symbol.lower()
    exchange_lower = "nse"
    table_name = f"data_{symbol_lower}_{exchange_lower}_{interval}"
    
    stats = {
        'table': table_name,
        'exists': False,
        'total_records_before': 0,
        'duplicate_groups': 0,
        'records_deleted': 0,
        'total_records_after': 0
    }
    
    with engine.begin() as conn:
        # Check if table exists
        result = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
            {"name": table_name}
        )
        if not result.fetchone():
            return stats
        
        stats['exists'] = True
        
        # Get total records before cleanup
        result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        stats['total_records_before'] = result.fetchone()[0]
        
        # Find duplicate groups
        dup_query = f'''
        SELECT date, time, COUNT(*) as cnt
        FROM "{table_name}"
        GROUP BY date, time
        HAVING COUNT(*) > 1
        '''
        result = conn.execute(text(dup_query))
        duplicates = result.fetchall()
        stats['duplicate_groups'] = len(duplicates)
        
        if stats['duplicate_groups'] == 0:
            stats['total_records_after'] = stats['total_records_before']
            return stats
        
        # For each duplicate group, delete all but the one with highest id
        for date_val, time_val, count in duplicates:
            # Build query to find ids to delete (all except the max id)
            # We keep the one with MAX(id) and delete the rest
            if time_val is None:
                find_ids_query = f'''
                SELECT id FROM "{table_name}"
                WHERE date = :date_val AND time IS NULL
                AND id < (
                    SELECT MAX(id) FROM "{table_name}"
                    WHERE date = :date_val AND time IS NULL
                )
                '''
                params = {"date_val": date_val}
            else:
                find_ids_query = f'''
                SELECT id FROM "{table_name}"
                WHERE date = :date_val AND time = :time_val
                AND id < (
                    SELECT MAX(id) FROM "{table_name}"
                    WHERE date = :date_val AND time = :time_val
                )
                '''
                params = {"date_val": date_val, "time_val": time_val}
            
            result = conn.execute(text(find_ids_query), params)
            ids_to_delete = [row[0] for row in result.fetchall()]
            
            if ids_to_delete:
                stats['records_deleted'] += len(ids_to_delete)
                
                if not dry_run:
                    # Delete the duplicate records
                    # Use parameterized query with proper binding
                    placeholders = ','.join([f':id{i}' for i in range(len(ids_to_delete))])
                    delete_query = f'''
                    DELETE FROM "{table_name}"
                    WHERE id IN ({placeholders})
                    '''
                    delete_params = {f'id{i}': id_val for i, id_val in enumerate(ids_to_delete)}
                    conn.execute(text(delete_query), delete_params)
        
        # Get total records after cleanup
        result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        stats['total_records_after'] = result.fetchone()[0]
    
    return stats


def main(dry_run=False):
    """
    Main cleanup function.
    
    Args:
        dry_run: If True, only report duplicates without deleting
    """
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    print(f"\n{'='*80}")
    print(f"DATABASE DUPLICATE CLEANUP {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*80}")
    print(f"Database: {DB_PATH}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Intervals: {INTERVALS}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("⚠️  DRY RUN MODE - No records will be deleted\n")
    else:
        print("⚠️  WARNING: This will permanently delete duplicate records!")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
        print()
    
    all_stats = []
    
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        
        for interval in INTERVALS:
            stats = cleanup_duplicates(symbol, interval, engine, dry_run)
            
            if stats['exists']:
                all_stats.append(stats)
                
                if stats['duplicate_groups'] > 0:
                    print(f"  [{interval:>4}] Found {stats['duplicate_groups']} duplicate groups, "
                          f"{stats['records_deleted']} records {'would be ' if dry_run else ''}deleted "
                          f"({stats['total_records_before']} → {stats['total_records_after']} records)")
                else:
                    print(f"  [{interval:>4}] No duplicates ({stats['total_records_before']} records)")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    tables_processed = len([s for s in all_stats if s['exists']])
    tables_with_dups = len([s for s in all_stats if s['duplicate_groups'] > 0])
    total_deleted = sum(s['records_deleted'] for s in all_stats)
    
    print(f"Tables processed: {tables_processed}")
    print(f"Tables with duplicates: {tables_with_dups}")
    print(f"Total records {'that would be ' if dry_run else ''}deleted: {total_deleted}")
    
    if tables_with_dups > 0:
        print(f"\nTables with duplicates:")
        for stats in all_stats:
            if stats['duplicate_groups'] > 0:
                print(f"  • {stats['table']}: {stats['duplicate_groups']} groups, "
                      f"{stats['records_deleted']} records deleted")
    
    print(f"\n{'='*80}\n")
    
    if dry_run:
        print("✅ Dry run complete. Run with dry_run=False to actually delete duplicates.")
    else:
        print("✅ Cleanup complete!")


if __name__ == "__main__":
    # First run in dry-run mode to see what would be deleted
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        main(dry_run=False)
    else:
        print("\n" + "="*80)
        print("DRY RUN MODE - Add --execute flag to actually delete duplicates")
        print("="*80 + "\n")
        main(dry_run=True)

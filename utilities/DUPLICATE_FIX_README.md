# Database Duplicate Fix - Solution Documentation

## Problem
The `fetch_historical_data()` function was creating duplicate records in the database when downloading data for date ranges that partially overlapped with existing data.

### Root Cause
- The function used `INSERT OR IGNORE` to handle duplicates
- However, when partial date coverage existed, it would download the entire date range from the API
- While `INSERT OR IGNORE` prevented errors, it still created duplicate records with different `id` values
- This happened because the UNIQUE constraint on `(date, time)` was not being properly enforced during inserts

### Example Issue
For ADANIPOWER daily data:
- **ID 1-3:** Nov 13, 14, 17 (downloaded at 13:13:22)
- **ID 406-408:** Nov 13, 14, 17 (downloaded again at 13:58:15)

Result: 408 total records instead of 405, with 3 duplicate date entries.

## Solution Implemented

### 1. Updated `data_fetcher.py` to use UPSERT
**Changed from:**
```sql
INSERT OR IGNORE INTO table (date, time, open, high, low, close, volume, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
```

**Changed to:**
```sql
INSERT INTO table (date, time, open, high, low, close, volume, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(date, time) DO UPDATE SET
    open = excluded.open,
    high = excluded.high,
    low = excluded.low,
    close = excluded.close,
    volume = excluded.volume,
    created_at = excluded.created_at
```

**Benefits:**
- No duplicate records created
- Existing records are updated with latest data
- Original `id` is preserved (no gaps in auto-increment)
- Latest `created_at` timestamp is stored

### 2. Created Cleanup Script
**File:** `utilities/cleanup_duplicates.py`

**Purpose:** Remove existing duplicate records from the database (one-time cleanup)

**Features:**
- Identifies duplicate `(date, time)` combinations
- Keeps the record with the highest `id` (most recent insert)
- Deletes older duplicate records
- Supports dry-run mode to preview changes
- Processes all 23 symbols and both daily ('d') and intraday ('15m') data

## How to Use

### Step 1: Review What Will Be Deleted (Dry Run)
```bash
python utilities/cleanup_duplicates.py
```

This will show:
- Number of duplicate groups per table
- Number of records that would be deleted
- Total summary across all tables

### Step 2: Execute the Cleanup
```bash
python utilities/cleanup_duplicates.py --execute
```

You will be prompted to confirm before any records are deleted.

### Step 3: Verify the Fix
```bash
python utilities/check_db_duplicates.py
```

This will verify that no duplicates remain in the ADANIPOWER daily table.

## Expected Results

### Before Cleanup
- **23 tables** with duplicates (all daily data tables)
- **69 total duplicate records** (3 per symbol: Nov 13, 14, 17)
- Example: ADANIPOWER has 408 records (should be 405)

### After Cleanup
- **0 duplicate records** in all tables
- Example: ADANIPOWER will have 405 records
- All future downloads will use UPSERT (no new duplicates)

## Affected Symbols
All 23 symbols in the download list have duplicate entries:
- ADANIPOWER, ATGL, AFCONS, ASTRAL, BAJAJHFL, BATAINDIA, CHOLAFIN, DABUR
- IGIL, JUBLFOOD, KAYNES, LLOYDSME, MANKIND, MPHASIS, NEWGEN, RAINBOW
- SCI, TORNTPHARM, VEDL, VIJAYA, WAAREEENER, WELCORP, ZYDUSLIFE

## Testing

### Test 1: Verify UPSERT Works
1. Note current record count for a symbol
2. Download data for a date range that overlaps existing data
3. Verify record count doesn't increase (records are updated, not duplicated)

### Test 2: Verify No Duplicates
```bash
python utilities/check_db_duplicates.py
```

Should show: "✅ No duplicates found! INSERT OR IGNORE is working correctly."

## Utility Scripts Created

1. **`check_db_duplicates.py`** - Check for duplicate records in a table
2. **`check_table_schema.py`** - View table schema and indexes
3. **`cleanup_duplicates.py`** - Remove existing duplicate records

## Notes

- The UNIQUE constraint on `(date, time)` was always present in the schema
- The issue was with the INSERT logic, not the constraint
- UPSERT ensures data consistency and prevents duplicates going forward
- The cleanup script is a one-time operation to fix historical duplicates
- After cleanup, you can safely delete the cleanup script if desired

## Date Fixed
November 17, 2025

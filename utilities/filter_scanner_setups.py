"""
Utility to push selected scanner setups into run_config.json for live trading.

- Edit the ALLOWED_SYMBOLS list below.
- Script reads scanner_setups.json (array of setups), filters to ALLOWED_SYMBOLS,
  then updates run_config.json -> live.setups with those entries.
- The script DOES NOT modify scanner_setups.json.
- A timestamped backup of run_config.json is saved in logs/ before writing.

Usage:
    python utilities/filter_scanner_setups.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Project root and paths
ROOT = Path(__file__).resolve().parents[1]
SETUPS_PATH = ROOT / "scanner_setups.json"
RUN_CONFIG_PATH = ROOT / "run_config.json"
BACKUP_DIR = ROOT / "logs"

# ================================
# EDIT: Allowed symbols to keep
# ================================
ALLOWED_SYMBOLS = [
    # Example: add your symbols here
    "M&M", "SCHNEIDER", "IRB", "SARDAEN", "SAGILITY", "SRF"
]


def main() -> None:
    # Validate input files
    if not SETUPS_PATH.exists():
        print(f"Error: scanner_setups.json not found at {SETUPS_PATH}")
        sys.exit(1)
    if not RUN_CONFIG_PATH.exists():
        print(f"Error: run_config.json not found at {RUN_CONFIG_PATH}")
        sys.exit(1)

    # Read scanner setups (array)
    try:
        with open(SETUPS_PATH, "r", encoding="utf-8") as f:
            scanner_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse scanner_setups.json: {e}")
        sys.exit(1)
    if not isinstance(scanner_data, list):
        print("Error: scanner_setups.json should contain a JSON array of setup objects.")
        sys.exit(1)

    # Filter by symbols
    allowed_set = set(ALLOWED_SYMBOLS)
    before = len(scanner_data)
    filtered = [row for row in scanner_data if isinstance(row, dict) and row.get("symbol") in allowed_set]

    # Deduplicate by symbol: keep the most recent setup_date for each symbol
    def _parse_dt(s: str):
        try:
            # Handle standard ISO with milliseconds, and optional 'Z'
            s2 = s.replace('Z', '+00:00') if isinstance(s, str) else s
            return datetime.fromisoformat(s2)
        except Exception:
            return None

    best_by_symbol = {}
    for row in filtered:
        sym = row.get("symbol")
        dt = _parse_dt(row.get("setup_date"))
        if sym is None:
            continue
        if sym not in best_by_symbol:
            best_by_symbol[sym] = row
            continue
        # Compare dates; if parsing fails, keep existing
        prev_dt = _parse_dt(best_by_symbol[sym].get("setup_date"))
        if dt is not None and (prev_dt is None or dt > prev_dt):
            best_by_symbol[sym] = row

    filtered = [best_by_symbol[s] for s in sorted(best_by_symbol.keys())]
    after = len(filtered)

    # Load run_config
    try:
        with open(RUN_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse run_config.json: {e}")
        sys.exit(1)
    if not isinstance(cfg, dict):
        print("Error: run_config.json root must be an object.")
        sys.exit(1)

    # Ensure live section exists and update setups
    live_cfg = cfg.get("live")
    if not isinstance(live_cfg, dict):
        live_cfg = {}
        cfg["live"] = live_cfg
    live_cfg["setups"] = filtered

    # Backup run_config.json
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"run_config_backup_{ts}.json"
    with open(backup_path, "w", encoding="utf-8") as bf:
        json.dump(cfg, bf, ensure_ascii=False, indent=2)

    # Write updated run_config.json
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Report
    print("Live setups update complete.")
    print(f"scanner_setups path: {SETUPS_PATH}")
    print(f"run_config path:     {RUN_CONFIG_PATH}")
    print(f"Allowed syms:        {len(allowed_set)}")
    print(f"Scanner rows:        {before}")
    print(f"Selected setups:     {after}")
    print("Note: Deduplicated by symbol, keeping the most recent setup_date per symbol.")
    print(f"Backup of run_config saved to: {backup_path}")


if __name__ == "__main__":
    main()

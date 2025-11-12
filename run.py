import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Make sure project root is on sys.path when running from subfolders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def do_backtest(cfg: Dict[str, Any]) -> None:
    # Get strategy name from config
    strategy_name = cfg.get("strategy", "strat80_20")
    
    # Lazy import to ensure DB_PATH from config/env is set before modules load
    from strategies.strategy_loader import load_backtest_strategy
    
    try:
        strategy = load_backtest_strategy(strategy_name)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Error loading backtest strategy '{strategy_name}': {e}")
        sys.exit(1)

    # Extract backtest config
    backtest_cfg = cfg.get("backtest", {})
    
    # Required: list of symbols to test
    symbols: Optional[List[str]] = backtest_cfg.get("symbols")
    if not symbols:
        print("Error: 'symbols' must be specified in backtest config")
        sys.exit(1)

    # Required: date range
    from_date: Optional[str] = backtest_cfg.get("from_date")
    to_date: Optional[str] = backtest_cfg.get("to_date")
    if not from_date:
        print("Error: 'from_date' must be specified in backtest config")
        sys.exit(1)
    if not to_date:
        to_date = datetime.now().date().isoformat()

    # Optional parameters
    optimize: bool = bool(backtest_cfg.get("optimize", False))
    scan_interval: str = backtest_cfg.get("scan_interval", "D")
    backtest_interval: str = backtest_cfg.get("backtest_interval", "15m")
    max_attempts = backtest_cfg.get("max_attempts")  # None means no limit
    if max_attempts is not None:
        max_attempts = int(max_attempts)

    print(f"[Backtest] strategy={strategy_name} optimize={optimize} symbols={symbols}")
    print(f"[Backtest] from_date={from_date} to_date={to_date} scan_interval={scan_interval} backtest_interval={backtest_interval}")
    print(f"[Backtest] max_attempts={max_attempts}")
    
    journal_df, backtest_run_id = strategy.run_backtest(
        symbols=symbols, 
        from_date=from_date,
        to_date=to_date,
        scan_interval=scan_interval,
        backtest_interval=backtest_interval,
        optimize=optimize,
        max_attempts=max_attempts
    )
    
    # Analyze using generic AnalysisService with returned journal_df
    if backtest_run_id is not None:
        try:
            if not journal_df.empty:
                from analysis.analysis_service import AnalysisService
                analysis = AnalysisService(strategy_name=strategy_name)
                kpi_df = analysis.analyze_and_store(
                    journal_df=journal_df,
                    backtest_run_id=backtest_run_id
                )
                print(f"[Analysis] Completed analysis and storage for backtest #{backtest_run_id}")
            else:
                print(f"[Backtest] Warning: journal DataFrame is empty, skipping analysis")
        except Exception as e:
            print(f"[Backtest] Warning: failed to analyze: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[Backtest] Warning: backtest_run_id is None, skipping analysis")


def do_live(cfg: Dict[str, Any]) -> None:
    """
    Live trading dispatcher. Uses precomputed setups from config and starts live trading.
    Does not perform any scanning.
    """
    strategy_name = cfg.get("strategy", "strat80_20")
    
    # Lazy imports
    from strategies.strategy_loader import load_live_strategy
    from data_manager.config import ensure_directories
    import json
    
    # Load strategy dynamically
    try:
        strategy = load_live_strategy(strategy_name)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Error loading live strategy '{strategy_name}': {e}")
        sys.exit(1)

    # Ensure directories exist
    try:
        ensure_directories()
    except Exception as e:
        print(f"[Live] Warning: failed to ensure directories: {e}")

    # Extract live config
    live_cfg = cfg.get("live", {})
    
    # Required: pre-configured setups must be provided
    provided_setups = live_cfg.get("setups")
    if not provided_setups:
        print("Error: 'live.setups' must be specified and contain at least one setup for live trading")
        sys.exit(1)
    
    # Trading parameters
    timezone = live_cfg.get("timezone", "Asia/Kolkata")
    market_open_hour = int(live_cfg.get("market_open_hour", 9))
    market_open_minute = int(live_cfg.get("market_open_minute", 15))
    market_close_hour = int(live_cfg.get("market_close_hour", 15))
    market_close_minute = int(live_cfg.get("market_close_minute", 30))
    fixed_qty = int(live_cfg.get("fixed_qty", 1))
    product_type = live_cfg.get("product_type", "MIS")
    order_validity = live_cfg.get("order_validity", "DAY")
    max_order_retries = live_cfg.get("max_order_retries", 3)
    if max_order_retries is not None:
        max_order_retries = int(max_order_retries)

    # New optional live controls
    order_timeout_minutes = live_cfg.get("order_timeout_minutes")
    if order_timeout_minutes is not None:
        try:
            order_timeout_minutes = int(order_timeout_minutes)
        except Exception:
            print("[Live] Warning: invalid order_timeout_minutes; using default inside strategy")
            order_timeout_minutes = None
    exit_before_close_minutes = live_cfg.get("exit_before_close_minutes")
    if exit_before_close_minutes is not None:
        try:
            exit_before_close_minutes = int(exit_before_close_minutes)
        except Exception:
            print("[Live] Warning: invalid exit_before_close_minutes; using default inside strategy")
            exit_before_close_minutes = None
    # New: optional ignore_initial_minutes to ignore ticks after market open
    ignore_initial_minutes = live_cfg.get("ignore_initial_minutes")
    if ignore_initial_minutes is not None:
        try:
            ignore_initial_minutes = int(ignore_initial_minutes)
        except Exception:
            print("[Live] Warning: invalid ignore_initial_minutes; defaulting to 1 minute inside strategy")
            ignore_initial_minutes = None

    # Load strategy config for trading parameters
    config_path = os.path.join(ROOT_DIR, 'strategies', strategy_name, 'strat_config.json')
    if not os.path.exists(config_path):
        print(f"Strategy config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        strategy_params = json.load(f)
    
    # Remove opt_ranges (only needed for optimization)
    strategy_params.pop('opt_ranges', None)

    print(f"[Live] strategy={strategy_name} (using precomputed setups)")
    print(f"[Live] timezone={timezone} market_hours={market_open_hour}:{market_open_minute:02d}-{market_close_hour}:{market_close_minute:02d}")
    print(f"[Live] fixed_qty={fixed_qty} product_type={product_type} max_order_retries={max_order_retries}")
    
    # Phase 1: Use pre-configured setups only
    import pandas as pd
    print("\n[Live] Phase 1: Using pre-configured setups (no scan)...")
    setups_df = pd.DataFrame(provided_setups)
    
    # Validate required columns
    required_cols = ['symbol', 'entry_price', 'trigger_price', 'tick_size', 'true_range']
    missing_cols = [col for col in required_cols if col not in setups_df.columns]
    if missing_cols:
        print(f"Error: Pre-configured setups missing required columns: {missing_cols}")
        sys.exit(1)
    
    print(f"[Live] Loaded {len(setups_df)} pre-configured setup(s)")
    
    # Parse max_attempts for per-symbol re-entry limiting (do not limit number of symbols)
    max_attempts = live_cfg.get("max_attempts")  # None means no limit
    if max_attempts is not None:
        try:
            max_attempts = int(max_attempts)
        except Exception:
            print("[Live] Warning: invalid max_attempts; using unlimited per-symbol attempts")
            max_attempts = None
    
    print(f"\n[Live] Total setups to trade: {len(setups_df)}")
    if not setups_df.empty:
        print(setups_df[['symbol', 'setup_date', 'entry_date', 'entry_price', 'trigger_price']].to_string(index=False))
    
    # Phase 2: Start live trading with setups
    print("\n[Live] Phase 2: Starting live trading...")
    strategy.start_trading(
        setups_df=setups_df,
        timezone_str=timezone,
        market_open_hour=market_open_hour,
        market_open_minute=market_open_minute,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
        fixed_qty=fixed_qty,
        product_type=product_type,
        order_validity=order_validity,
        max_attempts=max_attempts,
        max_order_retries=max_order_retries,
        take_profit_mult=strategy_params.get('take_profit_mult', 3.0),
        initial_sl_ticks=strategy_params.get('initial_sl_ticks', 20),
        use_take_profit=strategy_params.get('use_take_profit', False),
        trigger_window_minutes=strategy_params.get('trigger_window_minutes', 60)
        ,order_timeout_minutes=order_timeout_minutes
        ,exit_before_close_minutes=exit_before_close_minutes
        ,ignore_initial_minutes=ignore_initial_minutes
    )


def do_scan(cfg: Dict[str, Any]) -> None:
    """
    Run the strategy's scanner and write results to JSON.
    Config options:
      - scan.symbols: list[str] (required)
      - scan.from_date: ISO date (default: today - 5 days)
      - scan.to_date: ISO date (default: today)
      - scan.interval: e.g., "D" (default: "D")
      - scan.volume_threshold: int (default: 100000)
      - scan.sort_by_volume: bool (default: True)
      - scan.output_file: path (default: ROOT_DIR/scanner_setups.json)
    """
    strategy_name = cfg.get("strategy", "strat80_20")
    
    # Load scanner dynamically via Scanner ABC
    from strategies.strategy_loader import load_scanner_strategy
    try:
        scanner = load_scanner_strategy(strategy_name)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Error loading strategy '{strategy_name}': {e}")
        sys.exit(1)

    # Extract scan config
    scan_cfg = cfg.get("scan", {})

    # Symbols selection logic with support for symbol lists (no global symbols.csv)
    symbols: Optional[List[str]] = scan_cfg.get("symbols")
    use_symbols_csv_val = scan_cfg.get("use_symbols_csv")  # can be str filename or falsy

    def _read_symbols_from_file(path: str, label: str) -> List[str]:
        try:
            # Read file contents once (supports CSV with headers or plain list)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove empty/comment-only lines first to simplify parsing
            raw_lines = [ln for ln in content.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

            syms: List[str] = []

            # Try CSV parsing to pull 'Symbol' column (case-insensitive)
            if raw_lines:
                csv = __import__("csv")
                io = __import__("io")
                reader = csv.reader(io.StringIO("\n".join(raw_lines)))
                try:
                    header = next(reader)
                except StopIteration:
                    header = []

                # Find 'Symbol' column index if present
                idx = -1
                for i, col in enumerate(header):
                    if isinstance(col, str) and col.strip().lower() == "symbol":
                        idx = i
                        break

                if idx >= 0:
                    # Read remaining rows using the Symbol column
                    for row in reader:
                        if idx < len(row):
                            s = (row[idx] or "").strip()
                            if s and not s.startswith("#"):
                                syms.append(s)
                else:
                    # No header or no 'Symbol' column: treat as plain list or first-column CSV
                    for line in raw_lines:
                        # Take first token before comma/whitespace
                        token = line.split(",", 1)[0].strip()
                        if token and not token.startswith("#"):
                            syms.append(token)

            if not syms:
                print(f"Error: {label} at {path} did not contain any symbols")
                sys.exit(1)
            print(f"[Scan] Loaded {len(syms)} symbols from {label}")
            return syms
        except FileNotFoundError:
            print(f"Error: {label} not found at {path}")
            sys.exit(1)

    # Priority:
    # 1) If use_symbols_csv is a non-empty string, read from symbol_list/<file>
    # 2) Else if symbols provided (array), use them
    # 3) Else error
    if isinstance(use_symbols_csv_val, str) and use_symbols_csv_val.strip():
        fname = use_symbols_csv_val.strip()
        list_dir = os.path.join(ROOT_DIR, "symbol_list")
        csv_path = os.path.join(list_dir, fname)
        if not os.path.exists(csv_path) and not fname.lower().endswith(".csv"):
            # Try with .csv appended
            csv_path = os.path.join(list_dir, f"{fname}.csv")
        symbols = _read_symbols_from_file(csv_path, f"symbol_list file '{fname}'")
    elif isinstance(symbols, list) and symbols:
        # Keep as provided
        pass
    else:
        print("Error: Provide either scan.use_symbols_csv (filename under symbol_list) or scan.symbols array")
        sys.exit(1)

    # Dates
    to_date = scan_cfg.get("to_date")
    if not to_date:
        today = __import__("datetime").date.today()
        to_date = today.isoformat()
        from_date = (today - __import__("datetime").timedelta(days=4)).isoformat()
    else:
        from_date = scan_cfg.get("from_date")
        if not from_date:
            try:
                to_dt = __import__("datetime").date.fromisoformat(to_date)
                from_date = (to_dt - __import__("datetime").timedelta(days=4)).isoformat()
            except Exception:
                # Fallback to 4 days before today if parsing fails
                from_date = ( __import__("datetime").date.today() - __import__("datetime").timedelta(days=4) ).isoformat()

    # Other options
    interval = scan_cfg.get("interval", "D")
    volume_threshold = int(scan_cfg.get("volume_threshold", 100000))
    sort_by_volume = bool(scan_cfg.get("sort_by_volume", True))
    output_file = scan_cfg.get("output_file") or os.path.join(ROOT_DIR, "scanner_setups.json")

    # Ensure DB/logs directories exist before importing modules that open the SQLite engine
    try:
        from data_manager.config import ensure_directories
        ensure_directories()
    except Exception as e:
        print(f"[Scan] Warning: failed to ensure directories: {e}")
    
    print(f"[Scan] strategy={strategy_name} symbols={len(symbols)} interval={interval} range={from_date}..{to_date}")
    
    # Collect all setup days across all symbols
    all_setups = []
    for symbol in symbols:
        print(f"[Scan] Processing {symbol}...")
        try:
            setup_df = scanner.get_setup_days(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                volume_threshold=volume_threshold,
                return_details=True
            )
            if not setup_df.empty:
                all_setups.append(setup_df)
                print(f"[Scan] Found {len(setup_df)} setup(s) for {symbol}")
            else:
                print(f"[Scan] No setups found for {symbol}")
        except Exception as e:
            print(f"[Scan] Error processing {symbol}: {e}")
    
    # Combine all results
    if all_setups:
        import pandas as pd
        df = pd.concat(all_setups, ignore_index=True)
        # Sort by volume if requested and available
        if sort_by_volume and 'volume' in df.columns:
            df = df.sort_values('volume', ascending=False)
        print(f"\n[Scan] Total setups found: {len(df)}")
        if not df.empty:
            print(df[['symbol', 'setup_date', 'entry_date', 'entry_price', 'trigger_price']].to_string(index=False))
    else:
        import pandas as pd
        df = pd.DataFrame(columns=['symbol','setup_date','entry_date','entry_price','trigger_price','tick_size','true_range'])
        print("\n[Scan] No setups found across all symbols.")
    
    # Persist to JSON for downstream (e.g., live)
    try:
        df.to_json(output_file, orient="records", date_format="iso")
        print(f"[Scan] Wrote {len(df)} setups to {output_file}")
    except Exception as e:
        print(f"[Scan] Failed to write JSON to {output_file}: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project entrypoint to run backtests or go live based on config.")
    parser.add_argument("--config", "-c", default="run_config.json", help="Path to JSON configuration file")
    args = parser.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT_DIR, args.config)
    if not os.path.exists(cfg_path):
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    cfg = load_config(cfg_path)

    # Optional DB path can be provided in run_config and exported for data_manager.config
    db_path = cfg.get("db_path")
    if db_path:
        if not os.path.isabs(db_path):
            db_path = os.path.join(ROOT_DIR, db_path)
        os.environ["DB_PATH"] = db_path

    mode = (cfg.get("mode") or "backtest").lower()

    if mode == "backtest":
        do_backtest(cfg)
    elif mode == "live":
        do_live(cfg)
    elif mode == "scan":
        do_scan(cfg)
    else:
        print(f"Unknown mode '{mode}'. Use 'backtest', 'live', or 'scan'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
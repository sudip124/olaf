import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Make sure project root is on sys.path when running from subfolders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def do_backtest(cfg: Dict[str, Any]) -> None:
    # Lazy import to ensure DB_PATH from config/env is set before modules load
    from strategies.strat80_20.backtest import run_backtest

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
    if not from_date or not to_date:
        print("Error: 'from_date' and 'to_date' must be specified in backtest config")
        sys.exit(1)

    # Optional parameters
    optimize: bool = bool(backtest_cfg.get("optimize", False))
    scan_interval: str = backtest_cfg.get("scan_interval", "D")
    backtest_interval: str = backtest_cfg.get("backtest_interval", "15m")

    print(f"[Backtest] strategy=strat80_20 optimize={optimize} symbols={symbols}")
    print(f"[Backtest] from_date={from_date} to_date={to_date} scan_interval={scan_interval} backtest_interval={backtest_interval}")
    
    run_backtest(
        symbols=symbols, 
        from_date=from_date,
        to_date=to_date,
        scan_interval=scan_interval,
        backtest_interval=backtest_interval,
        optimize=optimize
    )


def do_live(cfg: Dict[str, Any]) -> None:
    """
    Live trading dispatcher. Runs scanner and starts live trading.
    Similar to do_backtest() but for live trading.
    """
    strategy = cfg.get("strategy", "strat80_20")

    if strategy == "strat80_20":
        # Lazy imports
        from strategies.strat80_20.live import start_trading
        from strategies.strat80_20.scanner_long import get_setup_days
        from data_manager.config import ensure_directories
        import json

        # Ensure directories exist
        try:
            ensure_directories()
        except Exception as e:
            print(f"[Live] Warning: failed to ensure directories: {e}")

        # Extract live config
        live_cfg = cfg.get("live", {})
        
        # Check if setups are provided directly (skip scanning)
        provided_setups = live_cfg.get("setups")
        
        # Required: either symbols to scan OR pre-configured setups
        symbols: Optional[List[str]] = live_cfg.get("symbols")
        if not symbols and not provided_setups:
            print("Error: Either 'symbols' or 'setups' must be specified in live config")
            sys.exit(1)

        # Dates for scanning (default: today - 5 days to today)
        to_date = live_cfg.get("to_date")
        if not to_date:
            to_date = __import__("datetime").date.today().isoformat()
        from_date = live_cfg.get("from_date")
        if not from_date:
            from_date = (__import__("datetime").date.today() - __import__("datetime").timedelta(days=5)).isoformat()

        # Scanner parameters
        scan_interval = live_cfg.get("scan_interval", "D")
        volume_threshold = int(live_cfg.get("volume_threshold", 100000))
        sort_by_volume = bool(live_cfg.get("sort_by_volume", True))
        max_entries = live_cfg.get("max_entries")  # None means no limit
        if max_entries is not None:
            max_entries = int(max_entries)
        
        # Trading parameters
        timezone = live_cfg.get("timezone", "Asia/Kolkata")
        fixed_qty = int(live_cfg.get("fixed_qty", 1))
        product_type = live_cfg.get("product_type", "MIS")
        order_validity = live_cfg.get("order_validity", "DAY")

        # Load strategy config for trading parameters
        config_path = os.path.join(ROOT_DIR, 'strategies', 'strat80_20', 'strat_config.json')
        if not os.path.exists(config_path):
            print(f"Strategy config not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            strategy_params = json.load(f)
        
        # Remove opt_ranges (only needed for optimization)
        strategy_params.pop('opt_ranges', None)

        print(f"[Live] strategy={strategy} symbols={symbols}")
        print(f"[Live] scan_range={from_date}..{to_date} interval={scan_interval}")
        print(f"[Live] timezone={timezone} fixed_qty={fixed_qty} product_type={product_type}")
        
        # Phase 1: Get setups (either from config or by scanning)
        import pandas as pd
        
        if provided_setups:
            # Use pre-configured setups (skip scanning)
            print("\n[Live] Phase 1: Using pre-configured setups (skipping scan)...")
            setups_df = pd.DataFrame(provided_setups)
            
            # Validate required columns
            required_cols = ['symbol', 'entry_price', 'trigger_price', 'tick_size', 'true_range']
            missing_cols = [col for col in required_cols if col not in setups_df.columns]
            if missing_cols:
                print(f"Error: Pre-configured setups missing required columns: {missing_cols}")
                sys.exit(1)
            
            print(f"[Live] Loaded {len(setups_df)} pre-configured setup(s)")
        else:
            # Run scanner to get today's setups
            print("\n[Live] Phase 1: Scanning for setups...")
            print(f"[Live] scan_range={from_date}..{to_date} interval={scan_interval} symbols={symbols}")
            
            all_setups = []
            for symbol in symbols:
                print(f"[Live] Scanning {symbol}...")
                try:
                    setup_df = get_setup_days(
                        symbol=symbol,
                        from_date=from_date,
                        to_date=to_date,
                        interval=scan_interval,
                        volume_threshold=volume_threshold,
                        return_details=True
                    )
                    if not setup_df.empty:
                        # Only take the latest setup for live trading
                        latest_setup = setup_df.iloc[-1:]
                        all_setups.append(latest_setup)
                        print(f"[Live] Found setup for {symbol}: entry_price={latest_setup['entry_price'].values[0]}")
                    else:
                        print(f"[Live] No setups found for {symbol}")
                except Exception as e:
                    print(f"[Live] Error scanning {symbol}: {e}")
            
            if not all_setups:
                print("\n[Live] No setups found. Exiting.")
                sys.exit(0)
            
            # Combine all setups
            setups_df = pd.concat(all_setups, ignore_index=True)
        
        # Sort by volume if requested
        if sort_by_volume:
            setups_df = setups_df.sort_values('volume', ascending=False)
        
        if max_entries is not None and len(setups_df) > max_entries:
            print(f"\n[Live] Found {len(setups_df)} setups, limiting to {max_entries} entries")
            setups_df = setups_df.head(max_entries)
        
        print(f"\n[Live] Total setups to trade: {len(setups_df)}")
        if not setups_df.empty:
            print(setups_df[['symbol', 'setup_date', 'entry_date', 'entry_price', 'trigger_price', 'volume']].to_string(index=False))
        
        # Phase 2: Start live trading with setups
        print("\n[Live] Phase 2: Starting live trading...")
        start_trading(
            setups_df=setups_df,
            timezone=timezone,
            fixed_qty=fixed_qty,
            product_type=product_type,
            order_validity=order_validity,
            max_entries=max_entries,
            take_profit_mult_param=strategy_params.get('take_profit_mult', 3.0),
            use_take_profit_param=strategy_params.get('use_take_profit', False),
            trigger_window_minutes_param=strategy_params.get('trigger_window_minutes', 60)
        )
    else:
        print(f"Live mode not implemented for strategy '{strategy}' yet.")
        sys.exit(1)


def do_scan(cfg: Dict[str, Any]) -> None:
    """
    Run the strategy's scanner and write results to JSON. Currently supports strat80_20 scanner_long.
    Config options:
      - scan.symbols: list[str] (required)
      - scan.from_date: ISO date (default: today - 5 days)
      - scan.to_date: ISO date (default: today)
      - scan.interval: e.g., "D" (default: "D")
      - scan.volume_threshold: int (default: 100000)
      - scan.sort_by_volume: bool (default: True)
      - scan.output_file: path (default: ROOT_DIR/scanner_setups.json)
    """
    strategy = cfg.get("strategy", "strat80_20")

    # Extract scan config
    scan_cfg = cfg.get("scan", {})

    # Symbols (required)
    symbols: Optional[List[str]] = scan_cfg.get("symbols")
    if not symbols:
        print("Error: 'symbols' must be specified in scan config")
        sys.exit(1)

    # Dates
    to_date = scan_cfg.get("to_date")
    if not to_date:
        to_date = __import__("datetime").date.today().isoformat()
    from_date = scan_cfg.get("from_date")
    if not from_date:
        from_date = ( __import__("datetime").date.today() - __import__("datetime").timedelta(days=5) ).isoformat()

    # Other options
    interval = scan_cfg.get("interval", "D")
    volume_threshold = int(scan_cfg.get("volume_threshold", 100000))
    sort_by_volume = bool(scan_cfg.get("sort_by_volume", True))
    output_file = scan_cfg.get("output_file") or os.path.join(ROOT_DIR, "scanner_setups.json")

    if strategy == "strat80_20":
        # Ensure DB/logs directories exist before importing modules that open the SQLite engine
        try:
            from data_manager.config import ensure_directories
            ensure_directories()
        except Exception as e:
            print(f"[Scan] Warning: failed to ensure directories: {e}")
        from strategies.strat80_20.scanner_long import get_setup_days  # lazy import
        print(f"[Scan] strategy={strategy} symbols={len(symbols)} interval={interval} range={from_date}..{to_date}")
        
        # Collect all setup days across all symbols
        all_setups = []
        for symbol in symbols:
            print(f"[Scan] Processing {symbol}...")
            try:
                setup_df = get_setup_days(
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
            # Sort by volume if requested
            if sort_by_volume:
                df = df.sort_values('volume', ascending=False)
            print(f"\n[Scan] Total setups found: {len(df)}")
            if not df.empty:
                print(df[['symbol', 'setup_date', 'entry_date', 'entry_price', 'trigger_price', 'volume']].to_string(index=False))
        else:
            df = pd.DataFrame(columns=['symbol','setup_date','entry_date','entry_price','trigger_price','tick_size','true_range','volume','open_pos','close_pos'])
            print("\n[Scan] No setups found across all symbols.")
        
        # Persist to JSON for downstream (e.g., live)
        try:
            df.to_json(output_file, orient="records", date_format="iso")
            print(f"[Scan] Wrote {len(df)} setups to {output_file}")
        except Exception as e:
            print(f"[Scan] Failed to write JSON to {output_file}: {e}")
            sys.exit(1)
    else:
        print(f"Scan mode not implemented for strategy '{strategy}' yet.")
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
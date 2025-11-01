import vectorbt as vbt
import pandas as pd
import numpy as np
import datetime
import argparse
import os
import json
from strategies.strat80_20.backtest_config import INIT_CASH, FEES, EXCHANGE, SESSION_START
from data_manager.data_fetcher import fetch_historical_data, fetch_instrument_info
from strategies.strat80_20.scanner_long import get_setup_days
from strategies.strat80_20.strat80_20 import generate_signals
from strategies.strat80_20.db_models import save_backtest_data, save_trade_logs, analyze_logs_from_db

# analyze_logs function removed - now using analyze_logs_from_db from db_models

def run_backtest(symbols, from_date, to_date, scan_interval='D', backtest_interval='15m', optimize=False):
    """
    Run backtest for strat80_20 strategy.
    
    Args:
        symbols: List of symbols to backtest
        from_date: Start date for backtest (ISO format)
        to_date: End date for backtest (ISO format)
        scan_interval: Interval for setup day detection (default: 'D' for daily)
        backtest_interval: Interval for actual backtest (default: '15m')
        optimize: Whether to run parameter optimization (default: False)
    
    Returns:
        DataFrame with backtest results
    """
    if not symbols:
        raise ValueError("symbols parameter is required")
    if not from_date or not to_date:
        raise ValueError("from_date and to_date parameters are required")

    print(f"\n=== Strat80_20 Backtest Configuration ===")
    print(f"Date Range: {from_date} to {to_date}")
    print(f"Scan Interval (for setup detection): {scan_interval}")
    print(f"Backtest Interval: {backtest_interval}")
    print(f"Symbols: {symbols}")
    print(f"Optimize: {optimize}")
    print(f"==========================================\n")

    # Phase 1: Fetch setup days using scanner on daily interval
    setup_days_map = {}
    setup_days_detailed = {}  # Store detailed DataFrames for database
    for symbol in symbols:
        print(f"[Phase 1] Scanning for setup days: {symbol} (interval={scan_interval})...")
        try:
            # Get detailed setup information for database storage
            setup_df = get_setup_days(symbol, from_date, to_date, interval=scan_interval, return_details=True)
            setup_days_detailed[symbol] = setup_df
            
            # Extract just the dates for backward compatibility with generate_signals
            if not setup_df.empty:
                setup_days = setup_df['setup_date'].tolist()
            else:
                setup_days = []
            
            setup_days_map[symbol] = setup_days
            print(f"  -> Found {len(setup_days)} setup day(s) for {symbol}")
        except Exception as e:
            print(f"  -> Warning: failed to fetch setup days for {symbol}: {e}")
            setup_days_map[symbol] = []
            setup_days_detailed[symbol] = pd.DataFrame()

    # Phase 2: Fetch intraday data for backtesting
    # Only process symbols that successfully completed Phase 1
    data_cache = {}
    symbols_to_backtest = [s for s in symbols if s in setup_days_map and setup_days_map[s] is not None]
    
    for symbol in symbols_to_backtest:
        print(f"[Phase 2] Fetching backtest data: {symbol} (interval={backtest_interval})...")
        try:
            df = fetch_historical_data(symbol, backtest_interval, from_date, to_date)

            if df.empty:
                print(f"  -> No data for {symbol}. Skipping.")
                continue
            data_cache[symbol] = df
            print(f"  -> Loaded {len(df)} bars for {symbol}")
        except Exception as e:
            print(f"  -> Error fetching data for {symbol}: {e}. Skipping.")
            continue

    # Load strategy config
    config_path = os.path.join('strategies', 'strat80_20', 'strat_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    
    with open(config_path, 'r') as f:
        params_config = json.load(f)
    opt_ranges = params_config.pop('opt_ranges', {}) if optimize else {}
    defaults = params_config  # Remaining flat params are defaults
    
    # Save backtest run and setup days to database
    # Only save symbols that have data to backtest
    backtest_run_id = None  # Initialize to None
    try:
        # Filter setup_days_detailed to only include symbols with data
        filtered_setup_days = {s: setup_days_detailed[s] for s in data_cache.keys() if s in setup_days_detailed}
        
        backtest_run_id = save_backtest_data(
            from_date=from_date,
            to_date=to_date,
            scan_interval=scan_interval,
            backtest_interval=backtest_interval,
            config_params=defaults,
            setup_days_map=filtered_setup_days,
            optimize=optimize,
            strategy_name='strat80_20'
        )
        print(f"[Database] Backtest run saved with ID: {backtest_run_id}")
    except Exception as db_error:
        print(f"[Database] Warning: Failed to save to database: {db_error}")
        print("[Database] Continuing with backtest...")

    results = {}
    all_trade_logs = []  # Collect all trade logs for database storage
    
    for symbol, df in data_cache.items():
        # Fetch tick size dynamically
        instrument_info = fetch_instrument_info(symbol, exchange=EXCHANGE)
        tick_size = instrument_info.get('tick_size', 0.05)

        price = df['close']

        if optimize:
            # Parameter optimization
            best_params, best_return = None, -np.inf
            returns = pd.Series(index=pd.MultiIndex.from_product(opt_ranges.values(), names=opt_ranges.keys()), dtype=float)

            for param_combo in pd.MultiIndex.from_product(opt_ranges.values(), names=opt_ranges.keys()):
                opt_params = dict(zip(opt_ranges.keys(), param_combo))
                
                # Merge defaults with optimization parameters
                merged_params = {**defaults, **opt_params}

                order_size, order_price, _ = generate_signals(
                    df=df,
                    symbol=symbol,
                    setup_days=setup_days_map.get(symbol, []),
                    take_profit_mult=merged_params.get('take_profit_mult'),
                    tick=tick_size,
                    use_take_profit=merged_params.get('use_take_profit'),
                    trigger_tick_mult=merged_params.get('trigger_tick_mult'),
                    trigger_window_minutes=merged_params.get('trigger_window_minutes'),
                    session_start=SESSION_START,
                    log=False  # Don't log during optimization iterations
                )

                pf_opt = vbt.Portfolio.from_orders(
                    close=price,
                    size=order_size,
                    price=order_price,
                    init_cash=INIT_CASH,
                    fees=FEES,
                    freq=backtest_interval.replace('m', 'min'),
                    direction='longonly',
                    size_type='amount'
                )
                total_return = pf_opt.total_return()
                returns.loc[param_combo] = total_return

                if total_return > best_return:
                    best_return = total_return
                    best_params = opt_params

            if not returns.empty:
                fig = returns.vbt.heatmap(x_level=list(opt_ranges.keys())[1] if len(opt_ranges) > 1 else None,
                                          y_level=list(opt_ranges.keys())[0])
                fig.write_image(f'logs/heatmap_strat80_20_{symbol}.png')

            # Merge defaults with best parameters
            final_params = {**defaults, **best_params}
            
            order_size, order_price, trade_logs = generate_signals(
                df=df,
                symbol=symbol,
                setup_days=setup_days_map.get(symbol, []),
                take_profit_mult=final_params.get('take_profit_mult'),
                tick=tick_size,
                use_take_profit=final_params.get('use_take_profit'),
                trigger_tick_mult=final_params.get('trigger_tick_mult'),
                trigger_window_minutes=final_params.get('trigger_window_minutes'),
                session_start=SESSION_START,
                log=True
            )
            all_trade_logs.extend(trade_logs)
        else:
            # Single run with defaults
            order_size, order_price, trade_logs = generate_signals(
                df=df,
                symbol=symbol,
                setup_days=setup_days_map.get(symbol, []),
                take_profit_mult=defaults.get('take_profit_mult'),
                tick=tick_size,
                use_take_profit=defaults.get('use_take_profit'),
                trigger_tick_mult=defaults.get('trigger_tick_mult'),
                trigger_window_minutes=defaults.get('trigger_window_minutes'),
                session_start=SESSION_START,
                log=True
            )
            all_trade_logs.extend(trade_logs)

        pf = vbt.Portfolio.from_orders(
            close=price,
            size=order_size,
            price=order_price,
            init_cash=INIT_CASH,
            fees=FEES,
            freq=backtest_interval.replace('m', 'min'),
            direction='longonly',
            size_type='amount'
        )

        stats = pf.stats()

        # Process trades
        trades = pf.trades.records.copy()
        
        if not trades.empty:
            # Create new columns with time information using .loc
            trades.loc[:, 'entry_time'] = pd.Series(
                [price.index[x] if x < len(price.index) else pd.NaT for x in trades['entry_idx']],
                index=trades.index
            )
            trades.loc[:, 'entry_date'] = pd.Series(
                [ts.date() if pd.notna(ts) else None for ts in trades['entry_time']],
                index=trades.index
            )
            
            trades.loc[:, 'exit_time'] = pd.Series(
                [price.index[x] if x < len(price.index) else pd.NaT for x in trades['exit_idx']],
                index=trades.index
            )
            trades.loc[:, 'exit_date'] = pd.Series(
                [ts.date() if pd.notna(ts) else None for ts in trades['exit_time']],
                index=trades.index
            )
        
            wins = trades[trades['pnl'] > 0]['pnl']
            losses = trades[trades['pnl'] < 0]['pnl']
            avg_win = wins.mean() if not wins.empty else 0
            avg_loss = losses.mean() if not losses.empty else 0
            max_win = wins.max() if not wins.empty else 0
            max_loss = losses.min() if not losses.empty else 0
            daily_trade_count = trades.groupby('entry_date').size().to_dict()
        else:
            avg_win = avg_loss = max_win = max_loss = 0
            daily_trade_count = {}

        combined_stats = {
            **stats,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'daily_trade_count': daily_trade_count
        }
        if optimize:
            combined_stats['best_params'] = best_params

        results[symbol] = combined_stats
        # print(f"Key KPIs for {symbol}:\n{combined_stats}\n")
    
    # Save all trade logs to database
    if all_trade_logs and backtest_run_id is not None:
        try:
            save_trade_logs(backtest_run_id, all_trade_logs, strategy_name='strat80_20')
        except Exception as log_error:
            print(f"[Database] Warning: Failed to save trade logs: {log_error}")
    
    # Analyze logs from database and add to results
    if backtest_run_id is not None:
        try:
            log_metrics = analyze_logs_from_db(backtest_run_id, strategy_name='strat80_20')
            # Add log metrics to each symbol's results
            for symbol in results:
                results[symbol].update(log_metrics)
        except Exception as analysis_error:
            print(f"[Database] Warning: Failed to analyze logs: {analysis_error}")

    # Build results dataframe
    if results:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.index.name = 'symbol'

        key_metrics = [
            'Total Trades', 'Win Rate [%]', 'Sharpe Ratio', 'avg_win', 'avg_loss',
            'max_win', 'max_loss', 'total_setups', 'total_success', 'total_missed',
            'exit_counts', 'total_reentries'
        ]
        print("Pool Aggregates:")
        agg_stats = {
            'mean_sharpe': results_df['Sharpe Ratio'].mean(),
            'total_trades_all': results_df['Total Trades'].sum(),
            'mean_win_rate': results_df['Win Rate [%]'].mean(),
        }
        print(agg_stats)

    return results_df if results_df is not None else pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest strat80_20 strategy.")
    parser.add_argument('--symbols', nargs='+', required=True, help="List of symbols to backtest")
    parser.add_argument('--from-date', type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--to-date', type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument('--scan-interval', type=str, default='D', help="Interval for setup detection (default: D)")
    parser.add_argument('--backtest-interval', type=str, default='15m', help="Interval for backtest (default: 15m)")
    parser.add_argument('--optimize', action='store_true', default=False, help="Enable parameter optimization")
    args = parser.parse_args()

    run_backtest(
        symbols=args.symbols,
        from_date=args.from_date,
        to_date=args.to_date,
        scan_interval=args.scan_interval,
        backtest_interval=args.backtest_interval,
        optimize=args.optimize
    )
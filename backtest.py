import vectorbt as vbt
import pandas as pd
import numpy as np
import datetime
import argparse
import os
import json  # Added for loading strategy configs
from backtest_config import INIT_CASH, FEES, EXCHANGE, FROM_DATE, TO_DATE, SESSION_START, SESSION_END, SYMBOLS, INTERVAL, DEFAULT_STRATEGY
from data_fetcher import fetch_historical_data, fetch_instrument_info
from strategies import STRATEGY_REGISTRY

def analyze_logs(logfile):
    """
    Analyze the signal log CSV to extract custom KPIs.
    Returns a dict with aggregated metrics.
    """
    if not os.path.exists(logfile):
        return {}

    log_df = pd.read_csv(logfile)
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
    # Use existing 'date' and 'time' columns if present, else compute
    if 'date' not in log_df.columns:
        log_df['date'] = log_df['timestamp'].dt.date
    if 'time' not in log_df.columns:
        log_df['time'] = log_df['timestamp'].dt.time

    # Daily setup count (signal generation)
    setups = log_df[log_df['event'] == 'Setup Detected']
    daily_setups = setups.groupby('date').size().to_dict()
    total_setups = len(setups)

    # Daily success count (entries following setups)
    entries = log_df[log_df['event'] == 'Entry Filled']
    # Setups are on prev_day for entry next day, so shift setup date +1
    setups['entry_date'] = pd.to_datetime(setups['date']) + datetime.timedelta(days=1)
    setups['entry_date'] = setups['entry_date'].dt.date
    successful_setups = setups[setups['entry_date'].isin(entries['date'])]
    daily_success = successful_setups.groupby('entry_date').size().to_dict()
    total_success = len(successful_setups)

    # Daily missed signals (setups without entry)
    missed_setups = setups[~setups['entry_date'].isin(entries['date'])]
    daily_missed = missed_setups.groupby('entry_date').size().to_dict()
    total_missed = len(missed_setups)

    # Trade exit trigger counts
    exits_sl = log_df[log_df['event'] == 'Stop Loss Exit']
    exits_tp = log_df[log_df['event'] == 'Take Profit Exit']
    exits_tsl = log_df[log_df['event'].str.contains('Trailing SL')]  # Count updates, but for exits, approximate
    exit_counts = {
        'stop_loss': len(exits_sl),
        'take_profit': len(exits_tp),
        'trailing_sl_updates': len(exits_tsl),
    }

    # Count of trade re-entries for the same setup day (multiple entries per day)
    daily_entries = entries.groupby('date').size()
    reentry_days = daily_entries[daily_entries > 1]
    total_reentries = (daily_entries - 1).clip(lower=0).sum()

    return {
        'total_setups': total_setups,
        'daily_setups': daily_setups,
        'total_success': total_success,
        'daily_success': daily_success,
        'total_missed': total_missed,
        'daily_missed': daily_missed,
        'exit_counts': exit_counts,
        'total_reentries': total_reentries,
        'reentry_days': reentry_days.to_dict()
    }

def run_multi_backtest(strategies=None, symbols=None, optimize=True):
    """
    Run backtests for multiple strategies and symbols from config or CLI.
    """
    if strategies is None:
        strategies = [DEFAULT_STRATEGY]
    if isinstance(strategies, str):
        strategies = [strategies]
    if symbols is None:
        symbols = SYMBOLS

    # Cache data to avoid repeated fetches
    data_cache = {}
    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = fetch_historical_data(symbol, INTERVAL, FROM_DATE, TO_DATE, exchange=EXCHANGE)
        if df.empty:
            print(f"No data for {symbol}. Skipping.")
            continue
        data_cache[symbol] = df

    results = {}
    for strat_name in strategies:
        if strat_name not in STRATEGY_REGISTRY:
            print(f"Strategy {strat_name} not found in registry: {list(STRATEGY_REGISTRY.keys())}")
            continue
        signal_func = STRATEGY_REGISTRY[strat_name]

        # Load strategy-specific config from JSON
        config_path = os.path.join('strategies', f'{strat_name}_config.json')
        if not os.path.exists(config_path):
            print(f"Config file {config_path} not found for {strat_name}. Skipping.")
            continue
        with open(config_path, 'r') as f:
            params_config = json.load(f)
        opt_ranges = params_config.pop('opt_ranges', {}) if optimize else {}
        defaults = params_config  # Remaining flat params are defaults

        for symbol, df in data_cache.items():
            # Fetch tick size dynamically
            instrument_info = fetch_instrument_info(symbol, exchange=EXCHANGE)
            tick_size = instrument_info.get('tick_size', 0.05)

            price = df['close']
            logfile = f"logs/{strat_name}_{symbol}_signals_{datetime.date.today().isoformat()}.csv"

            if optimize:
                # Parameter optimization (extend your loop logic)
                best_params, best_return = None, -np.inf
                returns = pd.Series(index=pd.MultiIndex.from_product(opt_ranges.values(), names=opt_ranges.keys()), dtype=float)

                for param_combo in pd.MultiIndex.from_product(opt_ranges.values(), names=opt_ranges.keys()):
                    opt_params = dict(zip(opt_ranges.keys(), param_combo))
                    order_size, order_price = signal_func(df, symbol=symbol, tick=tick_size, log=True, logfile=logfile, **{**defaults, **opt_params})

                    pf_opt = vbt.Portfolio.from_orders(
                        close=price,
                        size=order_size,
                        price=order_price,
                        init_cash=INIT_CASH,
                        fees=FEES,
                        freq=INTERVAL.replace('m', 'min'),
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
                    fig.write_image(f'logs/heatmap_{strat_name}_{symbol}.png')

                order_size, order_price = signal_func(df, symbol=symbol, tick=tick_size, log=True, logfile=logfile, **{**defaults, **best_params})
            else:
                # Single run with defaults
                # Before calling signal_func, remove 'tick' from defaults to avoid duplicate
                defaults_no_tick = {k: v for k, v in defaults.items() if k != 'tick'}

                logfile = f"logs/{strat_name}_{symbol}_signals_{datetime.date.today().isoformat()}.csv"
                order_size, order_price = signal_func(
                    df, symbol=symbol, tick=tick_size, log=True, logfile=logfile, **defaults_no_tick
                )

            pf = vbt.Portfolio.from_orders(
                close=price,
                size=order_size,
                price=order_price,
                init_cash=INIT_CASH,
                fees=FEES,
                freq=INTERVAL.replace('m', 'min'),
                direction='longonly',
                size_type='amount'
            )

            stats = pf.stats()

            # Before processing trades, create an explicit copy
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
                
                # If you need exit times too
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

            log_metrics = analyze_logs(logfile)

            combined_stats = {
                **stats,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win': max_win,
                'max_loss': max_loss,
                'daily_trade_count': daily_trade_count,
                **log_metrics
            }
            if optimize:
                combined_stats['best_params'] = best_params

            results[(strat_name, symbol)] = combined_stats
            print(f"Key KPIs for {strat_name} on {symbol}:\n{combined_stats}\n")

    # Initialize results_df as None
    results_df = None
    
    if results:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.index.names = ['strategy', 'symbol']

        key_metrics = [
            'Total Trades', 'Win Rate [%]', 'Sharpe Ratio', 'avg_win', 'avg_loss',
            'max_win', 'max_loss', 'total_setups', 'total_success', 'total_missed',
            'exit_counts', 'total_reentries'
        ]
        print("Aggregated KPIs Across Symbols:")
        print(results_df[key_metrics])

        today = datetime.date.today().isoformat()
        results_df.to_csv(f'logs/multi_backtest_kpis_{today}.csv')

        agg_stats = {
            'mean_sharpe': results_df['Sharpe Ratio'].mean(),
            'total_trades_all': results_df['Total Trades'].sum(),
            'mean_win_rate': results_df['Win Rate [%]'].mean(),
        }
        print("Pool Aggregates:", agg_stats)

    return results_df if results_df is not None else pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest intraday strategies with custom KPIs.")
    parser.add_argument('--strategy', type=str, default=DEFAULT_STRATEGY)
    parser.add_argument('--optimize', action='store_true', default=False)
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS)
    args = parser.parse_args()

    # Use the strategy name as-is without replacing hyphens
    run_multi_backtest(strategies=[args.strategy], symbols=args.symbols, optimize=args.optimize)
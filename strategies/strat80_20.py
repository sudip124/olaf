# Modified strat80_20.py
import pandas as pd
import numpy as np
import talib
import json
import os
from backtest_config import SESSION_START  # No need for data_fetcher anymore
import datetime

def generate_signals(df: pd.DataFrame, symbol: str = '', logfile: str = None, **kwargs) -> tuple:
    """
    Generate buy signals based on the modified 80-20 strategy: If the previous day's open is in the top 20% of the true range 
    and close is in the bottom 20%, check the next day's 15m data. If the price drops 10 ticks below the previous day's low 
    within the first 60 minutes, place a buy stop order at the previous day's low. Initial stop loss is the Day 2 low up 
    to the entry point. Uses a loop for state management. Includes take profit based on true range.
    
    Refactored to use provided 15m df and resample to daily for setup detection. No additional data fetching.
    
    Added trailing stop loss: After entry, trail the stop loss to the low of each subsequent 15m green bar (close >= open),
    moving it higher if the green bar's low is above the current stop loss.

    Added configuration for use_take_profit: If False, do not exit on take_profit and rely solely on trailing stop loss.
    """
    # Load default parameters from JSON
    config_path = os.path.join(os.path.dirname(__file__), 'strat80_20_config.json')
    with open(config_path, 'r') as f:
        params = json.load(f)
    
    # Override with provided kwargs
    params.update(kwargs)

    # Determine if logging is enabled
    log = params.get('log', False)
    if log:
        signals_log = []  # List of dicts for structured logging

    # Extract parameters
    stop_loss_mult = params.get('stop_loss_mult', 2.0)  # Currently unused, but included for future/optimization
    take_profit_mult = params.get('take_profit_mult', 3.0)
    tick = params.get('tick', 0.05)  # Default tick size if not provided
    use_take_profit = params.get('use_take_profit', True)  # New parameter to toggle take profit exits
    open_pos_threshold = params.get('open_pos_threshold', 0.8)
    close_pos_threshold = params.get('close_pos_threshold', 0.2)
    trigger_tick_mult = params.get('trigger_tick_mult', 10)
    trigger_window_minutes = params.get('trigger_window_minutes', 60)

    # Ensure required columns
    if 'open' not in df.columns:
        df['open'] = df['close']  # Fallback if no open
    if 'volume' not in df.columns:
        df['volume'] = np.nan  # Fallback if no volume

    # Resample 15m data to daily for setup detection
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()  # Drop non-trading days

    # Compute true range and positions for setups
    daily_df['prev_close'] = daily_df['close'].shift(1)
    daily_df['true_high'] = np.maximum(daily_df['high'], daily_df['prev_close'].fillna(daily_df['open']))
    daily_df['true_low'] = np.minimum(daily_df['low'], daily_df['prev_close'].fillna(daily_df['open']))
    daily_df['tr'] = daily_df['true_high'] - daily_df['true_low']
    daily_df['open_pos'] = (daily_df['open'] - daily_df['true_low']) / daily_df['tr'].replace(0, np.nan)
    daily_df['close_pos'] = (daily_df['close'] - daily_df['true_low']) / daily_df['tr'].replace(0, np.nan)
    daily_df['setup_long'] = (daily_df['open_pos'] >= open_pos_threshold) & (daily_df['close_pos'] <= close_pos_threshold)

    # Prepare signal series
    order_size = pd.Series(0.0, index=df.index)  # Size: +inf buy all, -inf sell all
    order_price = pd.Series(np.nan, index=df.index)  # Custom fill price

    # Global state variables
    in_long = False
    long_entry_price = 0.0
    entry_tr = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    # Per-day state variables
    potential_entry = False
    entry_price = 0.0
    threshold = 0.0
    trigger_time = None
    day_low_so_far = np.inf
    session_start = None
    first60_end = None
    current_day = None

    # Loop over 15m bars
    for i in range(len(df)):
        this_time = df.index[i]
        this_day = this_time.date()

        if current_day != this_day:
            # Reset per-day states for new day
            potential_entry = False
            trigger_time = None
            day_low_so_far = np.inf

            # Check if previous day had setup
            prev_day = this_day - datetime.timedelta(days=1)
            prev_day_times = daily_df.index[daily_df.index.date == prev_day]
            if not prev_day_times.empty:
                p_idx = prev_day_times[0]
                if daily_df.at[p_idx, 'setup_long']:
                    potential_entry = True
                    entry_price = daily_df.at[p_idx, 'low']
                    threshold = entry_price - trigger_tick_mult * tick
                    entry_tr = daily_df.at[p_idx, 'tr']

                    if log:
                        signals_log.append({
                            'timestamp': p_idx,
                            'date': p_idx.date().isoformat(),
                            'time': p_idx.time().isoformat(),
                            'event': 'Setup Detected',
                            'symbol': symbol,
                            'price': entry_price,
                            'details': f"80-20 setup for next day entry at {entry_price}; threshold: {threshold}; TR: {entry_tr}"
                        })

                    # Set session times for the current day
                    session_start_str = f"{this_day.isoformat()} {SESSION_START}"
                    session_start = pd.to_datetime(session_start_str)
                    first60_end = session_start + pd.Timedelta(minutes=trigger_window_minutes)

                    # Handle timezone awareness
                    if df.index.tz is not None and session_start.tzinfo is None:
                        session_start = session_start.tz_localize(df.index.tz)
                        first60_end = first60_end.tz_localize(df.index.tz)
                    elif df.index.tz is not None and session_start.tzinfo is not None:
                        session_start = session_start.tz_convert(df.index.tz)
                        first60_end = first60_end.tz_convert(df.index.tz)

            current_day = this_day

        # Update day_low_so_far if potential entry day
        if potential_entry:
            day_low_so_far = min(day_low_so_far, df['low'].iloc[i])

        # Entry logic
        if not in_long:
            if potential_entry:
                if session_start <= this_time < first60_end:
                    if df['low'].iloc[i] <= threshold and trigger_time is None:
                        trigger_time = this_time
                        if log:
                            signals_log.append({
                                'timestamp': this_time,
                                'date': this_time.date().isoformat(),
                                'time': this_time.time().isoformat(),
                                'event': 'Trigger Threshold Hit',
                                'symbol': symbol,
                                'price': df['low'].iloc[i],
                                'details': f"Price dropped below threshold {threshold}; day_low_so_far: {day_low_so_far}"
                            })

                if trigger_time is not None and this_time > trigger_time:
                    if df['high'].iloc[i] >= entry_price:
                        fill_price = entry_price
                        fill_price = round(fill_price / tick) * tick

                        order_size.iloc[i] = np.inf
                        order_price.iloc[i] = fill_price
                        in_long = True
                        long_entry_price = fill_price
                        stop_loss = day_low_so_far - 1e-8
                        risk = long_entry_price - stop_loss
                        take_profit = long_entry_price + take_profit_mult * risk

                        if log:
                            signals_log.append({
                                'timestamp': this_time,
                                'date': this_time.date().isoformat(),
                                'time': this_time.time().isoformat(),
                                'event': 'Entry Filled',
                                'symbol': symbol,
                                'price': fill_price,
                                'details': f"Long entry; SL: {stop_loss}; TP: {take_profit} (use_take_profit: {use_take_profit}); risk: {risk}"
                            })

        # Trailing stop loss update and exit logic if in position
        if in_long:

            # Check stop loss exit
            if df['low'].iloc[i] <= stop_loss:
                exit_price = min(df['open'].iloc[i], stop_loss)
                exit_price = round(exit_price / tick) * tick

                order_size.iloc[i] = -np.inf
                order_price.iloc[i] = exit_price
                in_long = False

                if log:
                    signals_log.append({
                        'timestamp': this_time,
                        'date': this_time.date().isoformat(),
                        'time': this_time.time().isoformat(),
                        'event': 'Stop Loss Exit',
                        'symbol': symbol,
                        'price': exit_price,
                        'details': f"Hit SL at {stop_loss}"
                    })

                continue  # Skip take profit if stopped out

            # Check take profit exit only if enabled
            if use_take_profit and df['high'].iloc[i] >= take_profit:
                exit_price = max(df['open'].iloc[i], take_profit)
                exit_price = round(exit_price / tick) * tick

                order_size.iloc[i] = -np.inf
                order_price.iloc[i] = exit_price
                in_long = False

                if log:
                    signals_log.append({
                        'timestamp': this_time,
                        'date': this_time.date().isoformat(),
                        'time': this_time.time().isoformat(),
                        'event': 'Take Profit Exit',
                        'symbol': symbol,
                        'price': exit_price,
                        'details': f"Hit TP at {take_profit}"
                    })
                continue
            # Update trailing stop loss if the current bar is green (close >= open)
            if df['open'].iloc[i] <= df['close'].iloc[i]:
                new_sl = df['low'].iloc[i] - 1e-8
                if new_sl > stop_loss:
                    old_sl = stop_loss
                    stop_loss = new_sl
                    if log:
                        signals_log.append({
                            'timestamp': this_time,
                            'date': this_time.date().isoformat(),
                            'time': this_time.time().isoformat(),
                            'event': 'Trailing SL Update',
                            'symbol': symbol,
                            'price': stop_loss,
                            'details': f"Updated from {old_sl} to {stop_loss} on green bar"
                        })

    if log:
        if logfile:
            log_df = pd.DataFrame(signals_log)
            log_df.to_csv(logfile, index=False)
        else:
            for entry in signals_log:
                print(entry)

    return order_size, order_price
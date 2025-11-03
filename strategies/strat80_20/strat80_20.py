# Modified strat80_20.py
import pandas as pd
import numpy as np
import talib
import os
import datetime

def generate_signals(
    df: pd.DataFrame, 
    symbol: str = '', 
    setup_days: list = None,
    take_profit_mult: float = 3.0,
    initial_sl_mult: float = 0.5,
    tick: float = 0.05,
    use_take_profit: bool = True,
    trigger_tick_mult: int = 10,
    trigger_window_minutes: int = 60,
    session_start: str = "09:15:00",
    max_attempts: int = None,
    log: bool = False
) -> tuple:
    """
    Generate buy signals based on the 80-20 strategy for intraday execution.
    
    This function expects setup_days to be provided. Setup days are detected externally
    by the scanner and passed in. The function handles the intraday execution logic:
    - Monitors for trigger (price drops below threshold in first 60 minutes)
    - Places entry order at previous day's low
    - Manages stop loss and take profit
    - Implements trailing stop loss on green bars
    
    Args:
        df: 15-minute intraday DataFrame with OHLC data
        symbol: Trading symbol (default: '')
        setup_days: List of setup day dicts with setup_date, entry_price, trigger_price, true_range (default: None)
        take_profit_mult: Take profit multiplier of risk (default: 3.0)
        initial_sl_mult: Initial stop loss multiplier of true range (default: 0.5)
        tick: Tick size for price rounding (default: 0.05)
        use_take_profit: Whether to use take profit exits (default: True)
        trigger_tick_mult: Multiplier for trigger threshold calculation (default: 10)
        trigger_window_minutes: Time window in minutes to monitor for trigger (default: 60)
        session_start: Session start time in HH:MM:SS format (default: "09:15:00")
        max_attempts: Maximum number of attempts allowed per day (default: None for unlimited)
        log: Enable logging of signal events (default: False)
    
    Returns:
        tuple: (order_size, order_price, signals_log) - pandas Series for vectorbt and list of log dicts
    """
    # Initialize logging if enabled
    if log:
        signals_log = []  # List of dicts for structured logging
    
    # Setup days must be provided by caller (from scanner)
    setup_days_input = setup_days
    if not setup_days_input:
        # If no setup days provided, return empty signals
        order_size = pd.Series(0.0, index=df.index)
        order_price = pd.Series(np.nan, index=df.index)
        if log:
            signals_log.append({
                'timestamp': df.index[0] if len(df) > 0 else pd.Timestamp.now(),
                'date': df.index[0].date() if len(df) > 0 else datetime.date.today(),
                'time': df.index[0].time() if len(df) > 0 else datetime.datetime.now().time(),
                'event': 'No Setup Days',
                'symbol': symbol,
                'price': 0,
                'details': 'No setup days provided to generate_signals'
            })
        return order_size, order_price, signals_log if log else []
    
    # Convert setup_days to a dict mapping date -> setup info
    # Expected format: list of dicts with keys: setup_date, entry_price, trigger_price, true_range, etc.
    setup_dict = {}
    for setup in setup_days_input:
        if isinstance(setup, dict):
            setup_date = pd.to_datetime(setup['setup_date']).date()
            setup_dict[setup_date] = setup
        else:
            # If just dates are provided (backward compatibility)
            setup_dict[setup] = {}
    
    # Get daily data for computing entry prices and true ranges
    daily_df = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Compute true range for days we need
    daily_df['prev_close'] = daily_df['close'].shift(1)
    daily_df['true_high'] = np.maximum(daily_df['high'], daily_df['prev_close'].fillna(daily_df['open']))
    daily_df['true_low'] = np.minimum(daily_df['low'], daily_df['prev_close'].fillna(daily_df['open']))
    daily_df['tr'] = daily_df['true_high'] - daily_df['true_low']

    # Ensure required columns in df
    if 'open' not in df.columns:
        df['open'] = df['close']

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
    entry_tr = 0.0
    threshold = 0.0
    trigger_time = None
    day_low_so_far = np.inf
    session_start_dt = None
    first60_end = None
    current_day = None
    entries_today = 0  # Track number of entries taken today

    # Loop over 15m bars
    for i in range(len(df)):
        this_time = df.index[i]
        this_day = this_time.date()

        if current_day != this_day:
            # Reset per-day states for new day
            potential_entry = False
            trigger_time = None
            day_low_so_far = np.inf
            entries_today = 0  # Reset entry counter for new day
            entry_tr = 0.0  # Reset true range for new day

            # Check if previous day had a setup
            prev_day = this_day - datetime.timedelta(days=1)
            if prev_day in setup_dict:
                setup_info = setup_dict[prev_day]
                potential_entry = True
                
                # Use provided entry price or compute from daily data
                if 'entry_price' in setup_info:
                    entry_price = setup_info['entry_price']
                    entry_tr = setup_info.get('true_range', 0)
                else:
                    # Fallback: compute from daily_df
                    prev_day_times = daily_df.index[daily_df.index.date == prev_day]
                    if not prev_day_times.empty:
                        p_idx = prev_day_times[0]
                        entry_price = daily_df.at[p_idx, 'low']
                        entry_tr = daily_df.at[p_idx, 'tr']
                    else:
                        potential_entry = False
                        continue
                
                threshold = entry_price - trigger_tick_mult * tick

                if log:
                    signals_log.append({
                        'timestamp': datetime.datetime.combine(prev_day, datetime.time(15, 30)),
                        'date': prev_day.isoformat(),
                        'time': '15:30:00',
                        'event': 'Setup Detected',
                        'symbol': symbol,
                        'price': entry_price,
                        'details': f"80-20 setup for next day entry at {entry_price}; threshold: {threshold}; TR: {entry_tr}"
                    })

                session_start_str = f"{this_day.isoformat()} {session_start}"
                session_start_dt = pd.to_datetime(session_start_str)
                first60_end = session_start_dt + pd.Timedelta(minutes=trigger_window_minutes)
                if df.index.tz is not None and session_start_dt.tzinfo is None:
                    session_start_dt = session_start_dt.tz_localize(df.index.tz)
                    first60_end = first60_end.tz_localize(df.index.tz)
                elif df.index.tz is not None and session_start_dt.tzinfo is not None:
                    session_start_dt = session_start_dt.tz_convert(df.index.tz)
                    first60_end = first60_end.tz_convert(df.index.tz)

            current_day = this_day

        # Update day_low_so_far if potential entry day
        if potential_entry:
            day_low_so_far = min(day_low_so_far, df['low'].iloc[i])

        # Entry logic
        if not in_long:
            if potential_entry:
                if session_start_dt <= this_time < first60_end:
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
                    # Check if we can take an entry (max_attempts limit)
                    can_enter = (max_attempts is None) or (entries_today < max_attempts)
                    
                    if can_enter and df['high'].iloc[i] >= entry_price:
                        fill_price = entry_price
                        fill_price = round(fill_price / tick) * tick

                        order_size.iloc[i] = np.inf
                        order_price.iloc[i] = fill_price
                        in_long = True
                        long_entry_price = fill_price
                        # New SL calculation: entry_price - (initial_sl_mult * true_range)
                        stop_loss = long_entry_price - (initial_sl_mult * entry_tr)
                        risk = long_entry_price - stop_loss
                        take_profit = long_entry_price + take_profit_mult * risk
                        entries_today += 1  # Increment entry counter

                        if log:
                            signals_log.append({
                                'timestamp': this_time,
                                'date': this_time.date().isoformat(),
                                'time': this_time.time().isoformat(),
                                'event': 'Entry Filled',
                                'symbol': symbol,
                                'price': fill_price,
                                'details': f"Long entry #{entries_today}; SL: {stop_loss} (entry - {initial_sl_mult}*TR); TP: {take_profit} (use_take_profit: {use_take_profit}); risk: {risk}; TR: {entry_tr}"
                            })
                    elif not can_enter and df['high'].iloc[i] >= entry_price:
                        # Max attempts reached, log missed opportunity
                        if log:
                            signals_log.append({
                                'timestamp': this_time,
                                'date': this_time.date().isoformat(),
                                'time': this_time.time().isoformat(),
                                'event': 'Entry Skipped - Max Attempts Reached',
                                'symbol': symbol,
                                'price': df['high'].iloc[i],
                                'details': f"Max attempts ({max_attempts}) already taken today. Entries: {entries_today}"
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
                    can_retry = (max_attempts is None) or (entries_today < max_attempts)
                    signals_log.append({
                        'timestamp': this_time,
                        'date': this_time.date().isoformat(),
                        'time': this_time.time().isoformat(),
                        'event': 'Stop Loss Exit',
                        'symbol': symbol,
                        'price': exit_price,
                        'details': f"Hit SL at {stop_loss}. Entries today: {entries_today}/{max_attempts if max_attempts else 'unlimited'}. Can retry: {can_retry}"
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
                    can_retry = (max_attempts is None) or (entries_today < max_attempts)
                    signals_log.append({
                        'timestamp': this_time,
                        'date': this_time.date().isoformat(),
                        'time': this_time.time().isoformat(),
                        'event': 'Take Profit Exit',
                        'symbol': symbol,
                        'price': exit_price,
                        'details': f"Hit TP at {take_profit}. Entries today: {entries_today}/{max_attempts if max_attempts else 'unlimited'}. Can retry: {can_retry}"
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

    # Return signals and logs
    return order_size, order_price, signals_log if log else []
import pandas as pd
import numpy as np
import talib
import json
import os

def generate_signals(df: pd.DataFrame, symbol: str = '', **kwargs) -> tuple:
    """
    Generate order size and price series for the strategy based on Keltner Channels, ADX, and MACD.
    Returns order_size, order_price as pd.Series for use in VectorBT Portfolio.from_orders.
    Positive size = buy (long entry or short exit), negative size = sell (long exit or short entry).
    Assumes fixed position size of 1.0 unit (adjust as needed for your risk rules).
    Requires df with columns: 'open' (optional, falls back to 'close'), 'high', 'low', 'close'.
    
    Parameters are loaded from 'strat001_config.json' in the same directory, and can be overridden via kwargs.
    
    Modifications:
    - Removed take-profit (initial targets) logic entirely.
    - Changed initial stop-loss to be ATR-based using stop_loss_pct as the multiplier (repurposed as stop_loss_mult).
    - Updated trailing stop-loss to use Chandelier Exit (based on highest high/low since entry minus/plus ATR multiple).
    - Trailing activates once the trade moves in the favorable direction by 1R (initial risk amount).
    """
    # Load default parameters from JSON
    config_path = os.path.join(os.path.dirname(__file__), 'strat001_config.json')
    with open(config_path, 'r') as f:
        params = json.load(f)
    
    # Override with provided kwargs
    params.update(kwargs)
    
    # Extract parameters
    kc_length = params['kc_length']
    kc_mult = params['kc_mult']
    adx_length = params['adx_length']
    adx_threshold = params['adx_threshold']
    adx_exit_threshold = params['adx_exit_threshold']
    macd_fast = params['macd_fast']
    macd_slow = params['macd_slow']
    macd_signal = params['macd_signal']
    use_stop_loss = params['use_stop_loss']
    stop_loss_mult = params['stop_loss_mult']  # Repurposed as ATR multiplier (e.g., 2.0 means 2x ATR)
    use_trailing_sl = params['use_trailing_sl']
    trailing_mult = params['trailing_mult']
    trailing_length = params['trailing_length']
    trade_direction = params['trade_direction']
    tick = params['tick']

    # Determine if logging is enabled
    log = kwargs.get('log', False)
    if log:
        signals_log = []

    # Ensure required columns
    if 'open' not in df.columns:
        df['open'] = df['close']  # Fallback if no open
    if 'volume' not in df.columns:
        df['volume'] = np.nan  # Fallback if no volume

    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']

    # Compute Keltner Channels using TA-Lib
    kc_middle = pd.Series(talib.EMA(close.values, timeperiod=kc_length), index=close.index)
    kc_atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=kc_length), index=close.index)
    kc_upper = kc_middle + (kc_mult * kc_atr)
    kc_lower = kc_middle - (kc_mult * kc_atr)

    # Compute DMI/ADX using TA-Lib
    di_plus = pd.Series(talib.PLUS_DI(high.values, low.values, close.values, timeperiod=adx_length), index=close.index)
    di_minus = pd.Series(talib.MINUS_DI(high.values, low.values, close.values, timeperiod=adx_length), index=close.index)
    adx = pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=adx_length), index=close.index)

    # Compute MACD using TA-Lib
    macd_line, signal_line, macd_histogram = talib.MACD(close.values, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    macd_line = pd.Series(macd_line, index=close.index)
    signal_line = pd.Series(signal_line, index=close.index)
    macd_histogram = pd.Series(macd_histogram, index=close.index)

    # Trailing ATR using TA-Lib
    trailing_atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=trailing_length), index=close.index)

    # Vectorized conditions
    is_range_bound = adx < adx_threshold
    trend_starting = adx > adx_exit_threshold
    price_at_lower_kc = close <= kc_lower
    price_at_upper_kc = close >= kc_upper
    macd_hist_turning_positive = (macd_histogram > 0) & (macd_histogram.shift(1) <= 0)
    macd_cross_below = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    long_condition = price_at_lower_kc & is_range_bound & macd_hist_turning_positive
    short_condition = price_at_upper_kc & is_range_bound & macd_cross_below

    # Initialize order series
    n = len(df)
    order_size = pd.Series(0.0, index=df.index)  # Size: +1.0 buy, -1.0 sell
    order_price = pd.Series(np.nan, index=df.index)  # Custom fill price

    # State variables
    in_long = False
    in_short = False
    pending_long = False
    pending_short = False
    long_entry_bar = 0
    short_entry_bar = 0
    long_stop_price = 0.0
    short_stop_price = 0.0
    long_sl_price = 0.0
    short_sl_price = 0.0
    long_trail_stop = 0.0
    short_trail_stop = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    atr_at_entry = 0.0
    initial_risk = 0.0
    activation_threshold = 0.0
    trailing_active = False

    # Loop through bars (stateful simulation)
    for i in range(1, n):
        current_open = open_.iloc[i]
        current_high = high.iloc[i]
        current_low = low.iloc[i]
        current_close = close.iloc[i]

        # Check for order cancellation
        if pending_long and (i - long_entry_bar >= 3):
            pending_long = False
        if pending_short and (i - short_entry_bar >= 3):
            pending_short = False

        # Check for pending order fills
        if pending_long:
            if current_high >= long_stop_price:
                fill_price = max(current_open, long_stop_price)  # Approximate fill at stop or better
                order_size.iloc[i] = 1.0  # Buy to enter long
                order_price.iloc[i] = fill_price
                in_long = True
                pending_long = False
                atr_at_entry = trailing_atr.iloc[i]
                initial_risk = stop_loss_mult * atr_at_entry
                activation_threshold = initial_risk * 1.0  # Activate trailing after 1R move (hardcoded; adjust if needed)
                highest_high_since_entry = max(fill_price, current_high)
                trailing_active = False
                # Set initial ATR-based stop-loss
                if use_stop_loss:
                    long_sl_price = fill_price - initial_risk
                if log:
                    signals_log.append({
                        'timestamp': df.index[i],
                        'symbol': symbol,
                        'price': order_price.iloc[i],
                        'volume': df['volume'].iloc[i],
                        'kc_upper': kc_upper.iloc[i],
                        'kc_lower': kc_lower.iloc[i],
                        'kc_middle': kc_middle.iloc[i],
                        'adx': adx.iloc[i],
                        'di_plus': di_plus.iloc[i],
                        'di_minus': di_minus.iloc[i],
                        'macd_line': macd_line.iloc[i],
                        'signal_line': signal_line.iloc[i],
                        'macd_histogram': macd_histogram.iloc[i],
                        'chandelier_stop': None,
                        'type': 'entry_long'
                    })

        if pending_short:
            if current_low <= short_stop_price:
                fill_price = min(current_open, short_stop_price)
                order_size.iloc[i] = -1.0  # Sell to enter short
                order_price.iloc[i] = fill_price
                in_short = True
                pending_short = False
                atr_at_entry = trailing_atr.iloc[i]
                initial_risk = stop_loss_mult * atr_at_entry
                activation_threshold = initial_risk * 1.0
                lowest_low_since_entry = min(fill_price, current_low)
                trailing_active = False
                if use_stop_loss:
                    short_sl_price = fill_price + initial_risk
                if log:
                    signals_log.append({
                        'timestamp': df.index[i],
                        'symbol': symbol,
                        'price': order_price.iloc[i],
                        'volume': df['volume'].iloc[i],
                        'kc_upper': kc_upper.iloc[i],
                        'kc_lower': kc_lower.iloc[i],
                        'kc_middle': kc_middle.iloc[i],
                        'adx': adx.iloc[i],
                        'di_plus': di_plus.iloc[i],
                        'di_minus': di_minus.iloc[i],
                        'macd_line': macd_line.iloc[i],
                        'signal_line': signal_line.iloc[i],
                        'macd_histogram': macd_histogram.iloc[i],
                        'chandelier_stop': None,
                        'type': 'entry_short'
                    })

        # Check exits if in position
        if in_long:
            exited = False

            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, current_high)

            # Trailing stop logic (Chandelier)
            if use_trailing_sl:
                if not trailing_active:
                    if highest_high_since_entry >= fill_price + activation_threshold:
                        trailing_active = True
                        long_trail_stop = highest_high_since_entry - trailing_mult * trailing_atr.iloc[i]
                if trailing_active:
                    long_trail_stop = highest_high_since_entry - trailing_mult * trailing_atr.iloc[i]
                    if current_low <= long_trail_stop:
                        order_size.iloc[i] = -1.0  # Sell to exit long
                        order_price.iloc[i] = long_trail_stop
                        in_long = False
                        exited = True
                        if log:
                            signals_log.append({
                                'timestamp': df.index[i],
                                'symbol': symbol,
                                'price': order_price.iloc[i],
                                'volume': df['volume'].iloc[i],
                                'kc_upper': kc_upper.iloc[i],
                                'kc_lower': kc_lower.iloc[i],
                                'kc_middle': kc_middle.iloc[i],
                                'adx': adx.iloc[i],
                                'di_plus': di_plus.iloc[i],
                                'di_minus': di_minus.iloc[i],
                                'macd_line': macd_line.iloc[i],
                                'signal_line': signal_line.iloc[i],
                                'macd_histogram': macd_histogram.iloc[i],
                                'chandelier_stop': long_trail_stop,
                                'type': 'exit_long_trailing'
                            })

            if not exited:
                # Check initial SL (if not trailing or before activation)
                if use_stop_loss and current_low <= long_sl_price:
                    order_size.iloc[i] = -1.0
                    order_price.iloc[i] = long_sl_price
                    in_long = False
                    exited = True
                    chandelier_stop = long_trail_stop if trailing_active else None
                    if log:
                        signals_log.append({
                            'timestamp': df.index[i],
                            'symbol': symbol,
                            'price': order_price.iloc[i],
                            'volume': df['volume'].iloc[i],
                            'kc_upper': kc_upper.iloc[i],
                            'kc_lower': kc_lower.iloc[i],
                            'kc_middle': kc_middle.iloc[i],
                            'adx': adx.iloc[i],
                            'di_plus': di_plus.iloc[i],
                            'di_minus': di_minus.iloc[i],
                            'macd_line': macd_line.iloc[i],
                            'signal_line': signal_line.iloc[i],
                            'macd_histogram': macd_histogram.iloc[i],
                            'chandelier_stop': chandelier_stop,
                            'type': 'exit_long_sl'
                        })

            if not exited:
                # Primary exit
                if current_close >= kc_middle.iloc[i] or trend_starting.iloc[i]:
                    order_size.iloc[i] = -1.0
                    order_price.iloc[i] = current_close
                    in_long = False
                    chandelier_stop = long_trail_stop if trailing_active else None
                    if log:
                        signals_log.append({
                            'timestamp': df.index[i],
                            'symbol': symbol,
                            'price': order_price.iloc[i],
                            'volume': df['volume'].iloc[i],
                            'kc_upper': kc_upper.iloc[i],
                            'kc_lower': kc_lower.iloc[i],
                            'kc_middle': kc_middle.iloc[i],
                            'adx': adx.iloc[i],
                            'di_plus': di_plus.iloc[i],
                            'di_minus': di_minus.iloc[i],
                            'macd_line': macd_line.iloc[i],
                            'signal_line': signal_line.iloc[i],
                            'macd_histogram': macd_histogram.iloc[i],
                            'chandelier_stop': chandelier_stop,
                            'type': 'exit_long_primary'
                        })

        if in_short:
            exited = False

            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, current_low)

            # Trailing stop logic (Chandelier)
            if use_trailing_sl:
                if not trailing_active:
                    if lowest_low_since_entry <= fill_price - activation_threshold:
                        trailing_active = True
                        short_trail_stop = lowest_low_since_entry + trailing_mult * trailing_atr.iloc[i]
                if trailing_active:
                    short_trail_stop = lowest_low_since_entry + trailing_mult * trailing_atr.iloc[i]
                    if current_high >= short_trail_stop:
                        order_size.iloc[i] = 1.0  # Buy to exit short
                        order_price.iloc[i] = short_trail_stop
                        in_short = False
                        exited = True
                        if log:
                            signals_log.append({
                                'timestamp': df.index[i],
                                'symbol': symbol,
                                'price': order_price.iloc[i],
                                'volume': df['volume'].iloc[i],
                                'kc_upper': kc_upper.iloc[i],
                                'kc_lower': kc_lower.iloc[i],
                                'kc_middle': kc_middle.iloc[i],
                                'adx': adx.iloc[i],
                                'di_plus': di_plus.iloc[i],
                                'di_minus': di_minus.iloc[i],
                                'macd_line': macd_line.iloc[i],
                                'signal_line': signal_line.iloc[i],
                                'macd_histogram': macd_histogram.iloc[i],
                                'chandelier_stop': short_trail_stop,
                                'type': 'exit_short_trailing'
                            })

            if not exited:
                # Check initial SL (if not trailing or before activation)
                if use_stop_loss and current_high >= short_sl_price:
                    order_size.iloc[i] = 1.0
                    order_price.iloc[i] = short_sl_price
                    in_short = False
                    exited = True
                    chandelier_stop = short_trail_stop if trailing_active else None
                    if log:
                        signals_log.append({
                            'timestamp': df.index[i],
                            'symbol': symbol,
                            'price': order_price.iloc[i],
                            'volume': df['volume'].iloc[i],
                            'kc_upper': kc_upper.iloc[i],
                            'kc_lower': kc_lower.iloc[i],
                            'kc_middle': kc_middle.iloc[i],
                            'adx': adx.iloc[i],
                            'di_plus': di_plus.iloc[i],
                            'di_minus': di_minus.iloc[i],
                            'macd_line': macd_line.iloc[i],
                            'signal_line': signal_line.iloc[i],
                            'macd_histogram': macd_histogram.iloc[i],
                            'chandelier_stop': chandelier_stop,
                            'type': 'exit_short_sl'
                        })

            if not exited:
                # Primary exit
                if current_close <= kc_middle.iloc[i] or trend_starting.iloc[i]:
                    order_size.iloc[i] = 1.0
                    order_price.iloc[i] = current_close
                    in_short = False
                    chandelier_stop = short_trail_stop if trailing_active else None
                    if log:
                        signals_log.append({
                            'timestamp': df.index[i],
                            'symbol': symbol,
                            'price': order_price.iloc[i],
                            'volume': df['volume'].iloc[i],
                            'kc_upper': kc_upper.iloc[i],
                            'kc_lower': kc_lower.iloc[i],
                            'kc_middle': kc_middle.iloc[i],
                            'adx': adx.iloc[i],
                            'di_plus': di_plus.iloc[i],
                            'di_minus': di_minus.iloc[i],
                            'macd_line': macd_line.iloc[i],
                            'signal_line': signal_line.iloc[i],
                            'macd_histogram': macd_histogram.iloc[i],
                            'chandelier_stop': chandelier_stop,
                            'type': 'exit_short_primary'
                        })

        # Check for new pending orders if no position and no pending
        if not in_long and not in_short and not pending_long and not pending_short:
            if (trade_direction == "Long Only" or trade_direction == "Long & Short") and long_condition.iloc[i]:
                pending_long = True
                long_entry_bar = i
                long_stop_price = high.iloc[i - 1] + tick

            if (trade_direction == "Short Only" or trade_direction == "Long & Short") and short_condition.iloc[i]:
                pending_short = True
                short_entry_bar = i
                short_stop_price = low.iloc[i - 1] - tick

    # Save logs if enabled and data exists
    if log and signals_log and symbol:
        log_df = pd.DataFrame(signals_log)
        file_path = f'logs/signals_{symbol}.csv'
        log_df.to_csv(file_path, index=False)
        print(f"Signal logs saved to {file_path}")

    return order_size, order_price
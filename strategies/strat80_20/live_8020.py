"""
Live trading implementation for strat80_20.
This module provides both the LiveStrategy class wrapper and the original function-based interface.
"""

import os
import pandas as pd
import datetime
import threading
import time
import pytz
import csv
import logging
from typing import Optional
from openalgo import api
from data_manager.config import OPENALGO_URL, API_KEY, EXCHANGE
from strategies.base_live import LiveStrategy
from .live_config import STRATEGY_NAME, WS_URL
from . import live_symbol_state
from .live_symbol_state import SymbolState

# Global runtime variables (set by start_trading)
logger = None
signals_logfile = None
client = None
live_run_id = None
timezone = None
symbol_states = {}


def is_market_open(ts=None, market_open_hour=9, market_open_minute=15, 
                   market_close_hour=15, market_close_minute=30):
    """Check if market is currently open."""
    if ts is None:
        ts = datetime.datetime.now(timezone)
    elif ts.tzinfo is None:
        ts = timezone.localize(ts)
    
    market_open = ts.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
    market_close = ts.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)
    
    return market_open <= ts < market_close


def init_logging():
    """Initialize logging."""
    global logger, signals_logfile
    
    os.makedirs('logs', exist_ok=True)
    today = datetime.date.today().isoformat()
    error_log_file = f'logs/live_strat80_20_errors_{today}.log'
    logging.basicConfig(
        filename=error_log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    signals_logfile = f'logs/live_strat80_20_signals_{today}.csv'
    if not os.path.exists(signals_logfile):
        with open(signals_logfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'date', 'time', 'event', 'symbol', 'price', 'details'])
            writer.writeheader()


def log_event(event_dict, logfile=None):
    """Append event to CSV log file."""
    if logfile is None:
        logfile = signals_logfile
    # Normalize date/time to ensure hours are preserved in CSV viewers
    try:
        ts_val = event_dict.get('timestamp')
        if ts_val is not None:
            if isinstance(ts_val, str):
                try:
                    ts_dt = pd.to_datetime(ts_val)
                except Exception:
                    ts_dt = None
            else:
                ts_dt = ts_val
            if ts_dt is not None:
                # Ensure timezone-aware conversion to local timezone if possible
                if hasattr(ts_dt, 'tzinfo') and ts_dt.tzinfo is not None:
                    ts_local = ts_dt.astimezone(timezone)
                else:
                    ts_local = ts_dt
                event_dict['date'] = getattr(ts_local, 'date')().isoformat()
                # Force HH:MM:SS format so Excel shows hours
                event_dict['time'] = getattr(ts_local, 'strftime')('%H:%M:%S')
    except Exception:
        # Best-effort normalization; proceed with original values on failure
        pass
    with open(logfile, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'date', 'time', 'event', 'symbol', 'price', 'details'])
        writer.writerow(event_dict)


def on_data_received(data):
    symbol = data['symbol']
    ltp = data['data']['ltp']
    ts = pd.to_datetime(data['data']['timestamp'], unit='ms', utc=True).astimezone(timezone)
    if symbol in symbol_states:
        symbol_states[symbol].process_tick(ts, ltp)


def on_order_update_received(update):
    symbol = update['symbol']
    if symbol in symbol_states:
        symbol_states[symbol].on_order_update(update)


def _normalize_symbol(sym: str) -> str:
    """Normalize broker symbol to compare with our configured symbols.
    - Uppercase
    - Strip whitespace
    - Remove common suffixes like '-EQ' or ' EQ'
    """
    if not isinstance(sym, str):
        return ''
    s = sym.strip().upper()
    # Remove common equity suffixes
    for suf in ('-EQ', ' EQ', '-BE', ' BE'):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s

def recover_state_from_broker():
    """Recover positions and orders from broker after restart/crash."""
    logger.info("[Recovery] Attempting to recover state from broker...")
    try:
        log_event({
            'timestamp': datetime.datetime.now(timezone).isoformat(),
            'date': datetime.date.today().isoformat(),
            'time': datetime.datetime.now(timezone).strftime('%H:%M:%S'),
            'event': 'Recovery Start',
            'symbol': '',
            'price': 0,
            'details': 'Attempting to recover positions and orders from broker'
        })
    except Exception:
        pass
    
    # Build allowed symbol set from run_config setups
    allowed_symbols = { _normalize_symbol(k) for k in symbol_states.keys() }
    
    # Recover positions
    try:
        position_response = client.positionbook()
        if position_response.get('status') == 'success':
            positions = position_response.get('data', [])
            for pos in positions:
                raw_symbol = pos.get('symbol') or pos.get('trading_symbol') or pos.get('tradingsymbol')
                symbol_norm = _normalize_symbol(raw_symbol)
                if symbol_norm not in allowed_symbols:
                    continue  # Ignore positions not in our run_config
                # Map back to our exact key casing
                symbol = next(k for k in symbol_states.keys() if _normalize_symbol(k) == symbol_norm)
                qty = int(pos.get('netqty', 0))
                if symbol in symbol_states and qty > 0:
                    avg_price = float(pos.get('avgprice', 0) or pos.get('average_price', 0))
                    symbol_states[symbol].in_position = True
                    symbol_states[symbol].long_entry_price = avg_price
                    # Reconstruct stop loss and take profit based on entry price
                    state = symbol_states[symbol]
                    state.stop_loss = avg_price - (state.initial_sl_ticks * state.tick_size)
                    state.stop_loss = round(state.stop_loss / state.tick_size) * state.tick_size
                    risk = avg_price - state.stop_loss
                    if state.use_take_profit:
                        state.take_profit = avg_price + state.take_profit_mult * risk
                        state.take_profit = round(state.take_profit / state.tick_size) * state.tick_size
                    state.entries_today = 1  # Assume at least 1 entry
                    logger.info(f"[Recovery] Restored position for {symbol}: qty={qty}, entry={avg_price}, SL={state.stop_loss}")
                    log_event({
                        'timestamp': datetime.datetime.now(timezone).isoformat(),
                        'date': datetime.date.today().isoformat(),
                        'time': datetime.datetime.now(timezone).strftime('%H:%M:%S'),
                        'event': 'Position Recovered',
                        'symbol': symbol,
                        'price': avg_price,
                        'details': f"Recovered position: qty={qty}, SL={state.stop_loss}, TP={state.take_profit}"
                    })
    except Exception as e:
        logger.error(f"[Recovery] Error recovering positions: {e}")
    
    # Recover pending orders
    try:
        order_response = client.orderbook()
        if order_response.get('status') == 'success':
            orders = order_response.get('data', [])
            for order in orders:
                raw_symbol = order.get('symbol') or order.get('trading_symbol') or order.get('tradingsymbol')
                symbol_norm = _normalize_symbol(raw_symbol)
                if symbol_norm not in allowed_symbols:
                    continue  # Ignore orders not in our run_config
                # Map back to our exact key casing
                symbol = next(k for k in symbol_states.keys() if _normalize_symbol(k) == symbol_norm)
                order_status = order.get('status') or order.get('order_status') or order.get('orderStatus')
                order_status_norm = str(order_status).strip().lower() if order_status is not None else ''
                # Accept common variants from brokers
                is_open_like = order_status_norm in (
                    'open', 'pending', 'trigger_pending', 'trigger pending', 'validation pending', 'put order req received'
                )
                if symbol in symbol_states and is_open_like:
                    order_id = order.get('order_id') or order.get('orderid') or order.get('orderId')
                    # Determine side/action and price type broadly across possible keys
                    action = (order.get('action') or order.get('transaction_type') or order.get('side') or '').upper()
                    price_type = (order.get('price_type') or order.get('order_type') or order.get('pricetype') or '').upper()
                    # If it's a BUY stop (entry), attach as entry order
                    if action == 'BUY':
                        symbol_states[symbol].entry_order_id = order_id
                        symbol_states[symbol].triggered = True  # Must have been triggered to place order
                        logger.info(f"[Recovery] Restored pending entry order for {symbol}: order_id={order_id}")
                        log_event({
                            'timestamp': datetime.datetime.now(timezone).isoformat(),
                            'date': datetime.date.today().isoformat(),
                            'time': datetime.datetime.now(timezone).strftime('%H:%M:%S'),
                            'event': 'Entry Order Recovered',
                            'symbol': symbol,
                            'price': 0,
                            'details': f"Recovered pending entry order: {order_id}"
                        })
                    # If it's a SELL Stop/Stop-Limit, attach as protective SL
                    elif action == 'SELL' and price_type in ('SL', 'SL-M', 'STOPLIMIT', 'STOP-LOSS'):
                        state = symbol_states[symbol]
                        state.sl_order_id = order_id
                        # Sync stop_loss from trigger price on order if available
                        trig = (
                            order.get('trigger_price') or order.get('triggerprice') or order.get('triggerPrice') or 0
                        )
                        try:
                            trig = float(trig)
                        except Exception:
                            trig = 0.0
                        if trig:
                            # Round to tick and set
                            trig = round(trig / state.tick_size) * state.tick_size
                            state.stop_loss = trig
                        logger.info(f"[Recovery] Restored protective SL for {symbol}: order_id={order_id}, trigger={trig}")
                        log_event({
                            'timestamp': datetime.datetime.now(timezone).isoformat(),
                            'date': datetime.date.today().isoformat(),
                            'time': datetime.datetime.now(timezone).strftime('%H:%M:%S'),
                            'event': 'Sell Stop Recovered',
                            'symbol': symbol,
                            'price': trig,
                            'details': f"Recovered broker-held Sell Stop: {order_id}"
                        })
    except Exception as e:
        logger.error(f"[Recovery] Error recovering orders: {e}")
    
    logger.info("[Recovery] State recovery complete")
    try:
        log_event({
            'timestamp': datetime.datetime.now(timezone).isoformat(),
            'date': datetime.date.today().isoformat(),
            'time': datetime.datetime.now(timezone).strftime('%H:%M:%S'),
            'event': 'Recovery Complete',
            'symbol': '',
            'price': 0,
            'details': 'Recovered positions and orders (if any)'
        })
    except Exception:
        pass


def load_setups(setups_df, market_open_hour, market_open_minute, market_close_hour, market_close_minute,
                fixed_qty, product_type, take_profit_mult, initial_sl_ticks, 
                use_take_profit, trigger_window_minutes, max_attempts, max_order_retries,
                order_timeout_minutes=None, exit_before_close_minutes=None,
                ignore_initial_minutes: int = 1):
    """Load setups from DataFrame and create SymbolState instances."""
    instruments = []
    for _, setup in setups_df.iterrows():
        symbol = setup['symbol']
        symbol_states[symbol] = SymbolState(
            symbol=symbol,
            entry_price=setup['entry_price'],
            trigger_price=setup['trigger_price'],
            tick_size=setup['tick_size'],
            true_range=setup['true_range'],
            market_open_hour=market_open_hour,
            market_open_minute=market_open_minute,
            market_close_hour=market_close_hour,
            market_close_minute=market_close_minute,
            fixed_qty=fixed_qty,
            product_type=product_type,
            take_profit_mult=take_profit_mult,
            initial_sl_ticks=initial_sl_ticks,
            use_take_profit=use_take_profit,
            trigger_window_minutes=trigger_window_minutes,
            max_attempts=max_attempts,
            max_order_retries=max_order_retries,
            order_timeout_minutes=order_timeout_minutes,
            exit_before_close_minutes=exit_before_close_minutes,
            ignore_initial_minutes=ignore_initial_minutes
        )
        instruments.append({"exchange": EXCHANGE, "symbol": symbol})
        now = datetime.datetime.now(timezone)
        log_event({
            'timestamp': now.isoformat(),
            'date': now.date().isoformat(),
            'time': now.strftime('%H:%M:%S'),
            'event': 'Setup Loaded',
            'symbol': symbol,
            'price': setup['entry_price'],
            'details': f"Trigger price: {setup['trigger_price']}; TR: {setup['true_range']}"
        })
    return instruments


def poll_orders_and_positions():
    while True:
        current_time = datetime.datetime.now(timezone)
        
        try:
            position_response = client.positionbook()
            if position_response.get('status') == 'success':
                positions = position_response.get('data', [])
            else:
                positions = []
                logger.warning(f"Failed to fetch positionbook: {position_response}")
        except Exception as e:
            logger.error(f"Error fetching positionbook: {e}")
            positions = []

        pos_dict = {pos['symbol']: int(pos.get('netqty', 0)) for pos in positions}

        for symbol, state in symbol_states.items():
            qty = pos_dict.get(symbol, 0)
            if qty > 0:
                state.in_position = True
            elif qty < 0:
                logger.warning(f"Negative position for {symbol}: {qty}. Squaring off.")
                state.place_buy_market()
                state.in_position = False
            else:
                state.in_position = False

            if (state.pending_retry and not state.in_position and 
                state.entry_order_id is None and is_market_open(current_time, state.market_open_hour, 
                                                                 state.market_open_minute, state.market_close_hour, 
                                                                 state.market_close_minute)):
                can_retry = (state.max_attempts is None) or (state.entries_today < state.max_attempts)
                if can_retry and state.triggered:
                    logger.info(f"[{symbol}] Retrying order placement after market open")
                    state.place_buy_stop(current_time)

            if state.entry_order_id and not state.in_position:
                try:
                    order_info = client.orderstatus(order_id=state.entry_order_id, strategy=STRATEGY_NAME)
                    if order_info.get('status') == 'success':
                        data = order_info.get('data', {}) or order_info
                        order_status = data.get('order_status') or data.get('status') or data.get('orderStatus')
                        avg_price = data.get('average_price') or data.get('avg_price') or data.get('averagePrice')
                        if order_status in ('complete', 'filled'):
                            fill_price = avg_price or state.entry_price
                            update = {
                                'order_id': state.entry_order_id,
                                'status': 'filled',
                                'fill_price': fill_price,
                                'symbol': symbol
                            }
                            state.on_order_update(update)
                            state.entry_order_id = None
                except Exception as e:
                    logger.error(f"[{symbol}] Error polling order status: {e}")

        time.sleep(5)


class Live8020(LiveStrategy):
    """Live trading strategy for strat80_20."""
    
    def __init__(self):
        super().__init__(strategy_name='strat80_20')
    
    def start_trading(
        self,
        setups_df: pd.DataFrame,
        timezone_str: str = 'Asia/Kolkata',
        market_open_hour: int = 9,
        market_open_minute: int = 15,
        market_close_hour: int = 15,
        market_close_minute: int = 30,
        fixed_qty: int = 1,
        product_type: str = 'MIS',
        order_validity: str = 'DAY',
        max_attempts: Optional[int] = None,
        max_order_retries: Optional[int] = 3,
        **kwargs
    ) -> None:
        """Start live trading with given setups and parameters."""
        global client, timezone, live_run_id
        
        # Extract strategy-specific parameters from kwargs
        take_profit_mult_param = kwargs.get('take_profit_mult', 3.0)
        initial_sl_ticks_param = kwargs.get('initial_sl_ticks', 20)
        use_take_profit_param = kwargs.get('use_take_profit', False)
        trigger_window_minutes_param = kwargs.get('trigger_window_minutes', 60)
        ws_url = kwargs.get('ws_url', WS_URL)
        # New optional live controls threaded from run_config
        order_timeout_minutes_param = kwargs.get('order_timeout_minutes')
        exit_before_close_minutes_param = kwargs.get('exit_before_close_minutes')
        ignore_initial_minutes_param = kwargs.get('ignore_initial_minutes', 1)
        
        # Initialize timezone
        timezone = pytz.timezone(timezone_str)
        
        # Initialize logging
        init_logging()
        
        # Initialize OpenAlgo client
        client = api(
            api_key=API_KEY,
            host=OPENALGO_URL,
            ws_url=ws_url
        )
        live_symbol_state.logger = logger
        live_symbol_state.client = client
        live_symbol_state.timezone = timezone
        live_symbol_state.log_event = log_event
        live_symbol_state.is_market_open = is_market_open
        
        # Create live run entry in database
        try:
            from .strategy_db_models import save_live_run
            live_run_id = save_live_run(
                symbols=setups_df['symbol'].tolist(),
                timezone=timezone_str,
                fixed_qty=fixed_qty,
                product_type=product_type,
                take_profit_mult=take_profit_mult_param,
                use_take_profit=use_take_profit_param,
                trigger_window_minutes=trigger_window_minutes_param,
                strategy_name=STRATEGY_NAME
            )
            print(f"[Database] Live run saved with ID: {live_run_id}")
        except Exception as db_error:
            print(f"[Database] Warning: Failed to create live run: {db_error}")
            live_run_id = None
        
        # Load setups from DataFrame
        instruments = load_setups(
            setups_df=setups_df,
            market_open_hour=market_open_hour,
            market_open_minute=market_open_minute,
            market_close_hour=market_close_hour,
            market_close_minute=market_close_minute,
            fixed_qty=fixed_qty,
            product_type=product_type,
            take_profit_mult=take_profit_mult_param,
            initial_sl_ticks=initial_sl_ticks_param,
            use_take_profit=use_take_profit_param,
            trigger_window_minutes=trigger_window_minutes_param,
            max_attempts=max_attempts,
            max_order_retries=max_order_retries,
            order_timeout_minutes=order_timeout_minutes_param,
            exit_before_close_minutes=exit_before_close_minutes_param,
            ignore_initial_minutes=ignore_initial_minutes_param
        )
        
        if not instruments:
            print("[Live] No instruments to trade. Exiting.")
            return
        
        client.connect()
        client.subscribe_ltp(instruments, on_data_received=on_data_received)
        
        # Recover state from broker (positions and orders)
        recover_state_from_broker()
        
        # Start polling thread for order statuses and positions
        polling_thread = threading.Thread(target=poll_orders_and_positions, daemon=True)
        polling_thread.start()
        
        # Wait for session start if early
        current_dt = datetime.datetime.now(timezone)
        session_start_dt = current_dt.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
        if current_dt < session_start_dt:
            wait_sec = (session_start_dt - current_dt).total_seconds()
            log_event({
                'timestamp': current_dt.isoformat(),
                'date': current_dt.date().isoformat(),
                'time': current_dt.time().isoformat(),
                'event': 'Waiting for Session Start',
                'symbol': '',
                'price': 0,
                'details': f"Waiting {wait_sec / 60:.1f} minutes for session start..."
            })
            time.sleep(wait_sec)
        
        # Keep running until session end
        try:
            while True:
                current_dt = datetime.datetime.now(timezone)
                session_end_dt = current_dt.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)
                if current_dt >= session_end_dt:
                    log_event({
                        'timestamp': current_dt.isoformat(),
                        'date': current_dt.date().isoformat(),
                        'time': current_dt.time().isoformat(),
                        'event': 'Session Ended',
                        'symbol': '',
                        'price': 0,
                        'details': "Disconnecting."
                    })
                    break
                time.sleep(1)
        finally:
            client.unsubscribe_ltp(instruments)
            client.disconnect()
            print(f"[Live] Trading session completed. Check logs for details.")
    
    def get_setup_days(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str = 'D',
        **kwargs
    ) -> pd.DataFrame:
        """
        Scan for setup days for a given symbol.
        
        Delegates to the strategy's Scanner implementation via the loader.
        """
        from strategies.strategy_loader import load_scanner_strategy
        volume_threshold = kwargs.get('volume_threshold', 100000)
        return_details = kwargs.get('return_details', True)
        scanner = load_scanner_strategy(self.strategy_name)
        return scanner.get_setup_days(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            volume_threshold=volume_threshold,
            return_details=return_details
        )

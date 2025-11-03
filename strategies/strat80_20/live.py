import os
import pandas as pd
import datetime
import threading
import time
import pytz
import csv
import logging
from openalgo import api
from data_manager.config import OPENALGO_URL, API_KEY, EXCHANGE
from .live_config import STRATEGY_NAME, WS_URL
from . import live_symbol_state as live_symbol_state
from .live_symbol_state import SymbolState

# Global runtime variables (set by start_trading)
logger = None
signals_logfile = None
client = None
live_run_id = None
timezone = None

def is_market_open(ts=None, market_open_hour=9, market_open_minute=15, 
                   market_close_hour=15, market_close_minute=30):
    """Check if market is currently open.
    
    Args:
        ts: datetime object (timezone aware). If None, uses current time in configured timezone.
        market_open_hour: Market open hour
        market_open_minute: Market open minute
        market_close_hour: Market close hour
        market_close_minute: Market close minute
    
    Returns:
        True if market is open, False otherwise.
    """
    if ts is None:
        ts = datetime.datetime.now(timezone)
    elif ts.tzinfo is None:
        ts = timezone.localize(ts)
    
    # Market hours are configurable
    market_open = ts.replace(hour=market_open_hour, minute=market_open_minute, second=0, microsecond=0)
    market_close = ts.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)
    
    return market_open <= ts < market_close

def init_logging():
    """Initialize logging."""
    global logger, signals_logfile
    
    # Configure logging for errors/warnings
    os.makedirs('logs', exist_ok=True)
    today = datetime.date.today().isoformat()
    error_log_file = f'logs/live_strat80_20_errors_{today}.log'
    logging.basicConfig(
        filename=error_log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # CSV log file for events
    signals_logfile = f'logs/live_strat80_20_signals_{today}.csv'
    if not os.path.exists(signals_logfile):
        with open(signals_logfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'date', 'time', 'event', 'symbol', 'price', 'details'])
            writer.writeheader()

def log_event(event_dict, logfile=None):
    """Append event to CSV log file."""
    if logfile is None:
        logfile = signals_logfile
    with open(logfile, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'date', 'time', 'event', 'symbol', 'price', 'details'])
        writer.writerow(event_dict)

# Global dict of symbol states
symbol_states = {}

def on_data_received(data):
    symbol = data['symbol']
    ltp = data['data']['ltp']
    ts = pd.to_datetime(data['data']['timestamp'], unit='ms', utc=True).astimezone(timezone)
    if symbol in symbol_states:
        symbol_states[symbol].process_tick(ts, ltp)

def on_order_update_received(update):
    # Assume client has on_order_update callback
    symbol = update['symbol']  # Assume
    if symbol in symbol_states:
        symbol_states[symbol].on_order_update(update)

def load_setups(setups_df, market_open_hour, market_open_minute, market_close_hour, market_close_minute,
                fixed_qty, product_type, take_profit_mult, initial_sl_mult, 
                use_take_profit, trigger_window_minutes, max_attempts, max_order_retries):
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
            initial_sl_mult=initial_sl_mult,
            use_take_profit=use_take_profit,
            trigger_window_minutes=trigger_window_minutes,
            max_attempts=max_attempts,
            max_order_retries=max_order_retries
        )
        instruments.append({"exchange": EXCHANGE, "symbol": symbol})
        now = datetime.datetime.now(timezone)
        log_event({
            'timestamp': now.isoformat(),
            'date': now.date().isoformat(),
            'time': now.time().isoformat(),
            'event': 'Setup Loaded',
            'symbol': symbol,
            'price': setup['entry_price'],
            'details': f"Trigger price: {setup['trigger_price']}; TR: {setup['true_range']}"
        })
    return instruments

def poll_orders_and_positions():
    while True:
        current_time = datetime.datetime.now(timezone)
        
        # Poll positions to sync in_position and handle unintended shorts
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

        # Build position dict - check product type per state since it may vary
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

            # Check for pending retries when market is open
            # This handles cases where orders were rejected before market open
            if (state.pending_retry and not state.in_position and 
                state.entry_order_id is None and is_market_open(current_time, state.market_open_hour, 
                                                                 state.market_open_minute, state.market_close_hour, 
                                                                 state.market_close_minute)):
                can_retry = (state.max_attempts is None) or (state.entries_today < state.max_attempts)
                if can_retry and state.triggered:
                    logger.info(f"[{symbol}] Retrying order placement after market open")
                    state.place_buy_stop(current_time)

            # Poll entry order status if pending
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
                            state.entry_order_id = None  # Clear after handling
                except Exception as e:
                    logger.error(f"[{symbol}] Error polling order status: {e}")

        time.sleep(5)  # Poll every 5 seconds

def start_trading(setups_df, timezone_str='Asia/Kolkata', market_open_hour=9, 
                 market_open_minute=15, market_close_hour=15, market_close_minute=30,
                 fixed_qty=1, product_type='MIS', order_validity='DAY', 
                 max_attempts=None, max_order_retries=3,
                 take_profit_mult_param=3.0, initial_sl_mult_param=0.5, 
                 use_take_profit_param=False, trigger_window_minutes_param=60,
                 ws_url=None):
    """Start live trading with given setups and parameters.
    
    Args:
        setups_df: DataFrame with columns [symbol, entry_price, trigger_price, tick_size, true_range, ...]
        timezone_str: Timezone string (default: 'Asia/Kolkata')
        market_open_hour: Market open hour (default: 9)
        market_open_minute: Market open minute (default: 15)
        market_close_hour: Market close hour (default: 15)
        market_close_minute: Market close minute (default: 30)
        fixed_qty: Fixed quantity per trade (default: 1)
        product_type: Product type for orders (default: 'MIS')
        order_validity: Order validity (default: 'DAY')
        max_attempts: Maximum number of entry attempts per symbol per day (default: None for no limit)
        max_order_retries: Maximum retry attempts for failed orders (default: 3, None for unlimited)
        take_profit_mult_param: Take profit multiplier (default: 3.0)
        initial_sl_mult_param: Initial stop loss multiplier of true range (default: 0.5)
        use_take_profit_param: Whether to use take profit (default: False)
        trigger_window_minutes_param: Trigger window in minutes (default: 60)
        ws_url: WebSocket URL (default: from live_config.WS_URL)
    """
    global client, timezone, live_run_id
    
    # Use config default for ws_url if not provided
    if ws_url is None:
        ws_url = WS_URL
    
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
        from .db_models import save_live_run
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
        initial_sl_mult=initial_sl_mult_param,
        use_take_profit=use_take_profit_param,
        trigger_window_minutes=trigger_window_minutes_param,
        max_attempts=max_attempts,
        max_order_retries=max_order_retries
    )
    
    if not instruments:
        print("[Live] No instruments to trade. Exiting.")
        return
    
    client.connect()
    # Subscribe to LTP
    client.subscribe_ltp(instruments, on_data_received=on_data_received)
    
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
            time.sleep(1)  # Check every second
    finally:
        client.unsubscribe_ltp(instruments)
        client.disconnect()
        print(f"[Live] Trading session completed. Check logs for details.")
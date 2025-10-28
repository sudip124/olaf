import json
import os
import pandas as pd
import datetime
import threading
import time
from openalgo import api
from backtest_config import OPENALGO_URL, API_KEY, SESSION_START, SESSION_END, EXCHANGE
from strategies.strat80_20 import generate_signals  # Import to reuse params, but we'll adapt logic
import pytz  # Added for timezone handling
import csv  # Added for CSV logging
import logging  # Keep for errors/warnings

# Timezone for Indian market (IST)
IST = pytz.timezone('Asia/Kolkata')

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

# CSV log file for events (similar to strat80_20.py)
signals_logfile = f'logs/live_strat80_20_signals_{today}.csv'
# Create file with header if not exists
if not os.path.exists(signals_logfile):
    with open(signals_logfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'date', 'time', 'event', 'symbol', 'price', 'details'])
        writer.writeheader()

def log_event(event_dict, logfile=signals_logfile):
    """Append event to CSV log file."""
    with open(logfile, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'date', 'time', 'event', 'symbol', 'price', 'details'])
        writer.writerow(event_dict)

# Initialize OpenAlgo client with WebSocket
client = api(
    api_key=API_KEY,
    host=OPENALGO_URL,
    ws_url="ws://127.0.0.1:8765"  # Adjust if different
)

# Load strategy config
config_path = os.path.join(os.path.dirname(__file__), 'strategies', 'strat80_20_config.json')
with open(config_path, 'r') as f:
    params = json.load(f)

# Global params from config
stop_loss_mult = params.get('stop_loss_mult', 2.0)
take_profit_mult = params.get('take_profit_mult', 3.0)
use_take_profit = params.get('use_take_profit', False)
trigger_window_minutes = params.get('trigger_window_minutes', 60)

# Live trading params (add to config if needed)
FIXED_QTY = 1  # Fixed quantity per trade; adjust based on risk management
PRODUCT_TYPE = 'MIS'  # Intraday product for NSE
ORDER_VALIDITY = 'DAY'
STRATEGY_NAME = "strat80_20"  # Define strategy name for API calls

# Per-symbol state class
class SymbolState:
    def __init__(self, symbol, entry_price, trigger_price, tick_size, true_range):
        self.symbol = symbol
        self.entry_price = entry_price
        self.trigger_price = trigger_price
        self.tick_size = tick_size
        self.true_range = true_range
        self.triggered = False
        self.trigger_time = None
        self.day_low = float('inf')
        self.in_position = False
        self.long_entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.ticks = []  # List of {'time': ts, 'price': ltp}
        self.bar_df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])  # Historical 15m bars for today
        self.current_bar_start = None
        self.current_bar_ticks = []
        self.entry_order_id = None  # For tracking buy stop order

    def process_tick(self, ts, ltp):
        # Update day_low
        self.day_low = min(self.day_low, ltp)

        # Determine 15m bar start
        bar_start = ts.floor('15min')

        # If new bar, finalize previous
        if self.current_bar_start is not None and bar_start != self.current_bar_start:
            if self.current_bar_ticks:
                bar_prices = [t['price'] for t in self.current_bar_ticks]
                bar_open = self.current_bar_ticks[0]['price']
                bar_high = max(bar_prices)
                bar_low = min(bar_prices)
                bar_close = self.current_bar_ticks[-1]['price']
                new_bar = pd.DataFrame({
                    'open': [bar_open],
                    'high': [bar_high],
                    'low': [bar_low],
                    'close': [bar_close]
                }, index=[self.current_bar_start])
                self.bar_df = pd.concat([self.bar_df, new_bar])

                # Update trailing SL if in position and green bar
                if self.in_position and bar_close >= bar_open:
                    new_sl = bar_low - 1e-8
                    if new_sl > self.stop_loss:
                        log_event({
                            'timestamp': self.current_bar_start.isoformat(),
                            'date': self.current_bar_start.date().isoformat(),
                            'time': self.current_bar_start.time().isoformat(),
                            'event': 'Trailing SL Update',
                            'symbol': self.symbol,
                            'price': new_sl,
                            'details': f"Updated SL to {new_sl} on green bar"
                        })
                        self.stop_loss = new_sl

            # Reset for new bar
            self.current_bar_ticks = []
            self.current_bar_start = bar_start

        # Add to current bar
        self.current_bar_ticks.append({'time': ts, 'price': ltp})
        if self.current_bar_start is None:
            self.current_bar_start = bar_start

        # Strategy logic
        session_start_dt = IST.localize(datetime.datetime(ts.year, ts.month, ts.day, 9, 15, 0))
        first60_end = session_start_dt + datetime.timedelta(minutes=trigger_window_minutes)

        if not self.triggered and ts < first60_end and ltp <= self.trigger_price:
            self.triggered = True
            self.trigger_time = ts
            log_event({
                'timestamp': ts.isoformat(),
                'date': ts.date().isoformat(),
                'time': ts.time().isoformat(),
                'event': 'Trigger Threshold Hit',
                'symbol': self.symbol,
                'price': ltp,
                'details': f"Price dropped below threshold {self.trigger_price}; day_low_so_far: {self.day_low}"
            })
            # Place buy SL-M order
            self.place_buy_stop()

        # Monitor for exits (even without full bar)
        if self.in_position:
            if ltp <= self.stop_loss:
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Stop Loss Exit',
                    'symbol': self.symbol,
                    'price': ltp,
                    'details': f"Hit SL at {self.stop_loss}"
                })
                self.place_sell_market()
                self.reset_after_exit()  # Reset states after exit to prevent re-triggers
            elif use_take_profit and ltp >= self.take_profit:
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Take Profit Exit',
                    'symbol': self.symbol,
                    'price': ltp,
                    'details': f"Hit TP at {self.take_profit}"
                })
                self.place_sell_market()
                self.reset_after_exit()  # Reset states after exit to prevent re-triggers

    def place_buy_stop(self):
        try:
            response = client.placeorder(
                strategy=STRATEGY_NAME,
                exchange=EXCHANGE,
                symbol=self.symbol,
                action='BUY',
                product=PRODUCT_TYPE,
                price_type='SL-M',
                quantity=FIXED_QTY,
                price=0,
                trigger_price=self.entry_price
            )
            # Accept multiple possible keys for order id returned by SDK
            self.entry_order_id = (
                response.get('order_id') or response.get('orderid') or
                response.get('orderId') or response.get('data', {}).get('order_id')
            )
            log_event({
                'timestamp': datetime.datetime.now(IST).isoformat(),
                'date': datetime.date.today().isoformat(),
                'time': datetime.datetime.now(IST).time().isoformat(),
                'event': 'Buy Stop Order Placed',
                'symbol': self.symbol,
                'price': self.entry_price,
                'details': f"Response: {response}; order_id={self.entry_order_id}"
            })
        except Exception as e:
            logger.error(f"[{self.symbol}] Error placing buy stop: {e}")

    def place_sell_market(self):
        if not self.in_position:  # Extra check to prevent selling when not in position
            logger.warning(f"[{self.symbol}] Skipping sell market: not in position")
            return
        try:
            response = client.placeorder(
                strategy=STRATEGY_NAME,  # Optional
                exchange=EXCHANGE,
                symbol=self.symbol,
                action='SELL',  # Uppercase
                product=PRODUCT_TYPE,
                price_type='MARKET',  # Changed from 'order_type'
                quantity=FIXED_QTY,
                price=0,
                trigger_price=0
            )
            log_event({
                'timestamp': datetime.datetime.now(IST).isoformat(),
                'date': datetime.date.today().isoformat(),
                'time': datetime.datetime.now(IST).time().isoformat(),
                'event': 'Sell Market Order Placed',
                'symbol': self.symbol,
                'price': 0,  # Market price
                'details': f"Response: {response}"
            })
        except Exception as e:
            logger.error(f"[{self.symbol}] Error placing sell market: {e}")

    def place_buy_market(self):  # Added to square off unintended shorts
        try:
            response = client.placeorder(
                strategy=STRATEGY_NAME,
                exchange=EXCHANGE,
                symbol=self.symbol,
                action='BUY',
                product=PRODUCT_TYPE,
                price_type='MARKET',
                quantity=FIXED_QTY,
                price=0,
                trigger_price=0
            )
            log_event({
                'timestamp': datetime.datetime.now(IST).isoformat(),
                'date': datetime.date.today().isoformat(),
                'time': datetime.datetime.now(IST).time().isoformat(),
                'event': 'Buy Market Order Placed (Square Off)',
                'symbol': self.symbol,
                'price': 0,  # Market price
                'details': f"Response: {response}"
            })
        except Exception as e:
            logger.error(f"[{self.symbol}] Error placing buy market to square off: {e}")

    # Callback for order updates (if filled)
    def on_order_update(self, update):
        if update['order_id'] == self.entry_order_id and update['status'] == 'filled':
            fill_price = update['fill_price']  # Assume available
            self.in_position = True
            self.long_entry_price = fill_price
            risk = fill_price - self.day_low
            self.stop_loss = self.day_low - 1e-8
            self.take_profit = fill_price + take_profit_mult * risk if use_take_profit else None
            log_event({
                'timestamp': datetime.datetime.now(IST).isoformat(),
                'date': datetime.date.today().isoformat(),
                'time': datetime.datetime.now(IST).time().isoformat(),
                'event': 'Entry Filled',
                'symbol': self.symbol,
                'price': fill_price,
                'details': f"SL: {self.stop_loss}, TP: {self.take_profit}"
            })

    def reset_after_exit(self):
        self.in_position = False
        self.long_entry_price = None
        self.stop_loss = None
        self.take_profit = None
        # Do not reset triggered or day_low, as re-entries are not allowed per day

# Global dict of symbol states
symbol_states = {}

def on_data_received(data):
    symbol = data['symbol']
    ltp = data['data']['ltp']
    ts = pd.to_datetime(data['data']['timestamp'], unit='ms', utc=True).astimezone(IST)
    if symbol in symbol_states:
        symbol_states[symbol].process_tick(ts, ltp)

def on_order_update_received(update):
    # Assume client has on_order_update callback
    symbol = update['symbol']  # Assume
    if symbol in symbol_states:
        symbol_states[symbol].on_order_update(update)

def load_setups(json_file):
    with open(json_file, 'r') as f:
        setups = json.load(f)
    instruments = []
    for setup in setups:
        symbol = setup['symbol']
        symbol_states[symbol] = SymbolState(
            symbol=symbol,
            entry_price=setup['entry_price'],
            trigger_price=setup['trigger_price'],
            tick_size=setup['tick_size'],
            true_range=setup['true_range']
        )
        instruments.append({"exchange": EXCHANGE, "symbol": symbol})
        log_event({
            'timestamp': datetime.datetime.now(IST).isoformat(),
            'date': datetime.date.today().isoformat(),
            'time': datetime.datetime.now(IST).time().isoformat(),
            'event': 'Setup Loaded',
            'symbol': symbol,
            'price': setup['entry_price'],
            'details': f"Trigger price: {setup['trigger_price']}; TR: {setup['true_range']}"
        })
    return instruments

def poll_orders_and_positions():
    while True:
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

        pos_dict = {pos['symbol']: int(pos.get('netqty', 0)) for pos in positions if pos.get('product') == PRODUCT_TYPE}

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

def start_trading(json_file):
    instruments = load_setups(json_file)
    client.connect()
    # Subscribe to LTP (unchanged)
    client.subscribe_ltp(instruments, on_data_received=on_data_received)
    # Remove: client.subscribe_orders(...) - not supported by SDK

    # Start polling thread for order statuses and positions
    polling_thread = threading.Thread(target=poll_orders_and_positions, daemon=True)
    polling_thread.start()

    # Wait for session start if early
    current_dt = datetime.datetime.now(IST)
    session_start_dt = current_dt.replace(hour=9, minute=15, second=0, microsecond=0)
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
            current_dt = datetime.datetime.now(IST)
            session_end_dt = current_dt.replace(hour=15, minute=30, second=0, microsecond=0)
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

if __name__ == "__main__":
    # Use absolute path by joining with script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "scanner_setups.json")
    
    if not os.path.exists(json_file):
        logger.error(f"JSON file not found at: {json_file}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error("Please ensure scanner_setups.json exists in the same directory as this script")
        exit(1)
    else:
        log_event({
            'timestamp': datetime.datetime.now(IST).isoformat(),
            'date': datetime.date.today().isoformat(),
            'time': datetime.datetime.now(IST).time().isoformat(),
            'event': 'Setup File Found',
            'symbol': '',
            'price': 0,
            'details': f"Found setup file: {json_file}"
        })
        start_trading(json_file)
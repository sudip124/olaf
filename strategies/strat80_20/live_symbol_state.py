import pandas as pd
import datetime

from data_manager.config import EXCHANGE
from .live_config import STRATEGY_NAME

# These will be injected by live.py after initialization
logger = None
client = None
timezone = None
log_event = None
is_market_open = None


class SymbolState:
    def __init__(self, symbol, entry_price, trigger_price, tick_size, true_range,
                 market_open_hour, market_open_minute, market_close_hour, market_close_minute,
                 fixed_qty, product_type, take_profit_mult, initial_sl_mult, 
                 use_take_profit, trigger_window_minutes, max_attempts, max_order_retries):
        self.symbol = symbol
        self.entry_price = entry_price
        self.trigger_price = trigger_price
        self.tick_size = tick_size
        self.true_range = true_range
        
        # Configuration parameters
        self.market_open_hour = market_open_hour
        self.market_open_minute = market_open_minute
        self.market_close_hour = market_close_hour
        self.market_close_minute = market_close_minute
        self.fixed_qty = fixed_qty
        self.product_type = product_type
        self.take_profit_mult = take_profit_mult
        self.initial_sl_mult = initial_sl_mult
        self.use_take_profit = use_take_profit
        self.trigger_window_minutes = trigger_window_minutes
        self.max_attempts = max_attempts
        self.max_order_retries = max_order_retries
        
        # State variables
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
        self.entries_today = 0  # Count of entries filled today
        self.order_rejected = False  # Track if last order was rejected
        self.last_rejection_time = None  # When the order was last rejected
        self.pending_retry = False  # Flag to indicate order needs retry when market opens
        self.retry_count = 0  # Track number of retry attempts for current trigger

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
                
                # Avoid FutureWarning by checking if DataFrame is empty
                if self.bar_df.empty:
                    self.bar_df = new_bar
                else:
                    self.bar_df = pd.concat([self.bar_df, new_bar])

                        # Update trailing SL if in position and green bar
                if self.in_position and self.stop_loss is not None and bar_close >= bar_open:
                    new_sl = bar_low - 1e-8
                    if new_sl > self.stop_loss:
                        logger.info(f"[{self.symbol}] Trailing SL update: {self.stop_loss} -> {new_sl}")
                        log_event({
                            'timestamp': self.current_bar_start.isoformat(),
                            'date': self.current_bar_start.date().isoformat(),
                            'time': self.current_bar_start.time().isoformat(),
                            'event': 'Trailing SL Update',
                            'symbol': self.symbol,
                            'price': new_sl,
                            'details': f"Updated SL from {self.stop_loss} to {new_sl} on green bar (bar_low: {bar_low})"
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
        session_start_dt = timezone.localize(datetime.datetime(ts.year, ts.month, ts.day, self.market_open_hour, self.market_open_minute, 0))
        first60_end = session_start_dt + datetime.timedelta(minutes=self.trigger_window_minutes)

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
            # Only place if under max attempts limit
            if self.max_attempts is None or self.entries_today < self.max_attempts:
                self.place_buy_stop(ts)
            else:
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Entry Skipped - Max Attempts Reached',
                    'symbol': self.symbol,
                    'price': ltp,
                    'details': f"Max attempts reached ({self.max_attempts}); not placing initial buy stop"
                })

        # Re-arm logic: after exit, if already triggered and not in position, and no pending order,
        # place a fresh buy stop when price is below entry_price and entries_today < max_attempts
        # Also retry if order was rejected and market is now open
        if self.triggered and not self.in_position and self.entry_order_id is None:
            can_retry = (self.max_attempts is None) or (self.entries_today < self.max_attempts)
            # Allow retry if:
            # 1. Price is below entry_price (normal condition)
            # 2. Order was rejected and we're flagged for retry (pending_retry)
            should_retry = can_retry and (ltp <= self.entry_price or self.pending_retry)
            if should_retry and is_market_open(ts, self.market_open_hour, self.market_open_minute,
                                               self.market_close_hour, self.market_close_minute):
                self.place_buy_stop(ts)
                self.pending_retry = False  # Clear the retry flag after attempting

        # Monitor for exits (even without full bar)
        if self.in_position:
            # Defensive check: ensure stop_loss is set
            if self.stop_loss is None:
                logger.error(f"[{self.symbol}] In position but stop_loss is None! Setting emergency SL.")
                self.stop_loss = self.long_entry_price - (self.initial_sl_mult * self.true_range) if self.long_entry_price else self.day_low - 1e-8
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Emergency SL Set',
                    'symbol': self.symbol,
                    'price': self.stop_loss,
                    'details': f"Stop loss was None, setting emergency SL: {self.stop_loss}"
                })
            
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
            elif self.use_take_profit and self.take_profit is not None and ltp >= self.take_profit:
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

    def place_buy_stop(self, ts=None):
        """Place a buy stop order. Checks market hours before placing.
        
        Args:
            ts: Current timestamp (timezone aware). If None, uses current time.
        """
        if ts is None:
            ts = datetime.datetime.now(timezone)
        
        # Check if market is open
        if not is_market_open(ts, self.market_open_hour, self.market_open_minute, 
                              self.market_close_hour, self.market_close_minute):
            log_event({
                'timestamp': ts.isoformat(),
                'date': ts.date().isoformat(),
                'time': ts.time().isoformat(),
                'event': 'Order Delayed - Market Closed',
                'symbol': self.symbol,
                'price': self.entry_price,
                'details': f"Market not open yet. Will retry when market opens at {self.market_open_hour}:{self.market_open_minute:02d}"
            })
            self.pending_retry = True  # Flag for retry when market opens
            return
        
        try:
            response = client.placeorder(
                strategy=STRATEGY_NAME,
                exchange=EXCHANGE,
                symbol=self.symbol,
                action='BUY',
                product=self.product_type,
                price_type='SL-M',
                quantity=self.fixed_qty,
                price=0,
                trigger_price=self.entry_price
            )
            
            # Check if order was rejected
            if response.get('status') == 'error':
                self.order_rejected = True
                self.last_rejection_time = ts
                self.entry_order_id = None
                self.retry_count += 1
                
                # Check if we've exceeded retry limit
                if self.max_order_retries is not None and self.retry_count >= self.max_order_retries:
                    self.pending_retry = False
                    log_event({
                        'timestamp': ts.isoformat(),
                        'date': ts.date().isoformat(),
                        'time': ts.time().isoformat(),
                        'event': 'Order Rejected - Max Retries Reached',
                        'symbol': self.symbol,
                        'price': self.entry_price,
                        'details': f"Response: {response}. Max retries ({self.max_order_retries}) reached. Giving up."
                    })
                else:
                    self.pending_retry = True
                    retry_info = f"retry {self.retry_count}/{self.max_order_retries if self.max_order_retries else 'unlimited'}"
                    log_event({
                        'timestamp': ts.isoformat(),
                        'date': ts.date().isoformat(),
                        'time': ts.time().isoformat(),
                        'event': 'Order Rejected',
                        'symbol': self.symbol,
                        'price': self.entry_price,
                        'details': f"Response: {response}. Will retry ({retry_info})."
                    })
            else:
                # Accept multiple possible keys for order id returned by SDK
                self.entry_order_id = (
                    response.get('order_id') or response.get('orderid') or
                    response.get('orderId') or response.get('data', {}).get('order_id')
                )
                self.order_rejected = False
                self.pending_retry = False
                # Reset retry count on successful order placement
                retry_info = f" (after {self.retry_count} retries)" if self.retry_count > 0 else ""
                self.retry_count = 0
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Buy Stop Order Placed',
                    'symbol': self.symbol,
                    'price': self.entry_price,
                    'details': f"Response: {response}; order_id={self.entry_order_id}{retry_info}"
                })
        except Exception as e:
            logger.error(f"[{self.symbol}] Error placing buy stop: {e}")
            self.order_rejected = True
            self.last_rejection_time = ts
            self.entry_order_id = None
            self.retry_count += 1
            
            # Check if we've exceeded retry limit
            if self.max_order_retries is not None and self.retry_count >= self.max_order_retries:
                self.pending_retry = False
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Order Failed - Max Retries Reached',
                    'symbol': self.symbol,
                    'price': self.entry_price,
                    'details': f"Exception: {e}. Max retries ({self.max_order_retries}) reached. Giving up."
                })
            else:
                self.pending_retry = True

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
                product=self.product_type,
                price_type='MARKET',  # Changed from 'order_type'
                quantity=self.fixed_qty,
                price=0,
                trigger_price=0
            )
            now = datetime.datetime.now(timezone)
            log_event({
                'timestamp': now.isoformat(),
                'date': now.date().isoformat(),
                'time': now.time().isoformat(),
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
                product=self.product_type,
                price_type='MARKET',
                quantity=self.fixed_qty,
                price=0,
                trigger_price=0
            )
            now = datetime.datetime.now(timezone)
            log_event({
                'timestamp': now.isoformat(),
                'date': now.date().isoformat(),
                'time': now.time().isoformat(),
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
            fill_price = float(update['fill_price'])  # Ensure it's a float
            self.in_position = True
            self.long_entry_price = fill_price
            # New SL calculation: entry_price - (initial_sl_mult * true_range)
            self.stop_loss = fill_price - (self.initial_sl_mult * self.true_range)
            risk = fill_price - self.stop_loss
            self.take_profit = fill_price + self.take_profit_mult * risk if self.use_take_profit else None
            # Count this entry
            self.entries_today += 1
            
            # Log with detailed info
            logger.info(f"[{self.symbol}] Entry filled at {fill_price}, TR={self.true_range}, risk={risk}, SL={self.stop_loss}, TP={self.take_profit}")
            now = datetime.datetime.now(timezone)
            log_event({
                'timestamp': now.isoformat(),
                'date': now.date().isoformat(),
                'time': now.time().isoformat(),
                'event': 'Entry Filled',
                'symbol': self.symbol,
                'price': fill_price,
                'details': f"SL: {self.stop_loss} (entry - {self.initial_sl_mult}*TR), TP: {self.take_profit}, risk: {risk}, TR: {self.true_range}; entries_today={self.entries_today}/{self.max_attempts if self.max_attempts is not None else 'unlimited'}"
            })

    def reset_after_exit(self):
        self.in_position = False
        self.long_entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.order_rejected = False
        self.pending_retry = False
        self.retry_count = 0  # Reset retry count after exit
        # Do not reset triggered or day_low, as re-entries are not allowed per day
        # With retry enabled, keeping 'triggered' True allows re-arming above; limit enforced via entries_today/MAX_ENTRIES

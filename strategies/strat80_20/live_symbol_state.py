import pandas as pd
import datetime
import threading
from decimal import Decimal, ROUND_HALF_UP

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
                 qty, product_type, take_profit_mult, initial_sl_ticks, 
                 use_take_profit, trigger_window_minutes, max_attempts, max_order_retries,
                 order_timeout_minutes=None, exit_before_close_minutes=None,
                 ignore_initial_minutes: int = 1):
        # Threading lock for protecting critical state transitions
        self._state_lock = threading.Lock()
        
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
        self.qty = qty
        self.product_type = product_type
        self.take_profit_mult = take_profit_mult
        self.initial_sl_ticks = initial_sl_ticks
        self.use_take_profit = use_take_profit
        self.trigger_window_minutes = trigger_window_minutes
        self.max_attempts = max_attempts
        self.max_order_retries = max_order_retries
        # New optional controls (fallback to sensible defaults if None)
        self.exit_before_close_minutes = int(exit_before_close_minutes) if exit_before_close_minutes is not None else 45
        self.ignore_initial_minutes = int(ignore_initial_minutes) if ignore_initial_minutes is not None else 1
        
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
        self.order_placed_time = None  # When the current order was placed
        # Default to 15 if not provided
        self.order_timeout_minutes = int(order_timeout_minutes) if order_timeout_minutes is not None else 15  # minutes
        # Broker-held protective Sell Stop order id
        self.sl_order_id = None
        # Track processed order update ids to avoid duplicate handling
        self.processed_order_ids = set()
        self.last_filled_entry_order_id = None
        # Cache for SL order status to reduce API calls
        self.sl_order_last_check_time = None
        self.sl_order_cached_status = None

    def process_tick(self, ts, ltp):
        # Ignore all ticks for the first N minutes after market open (configurable)
        session_open_dt = timezone.localize(
            datetime.datetime(ts.year, ts.month, ts.day, self.market_open_hour, self.market_open_minute, 0)
        )
        if ts < session_open_dt + datetime.timedelta(minutes=self.ignore_initial_minutes):
            return

        # Update day_low (no lock needed - monotonic decrease, no critical state)
        self.day_low = min(self.day_low, ltp)
        
        # CRITICAL SECTION: Check if we need to cancel stale orders
        # Must read entry_order_id and order_placed_time atomically
        logger.trace(f"[{self.symbol}] [TID:{threading.get_ident()}] Acquiring lock for stale order check") if hasattr(logger, 'trace') else None
        with self._state_lock:
            if self.entry_order_id is not None and self.order_placed_time is not None:
                time_since_order = (ts - self.order_placed_time).total_seconds() / 60
                if time_since_order > self.order_timeout_minutes:
                    order_to_cancel = self.entry_order_id
                    logger.info(f"[{self.symbol}] [TID:{threading.get_ident()}] Canceling stale order {order_to_cancel} (placed {time_since_order:.1f} min ago)")
                    # Release lock before API call to avoid blocking
                    # We copied order_id so it's safe
        
        # Cancel order outside lock (API call can take time)
        if 'order_to_cancel' in locals():
            logger.debug(f"[{self.symbol}] [TID:{threading.get_ident()}] Calling cancel_order API for {order_to_cancel}")
            cancelled = self.cancel_order(order_to_cancel, ts)
            logger.trace(f"[{self.symbol}] [TID:{threading.get_ident()}] Acquiring lock for post-cancel re-check") if hasattr(logger, 'trace') else None
            with self._state_lock:
                # Re-check: only clear if order_id hasn't changed (idempotency check)
                if self.entry_order_id == order_to_cancel:
                    if cancelled:
                        logger.debug(f"[{self.symbol}] [TID:{threading.get_ident()}] Cleared entry_order_id after successful cancel")
                        self.entry_order_id = None
                        self.order_placed_time = None
                    else:
                        logger.warning(f"[{self.symbol}] [TID:{threading.get_ident()}] Cancel failed but order_id unchanged")
                        log_event({
                            'timestamp': ts.isoformat(),
                            'date': ts.date().isoformat(),
                            'time': ts.strftime('%H:%M:%S'),
                            'event': 'Order Cancel Failed',
                            'symbol': self.symbol,
                            'price': 0,
                            'details': f"Cancel failed for order {order_to_cancel}; will keep state and retry on next cycle"
                        })
                else:
                    logger.warning(f"[{self.symbol}] [TID:{threading.get_ident()}] RACE DETECTED: entry_order_id changed from {order_to_cancel} to {self.entry_order_id} during cancel")

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

                # Trailing stoploss logic: Update on green bars (price moving up)
                # This implements the OpenAlgo trailing pattern by modifying the broker-held SELL STOP order
                # CRITICAL SECTION: Read in_position and stop_loss atomically
                with self._state_lock:
                    should_trail = self.in_position and self.stop_loss is not None and bar_open <= bar_close
                    current_sl = self.stop_loss
                
                if should_trail:
                    # Calculate new stoploss with a meaningful buffer below bar low
                    # Use 2 ticks below bar_low to provide cushion against minor retracements
                    buffer_ticks = 2
                    new_sl = bar_low - (buffer_ticks * self.tick_size)
                    # Round new trailing SL to nearest tick before comparison and assignment using Decimal precision
                    new_sl = float(Decimal(str(new_sl / self.tick_size)).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * Decimal(str(self.tick_size)))
                    
                    # Only trail upward (never move stoploss down)
                    if new_sl > current_sl:
                        # CRITICAL SECTION: Update stop_loss atomically
                        with self._state_lock:
                            # Re-check after acquiring lock (might have changed)
                            if new_sl > self.stop_loss:
                                old_sl = self.stop_loss
                                self.stop_loss = new_sl
                                logger.info(f"[{self.symbol}] Trailing SL update: {old_sl} -> {new_sl}")
                                log_event({
                                    'timestamp': self.current_bar_start.isoformat(),
                                    'date': self.current_bar_start.date().isoformat(),
                                    'time': self.current_bar_start.time().isoformat(),
                                    'event': 'Trailing SL Update',
                                    'symbol': self.symbol,
                                    'price': new_sl,
                                    'details': f"Updated SL from {old_sl} to {new_sl} on green bar (bar_open: {bar_open}, bar_close: {bar_close}, bar_low: {bar_low})"
                                })
                                # Modify the broker-held SELL STOP order to the new trailing stoploss
                                # place_or_update_sell_stop will acquire lock internally
                        self.place_or_update_sell_stop(new_sl, self.current_bar_start)

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

        # CRITICAL SECTION: Check trigger state and entries atomically
        with self._state_lock:
            should_trigger = not self.triggered and ts < first60_end and ltp <= self.trigger_price
            if should_trigger:
                self.triggered = True
                self.trigger_time = ts
                can_place = self.max_attempts is None or self.entries_today < self.max_attempts
        
        if should_trigger:
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
            if can_place:
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
        # CRITICAL SECTION: Check all conditions atomically to prevent race conditions
        with self._state_lock:
            should_attempt_reentry = (
                self.triggered and 
                not self.in_position and 
                self.entry_order_id is None and
                ((self.max_attempts is None) or (self.entries_today < self.max_attempts)) and
                (ltp <= self.entry_price or self.pending_retry) and
                is_market_open(ts, self.market_open_hour, self.market_open_minute,
                              self.market_close_hour, self.market_close_minute)
            )
            if should_attempt_reentry:
                # Clear retry flag while we have the lock
                self.pending_retry = False 
        
        if should_attempt_reentry:
            self.place_buy_stop(ts)

        # Auto-exit before market close based on configured minutes
        market_close_dt = ts.replace(hour=self.market_close_hour, minute=self.market_close_minute, second=0, microsecond=0)
        exit_cutoff_dt = market_close_dt - datetime.timedelta(minutes=self.exit_before_close_minutes)
        
        # CRITICAL SECTION: Check if should auto-exit atomically
        with self._state_lock:
            should_auto_exit = self.in_position and ts >= exit_cutoff_dt
            if should_auto_exit:
                sl_to_cancel = self.sl_order_id
        
        if should_auto_exit:
            log_event({
                'timestamp': ts.isoformat(),
                'date': ts.date().isoformat(),
                'time': ts.time().isoformat(),
                'event': 'Auto Exit - Market Close',
                'symbol': self.symbol,
                'price': ltp,
                'details': f"Exiting position {self.exit_before_close_minutes} min before market close (cutoff: {exit_cutoff_dt.time().isoformat(timespec='seconds')})"
            })
            # Cancel protective SL if present to avoid duplicate exits
            if sl_to_cancel:
                cancelled = self.cancel_order(sl_to_cancel, ts)
                with self._state_lock:
                    if cancelled and self.sl_order_id == sl_to_cancel:
                        self.sl_order_id = None
            self.place_sell_market()
            with self._state_lock:
                self.reset_after_exit()
            return  # Skip further processing this tick

        # Monitor for exits - broker-held SELL STOP will handle stoploss automatically
        # We only keep emergency checks here
        # CRITICAL SECTION: Check position state atomically
        with self._state_lock:
            should_check_emergency = self.in_position
            stop_loss_missing = should_check_emergency and self.stop_loss is None
            sl_order_missing = should_check_emergency and not stop_loss_missing and self.sl_order_id is None
            current_stop_loss = self.stop_loss
        
        if should_check_emergency:
            # Defensive check: ensure stop_loss is set and broker SL order exists
            if stop_loss_missing:
                logger.error(f"[{self.symbol}] In position but stop_loss is None! Setting emergency SL.")
                emergency_sl = self.long_entry_price - (self.initial_sl_ticks * self.tick_size) if self.long_entry_price else self.day_low - 1e-8
                # Use Decimal precision to avoid floating-point errors
                emergency_sl = float(Decimal(str(emergency_sl / self.tick_size)).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * Decimal(str(self.tick_size)))
                with self._state_lock:
                    self.stop_loss = emergency_sl
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Emergency SL Set',
                    'symbol': self.symbol,
                    'price': emergency_sl,
                    'details': f"Stop loss was None, setting emergency SL: {emergency_sl}"
                })
                # Place the missing broker SL order
                self.place_or_update_sell_stop(emergency_sl, ts)
            
            # Emergency failsafe: if broker-held SL order is missing, place it
            elif sl_order_missing:
                logger.warning(f"[{self.symbol}] In position but no broker SL order! Placing emergency SL.")
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Emergency SL Order Placed',
                    'symbol': self.symbol,
                    'price': current_stop_loss,
                    'details': f"Broker SL order missing, placing at {current_stop_loss}"
                })
                self.place_or_update_sell_stop(current_stop_loss, ts)


    def place_buy_stop(self, ts=None):
        """Place a buy stop order. Checks market hours and position before placing.
        
        Args:
            ts: Current timestamp (timezone aware). If None, uses current time.
        """
        if ts is None:
            ts = datetime.datetime.now(timezone)
        
        # CRITICAL SECTION: Check all conditions atomically before placing order
        logger.trace(f"[{self.symbol}] [TID:{threading.get_ident()}] Acquiring lock for place_buy_stop checks") if hasattr(logger, 'trace') else None
        with self._state_lock:
            # CRITICAL: Check if already in position to prevent duplicate orders
            if self.in_position:
                logger.warning(f"[{self.symbol}] [TID:{threading.get_ident()}] Skipping buy stop order: already in position")
                return
            
            # CRITICAL: Check if there's already a pending entry order to prevent duplicates
            # This handles race conditions where order fills but callback hasn't fired yet
            if self.entry_order_id is not None and self.entry_order_id != "":
                logger.warning(f"[{self.symbol}] [TID:{threading.get_ident()}] Skipping buy stop order: pending order {self.entry_order_id} already exists")
                return
            
            # All checks passed - proceed with order placement
            logger.debug(f"[{self.symbol}] [TID:{threading.get_ident()}] All checks passed, proceeding with buy stop order")
            # No other thread can change in_position or entry_order_id until we set them
        
        # Check if market is open (outside lock - doesn't access mutable state)
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
            with self._state_lock:
                self.pending_retry = True  # Flag for retry when market opens
            return
        
        try:
            # Use SL (Stop-Limit) order type with both price and trigger_price
            # This is the proper way to place a BUY STOP order per OpenAlgo docs
            response = client.placeorder(
                strategy=STRATEGY_NAME,
                exchange=EXCHANGE,
                symbol=self.symbol,
                action='BUY',
                product=self.product_type,
                price_type='SL',
                quantity=self.qty,
                price=self.entry_price,  # Limit price (same as trigger for immediate execution)
                trigger_price=self.entry_price  # Trigger price
            )
            
            # Check if order was rejected
            if response.get('status') == 'error':
                with self._state_lock:
                    self.order_rejected = True
                    self.last_rejection_time = ts
                    self.entry_order_id = None
                    self.retry_count += 1
                    retry_count_copy = self.retry_count
                
                # Check if we've exceeded retry limit (use local copy)
                if self.max_order_retries is not None and retry_count_copy >= self.max_order_retries:
                    with self._state_lock:
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
                    with self._state_lock:
                        self.pending_retry = True
                    retry_info = f"retry {retry_count_copy}/{self.max_order_retries if self.max_order_retries else 'unlimited'}"
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
                order_id = (
                    response.get('order_id') or response.get('orderid') or
                    response.get('orderId') or response.get('data', {}).get('order_id')
                )
                with self._state_lock:
                    self.entry_order_id = order_id
                    self.order_placed_time = ts  # Track when order was placed
                    self.order_rejected = False
                    self.pending_retry = False
                    # Reset retry count on successful order placement
                    retry_info = f" (after {self.retry_count} retries)" if self.retry_count > 0 else ""
                    self.retry_count = 0
                    order_id_copy = self.entry_order_id
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Buy Stop Order Placed',
                    'symbol': self.symbol,
                    'price': self.entry_price,
                    'details': f"Response: {response}; order_id={order_id_copy}{retry_info}"
                })
        except Exception as e:
            logger.error(f"[{self.symbol}] Error placing buy stop: {e}")
            with self._state_lock:
                self.order_rejected = True
                self.last_rejection_time = ts
                self.entry_order_id = None
                self.retry_count += 1
    
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
                quantity=self.qty,
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
                quantity=self.qty,
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

    def place_or_update_sell_stop(self, new_stop, ts=None):
        """Place or modify a SELL STOP order for protective stoploss.
        
        This method implements the trailing stoploss pattern from OpenAlgo:
        1. If sl_order_id exists and is open, modify it to new_stop price
        2. If sl_order_id doesn't exist or is not open, place a new SELL STOP order
        
        Args:
            new_stop: New stoploss price (will be rounded to tick_size)
            ts: Current timestamp (timezone aware). If None, uses current time.
        """
        if ts is None:
            ts = datetime.datetime.now(timezone)
        
        # Round to tick size using Decimal precision to avoid floating-point errors
        new_stop = float(Decimal(str(new_stop / self.tick_size)).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * Decimal(str(self.tick_size)))
        
        # CRITICAL SECTION: Check all conditions atomically
        with self._state_lock:
            # Safety checks
            if not self.in_position:
                logger.warning(f"[{self.symbol}] Cannot place/update SL: not in position")
                return
            if new_stop is None:
                logger.warning(f"[{self.symbol}] Cannot place/update SL: new_stop is None")
                return
            
            # Skip if new_stop equals current stop_loss (no change needed)
            if self.stop_loss is not None and abs(new_stop - self.stop_loss) < 1e-9:
                logger.debug(f"[{self.symbol}] Skipping SL update: new_stop ({new_stop}) equals current stop_loss ({self.stop_loss})")
                return
            
            # Copy sl_order_id for use outside lock
            current_sl_order_id = self.sl_order_id
        
        try:
            # If we have an existing SL order, try to modify it
            if self.sl_order_id:
                try:
                    # Check if the order is still open/pending
                    status_resp = client.orderstatus(order_id=self.sl_order_id, strategy=STRATEGY_NAME)
                    data = status_resp.get('data') if isinstance(status_resp, dict) else None
                    cur_status = (data or status_resp).get('order_status') if isinstance((data or status_resp), dict) else None
                    
                    # Check if order is in a modifiable state
                    is_open = str(cur_status).strip().lower() in (
                        'open', 'trigger pending', 'pending', 'validation pending', 
                        'put order req received', 'trigger_pending'
                    ) if cur_status else False
                    
                    if is_open:
                        # Modify existing SELL STOP order
                        logger.info(f"[{self.symbol}] Modifying SELL STOP order {self.sl_order_id} to {new_stop}")
                        resp = client.modifyorder(
                            order_id=self.sl_order_id,
                            strategy=STRATEGY_NAME,
                            symbol=self.symbol,
                            exchange=EXCHANGE,
                            action='SELL',
                            product=self.product_type,
                            price_type='SL',
                            quantity=self.qty,
                            price=new_stop,
                            trigger_price=new_stop
                        )
                        
                        # Check if modify was successful
                        if resp.get('status') == 'success' or 'success' in str(resp).lower():
                            log_event({
                                'timestamp': ts.isoformat(),
                                'date': ts.date().isoformat(),
                                'time': ts.time().isoformat(),
                                'event': 'Protective SL Modified',
                                'symbol': self.symbol,
                                'price': new_stop,
                                'details': f"Modified order {self.sl_order_id} to trigger={new_stop}; response={resp}"
                            })
                            return  # Successfully modified, we're done
                        else:
                            logger.warning(f"[{self.symbol}] Modify order failed: {resp}. Will place new order.")
                            self.sl_order_id = None  # Clear and place new order
                    else:
                        logger.info(f"[{self.symbol}] SL order {self.sl_order_id} not open (status: {cur_status}). Placing new order.")
                        self.sl_order_id = None  # Clear and place new order
                        
                except Exception as e:
                    logger.warning(f"[{self.symbol}] Error checking/modifying SL order: {e}. Will place new order.")
                    self.sl_order_id = None  # Clear and place new order
            
            # Place new SELL STOP order (either no existing order, or modify failed)
            logger.info(f"[{self.symbol}] Placing new SELL STOP order at {new_stop}")
            
            # CRITICAL: Set sl_order_id to placeholder BEFORE placing order
            # This prevents the emergency failsafe in process_tick() from triggering
            # during the brief window between placing the order and getting the response
            self.sl_order_id = "PENDING"
            
            resp = client.placeorder(
                strategy=STRATEGY_NAME,
                exchange=EXCHANGE,
                symbol=self.symbol,
                action='SELL',
                product=self.product_type,
                price_type='SL',
                quantity=self.qty,
                price=new_stop,
                trigger_price=new_stop
            )
            
            # Extract order ID from response
            order_id = (
                resp.get('order_id') or resp.get('orderid') or 
                resp.get('orderId') or resp.get('data', {}).get('order_id')
            )
            
            # Update with actual order ID (or clear if failed)
            if order_id:
                self.sl_order_id = order_id
            else:
                self.sl_order_id = None
            
            if self.sl_order_id and self.sl_order_id != "PENDING":
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.time().isoformat(),
                    'event': 'Protective SL Placed',
                    'symbol': self.symbol,
                    'price': new_stop,
                    'details': f"Placed SELL STOP order {self.sl_order_id} at trigger={new_stop}; response={resp}"
                })
            else:
                logger.error(f"[{self.symbol}] Failed to extract order_id from response: {resp}")
                self.sl_order_id = None  # Clear the PENDING placeholder
                
        except Exception as e:
            logger.error(f"[{self.symbol}] Error placing/updating Sell Stop: {e}")
            log_event({
                'timestamp': ts.isoformat(),
                'date': ts.date().isoformat(),
                'time': ts.time().isoformat(),
                'event': 'Protective SL Error',
                'symbol': self.symbol,
                'price': new_stop,
                'details': f"Exception: {e}"
            })

    # Callback for order updates (if filled)
    def on_order_update(self, update):
        order_id = update.get('order_id') or update.get('orderId') or update.get('orderid')
        status = update.get('status') or update.get('order_status') or update.get('orderStatus')
        # Debounce duplicate updates
        if order_id in getattr(self, 'processed_order_ids', set()):
            return
        
        if order_id == self.entry_order_id and status == 'filled':
            # Check if we already processed this fill (prevents race conditions)
            if self.last_filled_entry_order_id == order_id:
                self.processed_order_ids.add(order_id)
                return
            
            # Extract fill price with fallbacks for different broker response formats
            fill_price = (update.get('fill_price') or update.get('average_price') or 
                         update.get('avgprice') or update.get('averagePrice'))
            if not fill_price:
                logger.error(f"[{self.symbol}] [TID:{threading.get_ident()}] No fill_price in order update: {update}")
                return
            fill_price = float(fill_price)
            
            # CRITICAL SECTION: Acquire lock to prevent polling thread from interfering
            # during position state transition
            logger.trace(f"[{self.symbol}] [TID:{threading.get_ident()}] Acquiring lock for entry fill state transition") if hasattr(logger, 'trace') else None
            with self._state_lock:
                logger.info(f"[{self.symbol}] [TID:{threading.get_ident()}] Setting in_position=True for order {order_id}")
                self.in_position = True
                self.long_entry_price = fill_price
                # New SL calculation: entry_price - (initial_sl_ticks * tick_size)
                self.stop_loss = fill_price - (self.initial_sl_ticks * self.tick_size)
                # Round SL to nearest tick using Decimal precision to prevent floating-point errors and broker rejections
                self.stop_loss = float(Decimal(str(self.stop_loss / self.tick_size)).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * Decimal(str(self.tick_size)))
                # Recompute risk using rounded SL
                risk = fill_price - self.stop_loss
                # Calculate TP from risk and round to nearest tick using Decimal precision
                self.take_profit = fill_price + self.take_profit_mult * risk if self.use_take_profit else None
                if self.take_profit is not None:
                    self.take_profit = float(Decimal(str(self.take_profit / self.tick_size)).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * Decimal(str(self.tick_size)))
                # Count this entry only once per unique filled order id
                if self.last_filled_entry_order_id != order_id:
                    self.entries_today += 1
                    self.last_filled_entry_order_id = order_id
                    
                    # ANOMALY MONITORING: Detect if entries_today exceeds max_attempts unexpectedly
                    if self.max_attempts is not None and self.entries_today > self.max_attempts:
                        logger.error(f"[{self.symbol}] [TID:{threading.get_ident()}] ANOMALY DETECTED: entries_today ({self.entries_today}) EXCEEDS max_attempts ({self.max_attempts})! This should never happen - indicates race condition or logic bug!")
                # Clear any pending retry since entry succeeded
                self.pending_retry = False
                # CRITICAL: Clear entry_order_id after fill to prevent re-arm logic from being blocked
                # The re-arm logic checks entry_order_id is None before placing new orders
                # Without this, the strategy cannot re-enter after an exit
                self.entry_order_id = None
            
            # Log with detailed info
            logger.info(f"[{self.symbol}] Entry filled at {fill_price}, initial_sl_ticks={self.initial_sl_ticks}, risk={risk}, SL={self.stop_loss}, TP={self.take_profit}")
            now = datetime.datetime.now(timezone)
            log_event({
                'timestamp': now.isoformat(),
                'date': now.date().isoformat(),
                'time': now.time().isoformat(),
                'event': 'Entry Filled',
                'symbol': self.symbol,
                'price': fill_price,
                'details': f"Long entry #{self.entries_today}; SL: {self.stop_loss} (entry - {self.initial_sl_ticks} ticks); TP: {self.take_profit} (use_take_profit: {self.use_take_profit}); risk: {risk}; entries_today={self.entries_today}/{self.max_attempts if self.max_attempts is not None else 'unlimited'}"
            })
            # Place initial broker-held protective Sell Stop
            self.place_or_update_sell_stop(self.stop_loss, now)
            # Mark this order update as processed to avoid duplicate handling
            self.processed_order_ids.add(order_id)
        # If broker-held SL fills, clear state and exit
        elif self.sl_order_id and order_id == self.sl_order_id and status in ('filled', 'complete'):
            ts_now = datetime.datetime.now(timezone)
            log_event({
                'timestamp': ts_now.isoformat(),
                'date': ts_now.date().isoformat(),
                'time': ts_now.time().isoformat(),
                'event': 'Stop Loss Filled',
                'symbol': self.symbol,
                'price': update.get('fill_price') or 0,
                'details': f"Broker-held Sell Stop filled; order_id={self.sl_order_id}"
            })
            # CRITICAL SECTION: Acquire lock before clearing position state
            with self._state_lock:
                self.sl_order_id = None
                self.processed_order_ids.add(order_id)
                self.reset_after_exit()

    def cancel_order(self, order_id, ts=None):
        """Cancel a pending order. Returns True on success, False on failure.
        First verifies broker reports the order is still open before canceling."""
        if ts is None:
            ts = datetime.datetime.now(timezone)
        # Verify order status before cancel to avoid broker 400/500
        is_open = None
        status_str = None
        try:
            order_status = client.orderstatus(order_id=order_id, strategy=STRATEGY_NAME)
            if isinstance(order_status, dict):
                # Common schemas: {'data': {'order_status': 'OPEN'}} or {'order_status': 'OPEN'}
                data = order_status.get('data') if isinstance(order_status.get('data'), dict) else order_status
                status_str = (data or {}).get('order_status') or (data or {}).get('status')
            if status_str:
                status_norm = str(status_str).strip().lower()
                # Consider typical open-like states that are cancelable
                is_open = status_norm in ('open', 'put order req received', 'validation pending', 'trigger pending', 'pending')
        except Exception as e:
            logger.warning(f"[{self.symbol}] orderstatus check failed for {order_id}: {e}")
            # If status check fails, proceed to attempt cancel as best-effort
            is_open = None

        if is_open is False:
            log_event({
                'timestamp': ts.isoformat(),
                'date': ts.date().isoformat(),
                'time': ts.strftime('%H:%M:%S'),
                'event': 'Order Cancel Skipped',
                'symbol': self.symbol,
                'price': 0,
                'details': f"Order {order_id} not open (status: {status_str}); skipping cancel"
            })
            return False

        try:
            response = client.cancelorder(
                order_id=order_id,
                strategy=STRATEGY_NAME
            )
            # Determine success from broker response
            status = None
            message = None
            if isinstance(response, dict):
                status = response.get('status') or response.get('Status')
                message = response.get('message') or response.get('Message')
            else:
                message = str(response)

            success = (str(status).lower() == 'success') if status is not None else ('success' in str(response).lower())

            if success:
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.strftime('%H:%M:%S'),
                    'event': 'Order Cancelled',
                    'symbol': self.symbol,
                    'price': 0,
                    'details': f"Cancelled order {order_id}. Response: {response}"
                })
                return True
            else:
                log_event({
                    'timestamp': ts.isoformat(),
                    'date': ts.date().isoformat(),
                    'time': ts.strftime('%H:%M:%S'),
                    'event': 'Order Cancel Failed',
                    'symbol': self.symbol,
                    'price': 0,
                    'details': f"Failed to cancel order {order_id}. Response: {response}"
                })
                return False
        except Exception as e:
            logger.error(f"[{self.symbol}] Error canceling order {order_id}: {e}")
            log_event({
                'timestamp': ts.isoformat(),
                'date': ts.date().isoformat(),
                'time': ts.strftime('%H:%M:%S'),
                'event': 'Order Cancel Failed',
                'symbol': self.symbol,
                'price': 0,
                'details': f"Exception canceling order {order_id}: {e}"
            })
            return False
    
    def reset_after_exit(self):
        """Reset symbol state after position exit.
        
        CRITICAL: This method MUST be called with self._state_lock held!
        It modifies multiple shared state variables atomically.
        """
        # Note: Lock MUST be held by caller - we don't acquire it here to avoid nested locks
        self.in_position = False
        self.long_entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.order_rejected = False
        self.pending_retry = False
        self.retry_count = 0  # Reset retry count after exit
        self.trigger_time = None  # Reset trigger time
        self.triggered = False  # Reset triggered flag - re-entry requires NEW trigger hit (matches backtest)
        self.order_placed_time = None  # Reset order timestamp
        self.sl_order_id = None  # Clear SL order id
        self.last_filled_entry_order_id = None
        # Keep processed_order_ids bounded; clear on full reset to avoid growth
        self.processed_order_ids.clear()
        # Re-entry limit enforced via entries_today/max_attempts
        logger.debug(f"[{self.symbol}] [TID:{threading.get_ident()}] State reset after exit")

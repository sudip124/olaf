"""
Strategy-specific database models for strat80_20.

This module contains models specific to the strat80_20 strategy:
- BacktestRun: Metadata for backtest runs
- SetupDay: Setup detection results
- TradeLog: Trade events and signals
- LiveRun: Live trading session metadata
- LiveTradeLog: Live trading events

Note: Generic analysis models (BacktestJournal, KPIScore) are now in analysis/db_models.py
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()


class BacktestRun(Base):
    """Table to store backtest run metadata and configuration."""
    __tablename__ = 'backtest_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, nullable=False, default=datetime.now)
    from_date = Column(Date, nullable=False)
    to_date = Column(Date, nullable=False)
    scan_interval = Column(String(10), nullable=False)
    backtest_interval = Column(String(10), nullable=False)
    
    # Configuration parameters
    take_profit_mult = Column(Float)
    trigger_tick_mult = Column(Integer)
    trigger_window_minutes = Column(Integer)
    use_take_profit = Column(Boolean)
    
    # Optimization flag
    optimize = Column(Boolean, default=False)
    
    # Relationships
    setup_days = relationship('SetupDay', back_populates='backtest_run', cascade='all, delete-orphan')
    trade_logs = relationship('TradeLog', back_populates='backtest_run', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<BacktestRun(id={self.id}, timestamp={self.run_timestamp}, from={self.from_date}, to={self.to_date})>"


class SetupDay(Base):
    """Table to store individual setup days detected during backtest."""
    __tablename__ = 'setup_days'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id = Column(Integer, ForeignKey('backtest_runs.id'), nullable=False)
    
    # Setup day information
    symbol = Column(String(50), nullable=False)
    setup_date = Column(Date, nullable=False)
    weekday = Column(String(10), nullable=False)  # Monday, Tuesday, etc.
    
    # Trading parameters
    entry_price = Column(Float)
    trigger_price = Column(Float)
    tick_size = Column(Float)
    true_range = Column(Float)
    volume = Column(Integer)
    
    # Position metrics
    open_pos = Column(Float)
    close_pos = Column(Float)
    
    # Relationship back to backtest run
    backtest_run = relationship('BacktestRun', back_populates='setup_days')
    
    def __repr__(self):
        return f"<SetupDay(symbol={self.symbol}, date={self.setup_date}, weekday={self.weekday})>"


class TradeLog(Base):
    """Table to store all trade events and signals during backtest."""
    __tablename__ = 'trade_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id = Column(Integer, ForeignKey('backtest_runs.id'), nullable=False)
    
    # Event information
    timestamp = Column(DateTime, nullable=False)
    date = Column(Date, nullable=False)
    time = Column(String(15), nullable=False)
    weekday = Column(String(10), nullable=False)  # Monday, Tuesday, etc.
    
    # Event details
    event = Column(String(50), nullable=False)  # Setup Detected, Trigger Threshold Hit, Entry Filled, Stop Loss Exit, Take Profit Exit, Trailing SL Update
    symbol = Column(String(50), nullable=False)
    price = Column(Float)
    details = Column(String(500))  # Additional event details
    
    # Trade-specific fields (populated for entry/exit events)
    stop_loss = Column(Float)  # Current stop loss level
    take_profit = Column(Float)  # Take profit level
    risk = Column(Float)  # Risk amount (entry - stop_loss)
    
    # Relationship back to backtest run
    backtest_run = relationship('BacktestRun', back_populates='trade_logs')
    
    def __repr__(self):
        return f"<TradeLog(event={self.event}, symbol={self.symbol}, timestamp={self.timestamp})>"


class LiveRun(Base):
    """Table to store live trading session metadata."""
    __tablename__ = 'live_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, nullable=False, default=datetime.now)
    session_date = Column(Date, nullable=False)  # Trading session date
    
    # Trading parameters
    symbols = Column(String(500), nullable=False)  # Comma-separated list
    timezone = Column(String(50), nullable=False)
    fixed_qty = Column(Integer, nullable=False)
    product_type = Column(String(10), nullable=False)
    
    # Strategy parameters
    take_profit_mult = Column(Float)
    use_take_profit = Column(Boolean)
    trigger_window_minutes = Column(Integer)
    
    # Session status
    status = Column(String(20), default='active')  # active, completed, error
    
    # Relationships
    trade_logs = relationship('LiveTradeLog', back_populates='live_run', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<LiveRun(id={self.id}, date={self.session_date}, symbols={self.symbols})>"


class LiveTradeLog(Base):
    """Table to store all trade events during live trading."""
    __tablename__ = 'live_trade_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    live_run_id = Column(Integer, ForeignKey('live_runs.id'), nullable=False)
    
    # Event information
    timestamp = Column(DateTime, nullable=False)
    date = Column(Date, nullable=False)
    time = Column(String(15), nullable=False)
    
    # Event details
    event = Column(String(50), nullable=False)  # Setup Loaded, Trigger Threshold Hit, Entry Filled, Stop Loss Exit, Take Profit Exit, Trailing SL Update, etc.
    symbol = Column(String(50), nullable=False)
    price = Column(Float)
    details = Column(String(500))  # Additional event details
    
    # Trade-specific fields
    stop_loss = Column(Float)
    take_profit = Column(Float)
    order_id = Column(String(100))  # Broker order ID
    
    # Relationship back to live run
    live_run = relationship('LiveRun', back_populates='trade_logs')
    
    def __repr__(self):
        return f"<LiveTradeLog(event={self.event}, symbol={self.symbol}, timestamp={self.timestamp})>"


def get_db_engine(strategy_name='strat80_20'):
    """Create and return database engine for the strategy."""
    db_dir = os.path.join('strategies', strategy_name, 'data')
    os.makedirs(db_dir, exist_ok=True)
    
    db_path = os.path.join(db_dir, f'{strategy_name}.db')
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    return engine


def init_db(strategy_name='strat80_20'):
    """Initialize database tables."""
    engine = get_db_engine(strategy_name)
    Base.metadata.create_all(engine)
    return engine


def save_backtest_data(from_date, to_date, scan_interval, backtest_interval, 
                       config_params, setup_days_map, optimize=False, strategy_name='strat80_20'):
    """Save backtest run and setup days to database.
    
    Args:
        from_date: Start date string (YYYY-MM-DD)
        to_date: End date string (YYYY-MM-DD)
        scan_interval: Scan interval string
        backtest_interval: Backtest interval string
        config_params: Dictionary of configuration parameters
        setup_days_map: Dictionary mapping symbols to list of setup day dates or DataFrames
        optimize: Whether optimization was enabled
        strategy_name: Name of the strategy (default: 'strat80_20')
    
    Returns:
        backtest_run_id: ID of the created backtest run
    """
    import datetime as dt
    import pandas as pd
    
    engine = init_db(strategy_name)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create backtest run record
        backtest_run = BacktestRun(
            run_timestamp=datetime.now(),
            from_date=dt.datetime.fromisoformat(from_date).date(),
            to_date=dt.datetime.fromisoformat(to_date).date(),
            scan_interval=scan_interval,
            backtest_interval=backtest_interval,
            take_profit_mult=config_params.get('take_profit_mult'),
            trigger_tick_mult=config_params.get('trigger_tick_mult'),
            trigger_window_minutes=config_params.get('trigger_window_minutes'),
            use_take_profit=config_params.get('use_take_profit'),
            optimize=optimize
        )
        session.add(backtest_run)
        session.flush()  # Get the ID without committing
        
        # Create setup day records
        setup_count = 0
        for symbol, setup_data in setup_days_map.items():
            # Handle both list of dates and DataFrame formats
            if isinstance(setup_data, pd.DataFrame):
                # Skip empty DataFrames
                if setup_data.empty:
                    continue
                    
                # DataFrame with detailed setup information
                for idx, row in setup_data.iterrows():
                    try:
                        # Handle setup_date conversion
                        if isinstance(row['setup_date'], dt.date):
                            setup_date_val = row['setup_date']
                        elif isinstance(row['setup_date'], pd.Timestamp):
                            setup_date_val = row['setup_date'].date()
                        else:
                            setup_date_val = dt.datetime.fromisoformat(str(row['setup_date'])).date()
                        
                        setup_day = SetupDay(
                            backtest_run_id=backtest_run.id,
                            symbol=symbol,
                            setup_date=setup_date_val,
                            weekday=setup_date_val.strftime('%A'),
                            entry_price=float(row.get('entry_price')) if pd.notna(row.get('entry_price')) else None,
                            trigger_price=float(row.get('trigger_price')) if pd.notna(row.get('trigger_price')) else None,
                            tick_size=float(row.get('tick_size')) if pd.notna(row.get('tick_size')) else None,
                            true_range=float(row.get('true_range')) if pd.notna(row.get('true_range')) else None,
                            volume=int(row.get('volume')) if pd.notna(row.get('volume')) else None,
                            open_pos=float(row.get('open_pos')) if pd.notna(row.get('open_pos')) else None,
                            close_pos=float(row.get('close_pos')) if pd.notna(row.get('close_pos')) else None
                        )
                        session.add(setup_day)
                        setup_count += 1
                    except Exception as row_error:
                        print(f"[Database] Warning: Failed to save setup day for {symbol} at index {idx}: {row_error}")
                        continue
            elif isinstance(setup_data, list):
                # Skip empty lists
                if not setup_data:
                    continue
                    
                # List of date objects
                for setup_date in setup_data:
                    try:
                        if isinstance(setup_date, str):
                            setup_date_val = dt.datetime.fromisoformat(setup_date).date()
                        elif isinstance(setup_date, dt.date):
                            setup_date_val = setup_date
                        else:
                            setup_date_val = setup_date
                        
                        setup_day = SetupDay(
                            backtest_run_id=backtest_run.id,
                            symbol=symbol,
                            setup_date=setup_date_val,
                            weekday=setup_date_val.strftime('%A')
                        )
                        session.add(setup_day)
                        setup_count += 1
                    except Exception as date_error:
                        print(f"[Database] Warning: Failed to save setup date {setup_date} for {symbol}: {date_error}")
                        continue
        
        session.commit()
        print(f"\n[Database] Saved backtest run #{backtest_run.id} with {setup_count} setup days")
        return backtest_run.id
        
    except Exception as e:
        session.rollback()
        print(f"Error saving backtest data to database: {e}")
        raise
    finally:
        session.close()


def save_trade_logs(backtest_run_id, trade_logs, strategy_name='strat80_20'):
    """Save trade log events to database.
    
    Args:
        backtest_run_id: ID of the backtest run
        trade_logs: List of dictionaries containing trade log events
        strategy_name: Name of the strategy (default: 'strat80_20')
    
    Returns:
        Number of log entries saved
    """
    from datetime import datetime as dt
    import pandas as pd
    
    if not trade_logs:
        return 0
    
    engine = get_db_engine(strategy_name)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        count = 0
        for log_entry in trade_logs:
            # Parse timestamp
            timestamp = pd.to_datetime(log_entry['timestamp'])
            
            # Extract stop_loss, take_profit, risk from details if present
            details = log_entry.get('details', '')
            stop_loss = None
            take_profit = None
            risk = None
            
            # Parse details for specific values
            if 'SL:' in details:
                try:
                    sl_part = details.split('SL:')[1].split(';')[0].strip()
                    stop_loss = float(sl_part)
                except:
                    pass
            elif 'SL at' in details:
                try:
                    sl_part = details.split('SL at')[1].split()[0].strip()
                    stop_loss = float(sl_part)
                except:
                    pass
            
            if 'TP:' in details:
                try:
                    tp_part = details.split('TP:')[1].split('(')[0].strip()
                    take_profit = float(tp_part)
                except:
                    pass
            elif 'TP at' in details:
                try:
                    tp_part = details.split('TP at')[1].split()[0].strip()
                    take_profit = float(tp_part)
                except:
                    pass
            
            if 'risk:' in details:
                try:
                    risk_part = details.split('risk:')[1].split(';')[0].strip()
                    risk = float(risk_part)
                except:
                    pass
            
            # Handle time field - ensure it's a string for SQLite
            time_val = log_entry.get('time')
            if time_val is None:
                time_str = timestamp.time().isoformat()
            elif isinstance(time_val, str):
                time_str = time_val
            else:
                # Assume it's a datetime.time object
                time_str = time_val.isoformat()

            # Additional inference: for trailing SL updates, use price as stop_loss when not parsed
            event_name = log_entry.get('event', '')
            if event_name == 'Trailing SL Update' and stop_loss is None:
                try:
                    price_val = log_entry.get('price')
                    if price_val is not None:
                        stop_loss = float(price_val)
                except:
                    pass

            # Infer initial SL from risk and entry price for 'Entry Filled' when SL not parsed
            if event_name == 'Entry Filled' and stop_loss is None:
                try:
                    price_val = log_entry.get('price')
                    if price_val is not None and risk is not None:
                        stop_loss = float(price_val) - float(risk)
                except:
                    pass

            # Round parsed numeric values to 2 decimals for consistency
            if stop_loss is not None:
                try:
                    stop_loss = round(float(stop_loss), 2)
                except:
                    pass
            if take_profit is not None:
                try:
                    take_profit = round(float(take_profit), 2)
                except:
                    pass
            if risk is not None:
                try:
                    risk = round(float(risk), 2)
                except:
                    pass
            
            trade_log = TradeLog(
                backtest_run_id=backtest_run_id,
                timestamp=timestamp,
                date=timestamp.date(),
                time=time_str,
                weekday=timestamp.strftime('%A'),
                event=log_entry['event'],
                symbol=log_entry['symbol'],
                price=log_entry.get('price'),
                details=details,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk=risk
            )
            
            session.add(trade_log)
            count += 1
        
        session.commit()
        print(f"[Database] Saved {count} trade log entries for backtest run #{backtest_run_id}")
        return count
        
    except Exception as e:
        session.rollback()
        print(f"Error saving trade logs to database: {e}")
        raise
    finally:
        session.close()


def save_live_run(symbols, timezone, fixed_qty, product_type, take_profit_mult, 
                  use_take_profit, trigger_window_minutes, strategy_name='strat80_20'):
    """Save a live trading session to database.
    
    Args:
        symbols: List of symbols being traded
        timezone: Timezone for the session
        fixed_qty: Fixed quantity per trade
        product_type: Product type for orders
        take_profit_mult: Take profit multiplier
        use_take_profit: Whether to use take profit
        trigger_window_minutes: Trigger window in minutes
        strategy_name: Name of the strategy (default: 'strat80_20')
    
    Returns:
        live_run_id: ID of the created live run
    """
    import datetime as dt
    
    engine = init_db(strategy_name)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        live_run = LiveRun(
            run_timestamp=datetime.now(),
            session_date=dt.date.today(),
            symbols=','.join(symbols),
            timezone=timezone,
            fixed_qty=fixed_qty,
            product_type=product_type,
            take_profit_mult=take_profit_mult,
            use_take_profit=use_take_profit,
            trigger_window_minutes=trigger_window_minutes,
            status='active'
        )
        session.add(live_run)
        session.commit()
        
        print(f"\n[Database] Created live run #{live_run.id} for session {dt.date.today()}")
        return live_run.id
        
    except Exception as e:
        session.rollback()
        print(f"Error saving live run to database: {e}")
        raise
    finally:
        session.close()


def save_live_trade_log(live_run_id, event_dict, strategy_name='strat80_20'):
    """Save a single live trade event to database.
    
    Args:
        live_run_id: ID of the live run
        event_dict: Dictionary containing event details (timestamp, date, time, event, symbol, price, details)
        strategy_name: Name of the strategy (default: 'strat80_20')
    
    Returns:
        log_id: ID of the created log entry, or None if failed
    """
    import pandas as pd
    
    if not live_run_id:
        return None
    
    engine = get_db_engine(strategy_name)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Parse timestamp
        timestamp = pd.to_datetime(event_dict['timestamp'])
        
        # Extract stop_loss, take_profit, order_id from details if present
        details = event_dict.get('details', '')
        stop_loss = None
        take_profit = None
        order_id = None
        
        # Parse details for specific values
        if 'SL:' in details or 'SL at' in details:
            try:
                if 'SL:' in details:
                    sl_part = details.split('SL:')[1].split(',')[0].split(';')[0].strip()
                elif 'SL at' in details:
                    sl_part = details.split('SL at')[1].split()[0].strip()
                stop_loss = float(sl_part)
            except:
                pass
        
        if 'TP:' in details or 'TP at' in details:
            try:
                if 'TP:' in details:
                    tp_part = details.split('TP:')[1].split('(')[0].split(';')[0].strip()
                elif 'TP at' in details:
                    tp_part = details.split('TP at')[1].split()[0].strip()
                take_profit = float(tp_part)
            except:
                pass
        
        if 'order_id=' in details:
            try:
                order_id = details.split('order_id=')[1].split(';')[0].split()[0].strip()
            except:
                pass
        
        live_trade_log = LiveTradeLog(
            live_run_id=live_run_id,
            timestamp=timestamp,
            date=timestamp.date(),
            time=event_dict.get('time', timestamp.time().isoformat()),
            event=event_dict['event'],
            symbol=event_dict['symbol'],
            price=event_dict.get('price'),
            details=details,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=order_id
        )
        session.add(live_trade_log)
        session.commit()
        
        return live_trade_log.id
        
    except Exception as e:
        session.rollback()
        # Don't print error for every log entry to avoid spam
        return None
    finally:
        session.close()


def update_live_run_status(live_run_id, status, strategy_name='strat80_20'):
    """Update the status of a live run.
    
    Args:
        live_run_id: ID of the live run
        status: New status ('active', 'completed', 'error')
        strategy_name: Name of the strategy (default: 'strat80_20')
    """
    engine = get_db_engine(strategy_name)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        live_run = session.query(LiveRun).filter_by(id=live_run_id).first()
        if live_run:
            live_run.status = status
            session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error updating live run status: {e}")
    finally:
        session.close()

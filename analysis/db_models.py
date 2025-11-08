"""Generic database models for backtest journal and KPI analysis.

This module provides strategy-agnostic database models for storing:
- Backtest journal entries (trade-level data)
- KPI scores (aggregated metrics per symbol)

All strategies will use these tables in the centralized database (db_path from run_config).
"""
from sqlalchemy import Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from typing import Optional
from analysis.config import get_engine as _config_get_engine

Base = declarative_base()


class BacktestJournal(Base):
    """
    Generic backtest journal table for all strategies.
    
    Each row represents a single trade with entry/exit details and PnL metrics.
    The strategy_name column identifies which strategy generated this trade.
    """
    __tablename__ = 'backtest_journal'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Strategy and backtest identification
    strategy_name = Column(String(50), nullable=False, index=True)
    backtest_id = Column(Integer, nullable=False, index=True)
    
    # Trade timing
    date = Column(Date, nullable=False, index=True)
    time = Column(String(15), nullable=False)
    weekday = Column(String(10), nullable=False)
    signal_date = Column(Date)
    
    # Trade details
    symbol = Column(String(50), nullable=False, index=True)
    qty = Column(Integer, nullable=False, default=1)
    
    # Price and risk metrics
    initial_stoploss = Column(Float)
    final_stoploss = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    risk = Column(Float)  # 1R = entry_price - initial_stoploss
    
    # PnL metrics
    pnl_r = Column(Float)  # PnL in risk multiples (R)
    pnl_absolute = Column(Float)  # Absolute PnL
    
    # Trade outcome
    status = Column(String(10), index=True)  # win, lose, open
    exit_event = Column(String(50))  # Stop Loss Exit, Take Profit Exit, etc.
    
    def __repr__(self):
        return f"<BacktestJournal(strategy={self.strategy_name}, symbol={self.symbol}, date={self.date}, pnl_r={self.pnl_r})>"


class KPIScore(Base):
    """
    Generic KPI scores table for all strategies.
    
    Each row represents aggregated KPI metrics for a symbol in a specific backtest run.
    The strategy_name column identifies which strategy generated these metrics.
    """
    __tablename__ = 'kpi_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Strategy and backtest identification
    strategy_name = Column(String(50), nullable=False, index=True)
    backtest_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    
    # Composite score
    composite_score = Column(Float, index=True)
    
    # Setup and trade metrics
    total_setups = Column(Integer)
    total_trades = Column(Integer)
    success_rate = Column(Float)
    miss_rate = Column(Float)
    
    # Win/Loss metrics
    win_rate = Column(Float)
    win_loss_ratio = Column(Float)
    reentries = Column(Integer)
    
    # Risk-adjusted metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_dd_r = Column(Float)
    
    # Return metrics
    median_win_r = Column(Float)
    median_loss_r = Column(Float)
    avg_risk = Column(Float)
    
    # Temporal metrics
    top_weekday = Column(String(50))
    win_rate_last_30d = Column(Float)
    trades_last_30d = Column(Integer)
    
    def __repr__(self):
        return f"<KPIScore(strategy={self.strategy_name}, backtest_id={self.backtest_id}, symbol={self.symbol}, score={self.composite_score})>"


def get_analysis_engine(db_path: Optional[str] = None):
    """
    Create and return database engine for analysis.
    
    Args:
        db_path: Optional path to the database file. If None, resolves from analysis.config
        
    Returns:
        SQLAlchemy engine
    """
    return _config_get_engine(db_path)


def init_analysis_db(db_path: Optional[str] = None):
    """
    Initialize analysis database tables.
    
    Args:
        db_path: Optional path to the database file. If None, resolves from analysis.config
        
    Returns:
        SQLAlchemy engine
    """
    engine = get_analysis_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

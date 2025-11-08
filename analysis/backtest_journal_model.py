from sqlalchemy import Column, Integer, String, Float, Date


def create_backtest_journal_model(Base):
    """
    Factory function to create BacktestJournal model with the given Base.
    This allows the model to be strategy-agnostic.
    
    Args:
        Base: SQLAlchemy declarative base from the strategy's db_models
        
    Returns:
        BacktestJournal class bound to the provided Base
    """
    
    class BacktestJournal(Base):
        __tablename__ = 'backtest_journal'
        id = Column(Integer, primary_key=True, autoincrement=True)
        backtest_id = Column(Integer, nullable=False)
        date = Column(Date, nullable=False)
        time = Column(String(15), nullable=False)
        weekday = Column(String(10), nullable=False)
        signal_date = Column(Date)
        symbol = Column(String(50), nullable=False)
        qty = Column(Integer, nullable=False, default=1)
        initial_stoploss = Column(Float)
        final_stoploss = Column(Float)
        entry_price = Column(Float)
        exit_price = Column(Float)
        risk = Column(Float)  # 1R = entry_price - initial_stoploss
        pnl_r = Column(Float)  # PnL in risk multiples (R)
        pnl_absolute = Column(Float)  # Absolute PnL for reference
        status = Column(String(10))
        exit_event = Column(String(50))
        # trade_date ,entry_event, entry/exit time needed?
    
    return BacktestJournal

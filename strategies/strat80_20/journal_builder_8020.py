from sqlalchemy.orm import sessionmaker, Session as SessionType
from strategies.strat80_20 import strategy_db_models as db_models
from strategies.strat80_20.backtest_config import FEES
from analysis.backtest_journal_base import BacktestJournalBuilder
import pandas as pd


class JournalBuilder8020(BacktestJournalBuilder):
    """
    Builder class for creating backtest journal DataFrame from trade logs.
    
    This class processes trade logs and creates a structured journal DataFrame
    with calculated PnL and risk metrics. The DataFrame is returned to the caller
    for storage in the centralized analysis database.
    """
    
    def __init__(self, strategy_name: str = 'strat80_20'):
        """
        Initialize the JournalBuilder8020.
        
        Args:
            strategy_name: Name of the strategy (default: 'strat80_20')
        """
        super().__init__(strategy_name)
        self.db_models = db_models
        self.engine = db_models.get_db_engine(strategy_name)
        self.Session = sessionmaker(bind=self.engine)
    
    def build(self, backtest_run_id: int | None = None) -> pd.DataFrame:
        """
        Build the backtest journal DataFrame from trade logs.
        
        Args:
            backtest_run_id: ID of the backtest run to process. If None, uses the latest run.
            
        Returns:
            DataFrame with journal entries including setup metrics per symbol
        """
        session = self.Session()
        try:
            run_ids = self._get_run_ids(session, backtest_run_id)
            if not run_ids:
                return pd.DataFrame()
            
            all_journal_rows = []
            for rid in run_ids:
                journal_rows = self._process_run(session, rid)
                all_journal_rows.extend(journal_rows)
            
            # Convert to DataFrame
            if not all_journal_rows:
                return pd.DataFrame()
            
            journal_df = pd.DataFrame(all_journal_rows)
            
            # Enrich journal with setup metrics per symbol
            journal_df = self._add_setup_metrics(session, journal_df, run_ids)
            
            return journal_df
            
        except Exception:
            raise
        finally:
            session.close()
    
    def _get_run_ids(self, session: SessionType, backtest_run_id: int | None) -> list[int]:
        """Get the list of run IDs to process."""
        BacktestRun = self.db_models.BacktestRun
        run_ids = []
        if backtest_run_id is not None:
            run = session.query(BacktestRun).filter_by(id=backtest_run_id).first()
            if run:
                run_ids = [run.id]
        else:
            latest = session.query(BacktestRun).order_by(BacktestRun.id.desc()).first()
            if latest:
                run_ids = [latest.id]
        return run_ids
    
    def _process_run(self, session: SessionType, run_id: int) -> list:
        """Process a single backtest run and create journal rows."""
        TradeLog = self.db_models.TradeLog
        
        # Get all trade logs for this run
        logs = (
            session.query(TradeLog)
            .filter(TradeLog.backtest_run_id == run_id)
            .order_by(TradeLog.symbol.asc(), TradeLog.timestamp.asc())
            .all()
        )
        
        current = {}
        last_signal_date = {}
        journal_rows = []
        
        for log in logs:
            key = log.symbol
            
            # Track the most recent setup date per symbol
            if log.event == 'Setup Detected':
                try:
                    last_signal_date[key] = log.date
                except Exception:
                    last_signal_date[key] = None
                continue
            
            if log.event == 'Entry Filled':
                self._handle_entry(current, last_signal_date, log, run_id, key)
            elif log.event in ('Trailing SL Update',):
                self._handle_trailing_sl(current, log, key)
            elif log.event in ('Stop Loss Exit', 'Take Profit Exit'):
                completed_row = self._handle_exit(current, log, key)
                if completed_row:
                    journal_rows.append(completed_row)
        
        # Add any remaining open positions
        for key, vals in list(current.items()):
            journal_rows.append(vals)
        
        return journal_rows
    
    def _handle_entry(self, current: dict, last_signal_date: dict, log, run_id: int, key: str) -> None:
        """Handle entry filled event."""
        entry_price = log.price
        initial_sl = round(log.stop_loss, 2) if log.stop_loss is not None else None
        risk = round(entry_price - initial_sl, 2) if (entry_price and initial_sl) else None
        
        current[key] = {
            'date': log.date,
            'time': log.time,
            'weekday': log.weekday,
            'signal_date': last_signal_date.get(key),
            'symbol': log.symbol,
            'qty': 1,
            'initial_stoploss': initial_sl,
            'final_stoploss': initial_sl,
            'entry_price': entry_price,
            'exit_price': None,
            'risk': risk,
            'pnl_r': None,
            'pnl_absolute': None,
            'status': 'open',
            'exit_event': None,
        }
    
    def _handle_trailing_sl(self, current: dict, log, key: str) -> None:
        """Handle trailing stop loss update event."""
        if key in current and current[key]['status'] == 'open':
            if log.stop_loss is not None:
                current[key]['final_stoploss'] = round(log.stop_loss, 2)
    
    def _handle_exit(self, current: dict, log, key: str) -> dict | None:
        """Handle exit event (stop loss or take profit)."""
        if key not in current or current[key]['status'] != 'open':
            return None
        
        current[key]['exit_price'] = log.price
        current[key]['exit_event'] = log.event
        
        if current[key]['entry_price'] is not None and log.price is not None:
            qty = current[key]['qty']
            entry_price = current[key]['entry_price']
            exit_price = log.price
            risk = current[key]['risk']
            
            # Calculate absolute PnL (with fees)
            gross_pnl = (exit_price - entry_price) * qty
            fees_turnover = (entry_price + exit_price) * qty * FEES
            pnl_absolute = round(gross_pnl - fees_turnover, 2)
            current[key]['pnl_absolute'] = pnl_absolute
            
            # Calculate PnL in risk multiples (R) - including fees
            if risk and risk > 0:
                net_pnl_per_unit = (exit_price - entry_price) - (entry_price + exit_price) * FEES
                pnl_r = round(net_pnl_per_unit / risk, 2)
                current[key]['pnl_r'] = pnl_r
                current[key]['status'] = 'win' if pnl_r > 0 else 'lose'
            else:
                current[key]['pnl_r'] = None
                current[key]['status'] = 'lose'
        else:
            current[key]['status'] = 'lose'
        
        row = current[key].copy()
        del current[key]
        return row
    
    def _add_setup_metrics(self, session: SessionType, journal_df: pd.DataFrame, run_ids: list[int]) -> pd.DataFrame:
        """
        Add setup-related metrics to journal DataFrame by querying trade logs.
        Adds columns: total_setups, total_entries, total_missed per symbol.
        
        Args:
            session: Database session
            journal_df: Journal DataFrame
            run_ids: List of backtest run IDs
            
        Returns:
            Journal DataFrame with setup metrics added
        """
        if journal_df.empty:
            return journal_df
        
        TradeLog = self.db_models.TradeLog
        
        # Query trade logs for all runs
        trade_logs = (
            session.query(TradeLog)
            .filter(TradeLog.backtest_run_id.in_(run_ids))
            .order_by(TradeLog.symbol.asc(), TradeLog.timestamp.asc())
            .all()
        )
        
        # Calculate setup metrics per symbol
        setup_metrics = {}
        for symbol in journal_df['symbol'].unique():
            symbol_logs = [log for log in trade_logs if log.symbol == symbol]
            
            total_setups = len([log for log in symbol_logs if log.event == 'Setup Detected'])
            total_entries = len([log for log in symbol_logs if log.event == 'Entry Filled'])
            total_missed = len([log for log in symbol_logs if log.event and 'Entry Skipped' in log.event])
            
            setup_metrics[symbol] = {
                'total_setups': total_setups,
                'total_entries': total_entries,
                'total_missed': total_missed
            }
        
        # Add metrics to journal_df
        journal_df['total_setups'] = journal_df['symbol'].map(lambda s: setup_metrics.get(s, {}).get('total_setups', 0))
        journal_df['total_entries'] = journal_df['symbol'].map(lambda s: setup_metrics.get(s, {}).get('total_entries', 0))
        journal_df['total_missed'] = journal_df['symbol'].map(lambda s: setup_metrics.get(s, {}).get('total_missed', 0))
        
        return journal_df
    

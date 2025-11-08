"""Generic analysis service for storing journal data and calculating KPIs.

This service is strategy-agnostic and works with any trading strategy that provides
a journal dataframe with the required columns.
"""
from sqlalchemy import delete
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os

from analysis.db_models import BacktestJournal, KPIScore, init_analysis_db, get_analysis_engine


class AnalysisService:
    """
    Generic analysis service for storing and analyzing backtest results.
    
    This service handles:
    1. Storing journal dataframes in the centralized database
    2. Calculating KPI metrics from journal data
    3. Storing KPI scores in the database
    """
    
    def __init__(self, db_path: Optional[str] = None, strategy_name: str = ""):
        """
        Initialize the analysis service.
        
        Args:
            db_path: Optional path to the database file. If None, uses analysis.config
            strategy_name: Name of the strategy being analyzed
        """
        self.db_path = db_path
        self.strategy_name = strategy_name
        self.engine = init_analysis_db(db_path)
        self.Session = sessionmaker(bind=self.engine)
    
    def _validate_journal_df(self, journal_df: pd.DataFrame) -> None:
        """
        Validate that journal_df has all required columns.
        
        Args:
            journal_df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            'date', 'time', 'weekday', 'symbol', 'qty',
            'initial_stoploss', 'final_stoploss', 'entry_price', 'exit_price',
            'risk', 'pnl_r', 'pnl_absolute', 'status', 'exit_event',
            'total_setups', 'total_entries', 'total_missed'
        ]
        
        missing_columns = [col for col in required_columns if col not in journal_df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Journal DataFrame is missing required columns: {missing_columns}. "
                f"Ensure JournalBuilder includes all necessary data."
            )
    
    def store_journal(self, journal_df: pd.DataFrame, backtest_run_id: int) -> int:
        """
        Store journal dataframe in the centralized database.
        
        Args:
            journal_df: DataFrame with journal entries (must have required columns)
            backtest_run_id: ID of the backtest run
            
        Returns:
            Number of journal entries stored
        """
        if journal_df.empty:
            print(f"[Analysis] No journal entries to store")
            return 0
        
        # Validate journal_df has required columns
        self._validate_journal_df(journal_df)
        
        session = self.Session()
        
        try:
            # Delete existing journal entries for this strategy and backtest
            session.execute(
                delete(BacktestJournal).where(
                    BacktestJournal.strategy_name == self.strategy_name,
                    BacktestJournal.backtest_id == backtest_run_id
                )
            )
            
            # Insert new entries
            count = 0
            for _, row in journal_df.iterrows():
                journal_entry = BacktestJournal(
                    strategy_name=self.strategy_name,
                    backtest_id=backtest_run_id,
                    date=row['date'],
                    time=row['time'],
                    weekday=row['weekday'],
                    signal_date=row.get('signal_date'),
                    symbol=row['symbol'],
                    qty=int(row.get('qty', 1)),
                    initial_stoploss=float(row['initial_stoploss']) if pd.notna(row.get('initial_stoploss')) else None,
                    final_stoploss=float(row['final_stoploss']) if pd.notna(row.get('final_stoploss')) else None,
                    entry_price=float(row['entry_price']) if pd.notna(row.get('entry_price')) else None,
                    exit_price=float(row['exit_price']) if pd.notna(row.get('exit_price')) else None,
                    risk=float(row['risk']) if pd.notna(row.get('risk')) else None,
                    pnl_r=float(row['pnl_r']) if pd.notna(row.get('pnl_r')) else None,
                    pnl_absolute=float(row['pnl_absolute']) if pd.notna(row.get('pnl_absolute')) else None,
                    status=row.get('status', 'open'),
                    exit_event=row.get('exit_event')
                )
                session.add(journal_entry)
                count += 1
            
            session.commit()
            print(f"[Analysis] Stored {count} journal entries for {self.strategy_name} backtest #{backtest_run_id}")
            return count
            
        except Exception as e:
            session.rollback()
            print(f"[Analysis] Error storing journal: {e}")
            raise
        finally:
            session.close()
    
    def calculate_kpis(self, backtest_run_id: int) -> pd.DataFrame:
        """
        Calculate KPI metrics from journal data.
        
        Args:
            backtest_run_id: ID of the backtest run
            
        Returns:
            DataFrame with KPI metrics per symbol
        """
        session = self.Session()
        
        try:
            # Query journal entries for this backtest
            journals = (
                session.query(BacktestJournal)
                .filter(
                    BacktestJournal.strategy_name == self.strategy_name,
                    BacktestJournal.backtest_id == backtest_run_id,
                    BacktestJournal.status.in_(['win', 'lose'])
                )
                .all()
            )
            
            if not journals:
                print(f"[Analysis] No completed trades found for backtest #{backtest_run_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'symbol': j.symbol,
                'date': j.date,
                'weekday': j.weekday,
                'signal_date': j.signal_date,
                'pnl_r': j.pnl_r,
                'pnl_absolute': j.pnl_absolute,
                'risk': j.risk,
                'status': j.status,
                'entry_price': j.entry_price,
                'exit_price': j.exit_price,
            } for j in journals])
            
            # Calculate KPIs per symbol
            kpi_results = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                kpi_dict = self._calculate_symbol_kpis(symbol_df, symbol)
                kpi_results.append(kpi_dict)
            
            # Create DataFrame
            kpi_df = pd.DataFrame(kpi_results)
            
            if kpi_df.empty:
                return kpi_df
            
            # Compute composite scores
            kpi_df = self._compute_composite_scores(kpi_df)
            
            # Sort by composite score
            if 'composite_score' in kpi_df.columns:
                kpi_df = kpi_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
            else:
                kpi_df = kpi_df.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
            
            return kpi_df
            
        except Exception as e:
            print(f"[Analysis] Error calculating KPIs: {e}")
            raise
        finally:
            session.close()
    
    def _calculate_symbol_kpis(self, symbol_df: pd.DataFrame, symbol: str) -> dict:
        """Calculate KPI metrics for a single symbol."""
        # Basic metrics
        total_trades = len(symbol_df)
        winning_trades = symbol_df[symbol_df['status'] == 'win']
        losing_trades = symbol_df[symbol_df['status'] == 'lose']
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        # Median returns in R
        median_win_r = winning_trades['pnl_r'].median() if not winning_trades.empty else 0
        median_loss_r = losing_trades['pnl_r'].median() if not losing_trades.empty else 0
        
        # Average risk
        avg_risk = symbol_df['risk'].mean() if 'risk' in symbol_df.columns else 0
        
        # Sharpe Ratio
        returns = symbol_df['pnl_r'].dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty else 0
        sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # Top weekday
        if not winning_trades.empty:
            weekday_performance = winning_trades.groupby('weekday')['pnl_r'].agg(['sum', 'count'])
            if not weekday_performance.empty:
                top_weekday = weekday_performance['sum'].idxmax()
                top_weekday_wins = int(weekday_performance.loc[top_weekday, 'count'])
                top_weekday_pnl_r = weekday_performance.loc[top_weekday, 'sum']
            else:
                top_weekday, top_weekday_wins, top_weekday_pnl_r = 'N/A', 0, 0
        else:
            top_weekday, top_weekday_wins, top_weekday_pnl_r = 'N/A', 0, 0
        
        # Recent performance (last 30 days)
        if not symbol_df.empty:
            latest_date = symbol_df['date'].max()
            thirty_days_ago = latest_date - timedelta(days=30)
            recent_trades = symbol_df[symbol_df['date'] >= thirty_days_ago]
            recent_total = len(recent_trades)
            recent_wins = len(recent_trades[recent_trades['status'] == 'win'])
            recent_win_rate = (recent_wins / recent_total * 100) if recent_total > 0 else 0
        else:
            recent_win_rate, recent_total = 0, 0
        
        # Re-entry count
        if 'signal_date' in symbol_df.columns:
            trades_per_signal = symbol_df.groupby('signal_date').size()
            reentry_count = (trades_per_signal - 1).sum() if not trades_per_signal.empty else 0
        else:
            reentry_count = 0
        
        # Win/Loss Ratio
        win_loss_ratio = (win_count / loss_count) if loss_count > 0 else (win_count if win_count > 0 else 0)
        
        # Max Drawdown (in R)
        if not symbol_df.empty and 'pnl_r' in symbol_df.columns:
            symbol_df_sorted = symbol_df.sort_values('date')
            cumulative_pnl_r = symbol_df_sorted['pnl_r'].cumsum()
            running_max = cumulative_pnl_r.expanding().max()
            drawdown = cumulative_pnl_r - running_max
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        else:
            max_drawdown = 0
        
        # Get setup metrics from journal (should be same for all rows of this symbol)
        # Use first row's values since they're symbol-level metrics
        total_setups = int(symbol_df['total_setups'].iloc[0]) if 'total_setups' in symbol_df.columns else 0
        total_entries = int(symbol_df['total_entries'].iloc[0]) if 'total_entries' in symbol_df.columns else 0
        total_missed = int(symbol_df['total_missed'].iloc[0]) if 'total_missed' in symbol_df.columns else 0
        
        # Calculate success and miss rates
        success_rate = (total_entries / total_setups * 100) if total_setups > 0 else 0
        miss_rate = (total_missed / (total_entries + total_missed) * 100) if (total_entries + total_missed) > 0 else 0
        
        # Compile results
        return {
            'Symbol': symbol,
            'Total Setups': total_setups,
            'Total Trades': total_trades,
            'Success Rate (%)': round(success_rate, 2),
            'Miss Rate (%)': round(miss_rate, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Win/Loss Ratio': round(win_loss_ratio, 2),
            'Re-entries': int(reentry_count),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Sortino Ratio': round(sortino_ratio, 2),
            'Max DD (R)': round(max_drawdown, 2),
            'Median Win (R)': round(median_win_r, 2),
            'Median Loss (R)': round(median_loss_r, 2),
            'Avg Risk (₹)': round(avg_risk, 2),
            'Top Weekday': f"{top_weekday} ({top_weekday_wins}W, {round(top_weekday_pnl_r, 2)}R)",
            'Win Rate Last 30D (%)': round(recent_win_rate, 2),
            'Trades Last 30D': recent_total,
        }
    
    def _compute_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite scores based on normalized KPI metrics."""
        if df.empty:
            return df
        
        metrics = [
            'Total Trades', 'Total Setups', 'Win Rate (%)', 'Win/Loss Ratio',
            'Sharpe Ratio', 'Sortino Ratio', 'Max DD (R)',
            'Success Rate (%)', 'Miss Rate (%)'
        ]
        
        invert_metrics = ['Max DD (R)', 'Miss Rate (%)']
        
        # Normalize (min-max scaling)
        norm_df = pd.DataFrame(index=df.index)
        for metric in metrics:
            if metric in df.columns:
                vals = df[metric].replace([np.inf, -np.inf], np.nan).fillna(0)
                min_val, max_val = vals.min(), vals.max()
                if max_val > min_val:
                    norm = (vals - min_val) / (max_val - min_val)
                else:
                    norm = pd.Series(0, index=vals.index)
                if metric in invert_metrics:
                    norm = 1 - norm
                norm_df[metric] = norm
        
        # Weights
        weights = {
            'Total Trades': 0.10,
            'Total Setups': 0.05,
            'Win Rate (%)': 0.15,
            'Win/Loss Ratio': 0.15,
            'Sharpe Ratio': 0.15,
            'Sortino Ratio': 0.15,
            'Max DD (R)': 0.10,
            'Success Rate (%)': 0.10,
            'Miss Rate (%)': 0.05
        }
        
        # Compute weighted sum
        df['composite_score'] = norm_df.mul(pd.Series(weights), axis=1).sum(axis=1)
        df['composite_score'] = df['composite_score'].round(4)
        
        return df
    
    def store_kpis(self, kpi_df: pd.DataFrame, backtest_run_id: int) -> int:
        """
        Store KPI metrics in the database.
        
        Args:
            kpi_df: DataFrame with KPI metrics
            backtest_run_id: ID of the backtest run
            
        Returns:
            Number of KPI records stored
        """
        if kpi_df.empty:
            print(f"[Analysis] No KPI data to store")
            return 0
        
        session = self.Session()
        
        try:
            # Delete existing KPI scores for this strategy and backtest
            session.execute(
                delete(KPIScore).where(
                    KPIScore.strategy_name == self.strategy_name,
                    KPIScore.backtest_id == backtest_run_id
                )
            )
            
            # Insert new KPI scores
            count = 0
            for _, row in kpi_df.iterrows():
                kpi_score = KPIScore(
                    strategy_name=self.strategy_name,
                    backtest_id=backtest_run_id,
                    symbol=row.get('Symbol'),
                    composite_score=float(row.get('composite_score')) if pd.notna(row.get('composite_score')) else None,
                    total_setups=int(row.get('Total Setups', 0)),
                    total_trades=int(row.get('Total Trades', 0)),
                    success_rate=float(row.get('Success Rate (%)', 0)),
                    miss_rate=float(row.get('Miss Rate (%)', 0)),
                    win_rate=float(row.get('Win Rate (%)', 0)),
                    win_loss_ratio=float(row.get('Win/Loss Ratio', 0)),
                    reentries=int(row.get('Re-entries', 0)),
                    sharpe_ratio=float(row.get('Sharpe Ratio', 0)),
                    sortino_ratio=float(row.get('Sortino Ratio', 0)),
                    max_dd_r=float(row.get('Max DD (R)', 0)),
                    median_win_r=float(row.get('Median Win (R)', 0)),
                    median_loss_r=float(row.get('Median Loss (R)', 0)),
                    avg_risk=float(row.get('Avg Risk (₹)', 0)),
                    top_weekday=str(row.get('Top Weekday', 'N/A')),
                    win_rate_last_30d=float(row.get('Win Rate Last 30D (%)', 0)),
                    trades_last_30d=int(row.get('Trades Last 30D', 0))
                )
                session.add(kpi_score)
                count += 1
            
            session.commit()
            print(f"[Analysis] Stored {count} KPI scores for {self.strategy_name} backtest #{backtest_run_id}")
            return count
            
        except Exception as e:
            session.rollback()
            print(f"[Analysis] Error storing KPIs: {e}")
            raise
        finally:
            session.close()
    
    def save_kpi_csv(self, kpi_df: pd.DataFrame, backtest_run_id: int) -> Optional[str]:
        """
        Save KPI dataframe to CSV file.
        
        Args:
            kpi_df: DataFrame with KPI metrics
            backtest_run_id: ID of the backtest run
            
        Returns:
            Path to saved CSV file, or None if failed
        """
        try:
            os.makedirs('logs', exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join('logs', f'kpis_{self.strategy_name}_{backtest_run_id}_{ts}.csv')
            kpi_df.to_csv(csv_path, index=False)
            print(f"[Analysis] Saved KPI CSV to: {csv_path}")
            return csv_path
        except Exception as e:
            print(f"[Analysis] Warning: Failed to save KPI CSV: {e}")
            return None
    
    def analyze_and_store(self, journal_df: pd.DataFrame, backtest_run_id: int) -> pd.DataFrame:
        """
        Complete analysis workflow: store journal, calculate KPIs, store KPIs.
        
        Args:
            journal_df: DataFrame with journal entries (must include setup metrics)
            backtest_run_id: ID of the backtest run
            
        Returns:
            DataFrame with calculated KPIs
        """
        # Store journal
        self.store_journal(journal_df, backtest_run_id)
        
        # Calculate KPIs
        kpi_df = self.calculate_kpis(backtest_run_id)
        
        if not kpi_df.empty:
            # Store KPIs
            self.store_kpis(kpi_df, backtest_run_id)
            
            # Save CSV
            self.save_kpi_csv(kpi_df, backtest_run_id)
            
            # Print top symbols
            if 'composite_score' in kpi_df.columns:
                print("\n=== Top Symbols by Composite Score ===")
                display_cols = ['Symbol', 'composite_score', 'Sharpe Ratio', 'Win Rate (%)', 
                               'Success Rate (%)', 'Total Trades']
                available_cols = [col for col in display_cols if col in kpi_df.columns]
                print(kpi_df[available_cols].head(10).to_string(index=False))
        
        return kpi_df

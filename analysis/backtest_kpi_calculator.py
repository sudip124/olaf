from sqlalchemy import delete
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Any


class BacktestCalculator:
    """
    Generic calculator for backtest KPI metrics and composite scores.
    Handles KPI calculation, database persistence, and backfilling operations.
    Works with any strategy by accepting db_models and journal_model as parameters.
    """
    
    def __init__(self, db_models_module: Any, journal_model_class: Any, strategy_name: str = 'strat80_20'):
        """
        Initialize the BacktestCalculator.
        
        Args:
            db_models_module: The strategy's db_models module (must have Base, get_db_engine, KPIScore, TradeLog)
            journal_model_class: The BacktestJournal model class for this strategy
            strategy_name: Strategy name (default: 'strat80_20')
        """
        self.strategy_name = strategy_name
        self.db_models = db_models_module
        self.BacktestJournal = journal_model_class
        self.engine = db_models_module.get_db_engine(strategy_name)
        db_models_module.Base.metadata.create_all(self.engine)
    
    def save_kpi_scores_to_db(self, backtest_run_id: int, kpi_df: pd.DataFrame) -> int:
        """
        Save KPI scores and composite scores to the database.
        
        Args:
            backtest_run_id: ID of the backtest run
            kpi_df: DataFrame containing KPI metrics and composite scores
            
        Returns:
            Number of records saved
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        try:
            # Delete existing KPI scores for this backtest run
            KPIScore = self.db_models.KPIScore
            session.execute(delete(KPIScore).where(KPIScore.backtest_id == backtest_run_id))
            
            count = 0
            for _, row in kpi_df.iterrows():
                kpi_score = KPIScore(
                    backtest_id=backtest_run_id,
                    symbol=row.get('Symbol'),
                    composite_score=row.get('composite_score'),
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
            print(f"[Database] Saved {count} KPI score records for backtest run #{backtest_run_id}")
            return count
            
        except Exception as e:
            session.rollback()
            print(f"Error saving KPI scores to database: {e}")
            raise
        finally:
            session.close()
    
    def calculate_symbol_kpis(self, backtest_run_id: int) -> pd.DataFrame:
        """
        Calculate comprehensive KPIs for each symbol after journal creation, including composite scores.
        
        KPIs calculated:
        - Sharpe Ratio: Risk-adjusted return using all trades
        - Sortino Ratio: Risk-adjusted return using only downside deviation
        - Total Trades: Total number of completed trades
        - Win Rate: Percentage of winning trades
        - Median Win Return: Median PnL of winning trades
        - Median Loss Return: Median PnL of losing trades
        - Top Weekday: Best performing weekday for winning trades
        - Win/Loss Ratio (Last 30 Days): Win rate for trades in the last 30 days
        - Composite Score: Weighted score based on multiple KPIs
        
        Args:
            backtest_run_id: ID of the backtest run
            
        Returns:
            DataFrame with KPIs and composite scores per symbol
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        BacktestJournal = self.BacktestJournal
        TradeLog = self.db_models.TradeLog
        
        try:
            # Query all journal entries for this backtest
            journals = (
                session.query(BacktestJournal)
                .filter(BacktestJournal.backtest_id == backtest_run_id)
                .filter(BacktestJournal.status.in_(['win', 'lose']))
                .all()
            )
            
            if not journals:
                print(f"No completed trades found for backtest_id {backtest_run_id}")
                return pd.DataFrame()
            
            # Query TradeLog for setup-related metrics
            trade_logs = (
                session.query(TradeLog)
                .filter(TradeLog.backtest_run_id == backtest_run_id)
                .order_by(TradeLog.symbol.asc(), TradeLog.timestamp.asc())
                .all()
            )
            
            # Convert to DataFrame for easier processing
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
            
            # Group by symbol and calculate KPIs
            kpi_results = []
            
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                
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
                
                # Average risk (1R) in absolute terms
                avg_risk = symbol_df['risk'].mean() if 'risk' in symbol_df.columns else 0
                
                # Sharpe Ratio calculation
                # Using PnL in R as returns
                returns = symbol_df['pnl_r'].dropna()
                mean_return = returns.mean()
                std_return = returns.std()
                sharpe_ratio = (mean_return / std_return) if std_return > 0 else 0
                # Annualize assuming ~252 trading days
                sharpe_ratio_annual = sharpe_ratio * np.sqrt(252) if std_return > 0 else 0
                
                # Sortino Ratio calculation
                # Only use downside deviation (negative returns)
                downside_returns = returns[returns < 0]
                downside_std = downside_returns.std() if not downside_returns.empty else 0
                sortino_ratio = (mean_return / downside_std) if downside_std > 0 else 0
                # Annualize
                sortino_ratio_annual = sortino_ratio * np.sqrt(252) if downside_std > 0 else 0
                
                # Top performing weekday for winning trades
                if not winning_trades.empty:
                    weekday_performance = winning_trades.groupby('weekday')['pnl_r'].agg(['sum', 'count'])
                    if not weekday_performance.empty:
                        top_weekday = weekday_performance['sum'].idxmax()
                        top_weekday_wins = int(weekday_performance.loc[top_weekday, 'count'])
                        top_weekday_pnl_r = weekday_performance.loc[top_weekday, 'sum']
                    else:
                        top_weekday = 'N/A'
                        top_weekday_wins = 0
                        top_weekday_pnl_r = 0
                else:
                    top_weekday = 'N/A'
                    top_weekday_wins = 0
                    top_weekday_pnl_r = 0
                
                # Win/Loss ratio in last 30 days
                # Find the latest date in the backtest
                if not symbol_df.empty:
                    latest_date = symbol_df['date'].max()
                    thirty_days_ago = latest_date - timedelta(days=30)
                    recent_trades = symbol_df[symbol_df['date'] >= thirty_days_ago]
                    
                    recent_total = len(recent_trades)
                    recent_wins = len(recent_trades[recent_trades['status'] == 'win'])
                    recent_win_rate = (recent_wins / recent_total * 100) if recent_total > 0 else 0
                else:
                    recent_win_rate = 0
                    recent_total = 0
                
                # Calculate setup-related metrics from TradeLog
                symbol_logs = [log for log in trade_logs if log.symbol == symbol]
                total_setups = len([log for log in symbol_logs if log.event == 'Setup Detected'])
                total_entries = len([log for log in symbol_logs if log.event == 'Entry Filled'])
                total_missed = len([log for log in symbol_logs if log.event and 'Entry Skipped' in log.event])
                
                # Success rate = entries / setups (if setup led to at least one entry)
                success_rate = (total_entries / total_setups * 100) if total_setups > 0 else 0
                # Miss rate = missed opportunities / total opportunities
                miss_rate = (total_missed / (total_entries + total_missed) * 100) if (total_entries + total_missed) > 0 else 0
                
                # Re-entry count: count trades per signal_date, subtract 1 for initial entry
                if 'signal_date' in symbol_df.columns:
                    trades_per_signal = symbol_df.groupby('signal_date').size()
                    reentry_count = (trades_per_signal - 1).sum() if not trades_per_signal.empty else 0
                else:
                    reentry_count = 0
                
                # Win/Loss Ratio
                win_loss_ratio = (win_count / loss_count) if loss_count > 0 else (win_count if win_count > 0 else 0)
                
                # Max Drawdown calculation (in R)
                # Calculate cumulative PnL and track peak-to-trough drawdown
                if not symbol_df.empty and 'pnl_r' in symbol_df.columns:
                    symbol_df_sorted = symbol_df.sort_values('date')
                    cumulative_pnl_r = symbol_df_sorted['pnl_r'].cumsum()
                    running_max = cumulative_pnl_r.expanding().max()
                    drawdown = cumulative_pnl_r - running_max
                    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
                else:
                    max_drawdown = 0
                
                # Compile results
                kpi_results.append({
                    'Symbol': symbol,
                    'Total Setups': total_setups,
                    'Total Trades': total_trades,
                    'Success Rate (%)': round(success_rate, 2),
                    'Miss Rate (%)': round(miss_rate, 2),
                    'Win Rate (%)': round(win_rate, 2),
                    'Win/Loss Ratio': round(win_loss_ratio, 2),
                    'Re-entries': int(reentry_count),
                    'Sharpe Ratio': round(sharpe_ratio_annual, 2),
                    'Sortino Ratio': round(sortino_ratio_annual, 2),
                    'Max DD (R)': round(max_drawdown, 2),
                    'Median Win (R)': round(median_win_r, 2),
                    'Median Loss (R)': round(median_loss_r, 2),
                    'Avg Risk (₹)': round(avg_risk, 2),
                    'Top Weekday': f"{top_weekday} ({top_weekday_wins}W, {round(top_weekday_pnl_r, 2)}R)",
                    'Win Rate Last 30D (%)': round(recent_win_rate, 2),
                    'Trades Last 30D': recent_total,
                })
            
            # Create DataFrame
            kpi_df = pd.DataFrame(kpi_results)
            
            if kpi_df.empty:
                return kpi_df
            
            # Compute composite scores
            kpi_df = self._compute_composite_scores(kpi_df)
            
            # Sort by composite score (if available), otherwise by Sharpe Ratio
            if 'composite_score' in kpi_df.columns:
                kpi_df = kpi_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
            else:
                kpi_df = kpi_df.sort_values('Sharpe Ratio', ascending=False).reset_index(drop=True)
            
            return kpi_df
            
        except Exception as e:
            print(f"Error calculating KPIs: {e}")
            raise
        finally:
            session.close()
    
    def _compute_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite scores for symbols based on normalized KPI metrics.
        
        Args:
            df: DataFrame with KPI metrics
            
        Returns:
            DataFrame with composite_score column added
        """
        if df.empty:
            return df
        
        # For Miss Rate, it's already there as 'Miss Rate (%)'
        # For Max DD, use 'Max DD (R)'
        
        # Select metrics (align with available columns)
        metrics = [
            'Total Trades', 'Total Setups', 'Win Rate (%)', 'Win/Loss Ratio',
            'Sharpe Ratio', 'Sortino Ratio', 'Max DD (R)',
            'Success Rate (%)', 'Miss Rate (%)'  # Add 'exit_quality' if computable from journal
        ]
        
        # Compute exit_quality if possible (would require querying journal; for simplicity, skip or add query)
        # For now, assume not including 'exit_quality' and 'reentry_penalty'; adjust weights accordingly
        
        # "Lower is better" metrics
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
        
        # Weights (adjusted for available metrics; sum to 1.0)
        weights = {
            'Total Trades': 0.10,
            'Total Setups': 0.05,
            'Win Rate (%)': 0.15,
            'Win/Loss Ratio': 0.15,
            'Sharpe Ratio': 0.15,
            'Sortino Ratio': 0.15,
            'Max DD (R)': 0.10,  # Inverted
            'Success Rate (%)': 0.10,
            'Miss Rate (%)': 0.05   # Inverted
        }
        
        # Compute weighted sum
        df['composite_score'] = norm_df.mul(pd.Series(weights), axis=1).sum(axis=1)
        
        # Round composite score to 4 decimal places for readability
        df['composite_score'] = df['composite_score'].round(4)
        
        return df

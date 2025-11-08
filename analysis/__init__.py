"""
Generic analysis module for backtest journal creation and KPI calculation.

This module provides:
1. BacktestJournalBuilder: Abstract base class for strategy-specific journal builders
2. AnalysisService: Generic service for storing journal data and calculating KPIs
3. Database models: Generic BacktestJournal and KPIScore models for centralized storage

All strategies use the centralized database (db_path from run_config.json) for analysis data.
"""

from analysis.backtest_journal_base import BacktestJournalBuilder
from analysis.analysis_service import AnalysisService
from analysis.db_models import BacktestJournal, KPIScore, init_analysis_db, get_analysis_engine
from analysis.config import get_db_path, get_engine

# Legacy exports (deprecated, kept for backward compatibility)
from analysis.backtest_journal_model import create_backtest_journal_model
from analysis.backtest_kpi_calculator import BacktestCalculator

__all__ = [
    # New generic analysis components
    'BacktestJournalBuilder',
    'AnalysisService',
    'BacktestJournal',
    'KPIScore',
    'init_analysis_db',
    'get_analysis_engine',
    'get_db_path',
    'get_engine',
    
    # Legacy components (deprecated)
    'create_backtest_journal_model',
    'BacktestCalculator',
]

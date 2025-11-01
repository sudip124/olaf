from data_manager.config import OPENALGO_URL, EXCHANGE, API_KEY

BROKER = 'Zerodha'

# Backtesting Configuration
INIT_CASH = 1000000
FEES = 0.00076  # Approximately 0.076% per trade
SLIPAGE = 0.0005  # 0.05% slippage

# Trading Session Times (NSE)
SESSION_START = "09:15:00"
SESSION_END = "15:30:00"

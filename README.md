# README.md (minor update to reflect config changes)
Project Overview
This project provides a modular Python-based framework for backtesting intraday swing trading strategies. It is designed enable rapid prototyping, optimization, and validation of strategies on historical data before transitioning to live execution. 
Key features:

Backtesting Engine: Uses VectorBT for efficient, vectorized simulations and parameter optimizations (e.g., heatmaps for MA windows).
Data Integration: Leverages OpenAlgo for fetching historical OHLCV data from your broker (e.g., Zerodha via NSE).
Strategies: Modular files for trend following and mean reversion along with config files.


Frameworks and Libraries Used

VectorBT: A high-performance backtesting library for Python, ideal for intraday strategies. It handles signal-based portfolio simulations, metrics (e.g., Sharpe, drawdown), and optimizations without needing Pro features (we use free version with run_combs for param sweeps). GitHub: https://github.com/polakowo/vectorbt. Why used: Vectorized operations make it fast for testing trend/mean reversion on large datasets.
OpenAlgo: An open-source trade management platform for algo trading, deployed locally and integrated with brokers like Zerodha. We use its Python SDK for secure data fetching (historical quotes) and potential order placement. GitHub: https://github.com/marketcalls/openalgo. Why used: Seamless broker integration for NSE symbols, REST/WebSocket APIs for data/orders, and local deployment for privacy/low latency.
Other Dependencies:

pandas: Data manipulation (e.g., OHLCV DataFrames).
requests (optional fallback): For manual API calls, but SDK preferred.
python-dotenv: Securely load API keys from .env.
anywidget: For interactive VectorBT visualizations (e.g., heatmaps).
openalgo : OpenAlgo Python sdk.
ta-lib: Technical analysis library for calculating indicators like EMA, ATR, ADX, MACD in custom strategies.



Prerequisites

Python 3.8+.
OpenAlgo running locally (e.g., on http://127.0.0.1:5000) with broker integration.
Valid OpenAlgo API key (obtained after login to your local instance).

Installation

Clone or create the project directory:
textmkdir my_trading_system
cd olaf

Set up a virtual environment:
textpython -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate on Windows

Install dependencies from requirements.txt:
textpip install -r requirements.txt
Sample requirements.txt:
textvectorbt
pandas
openalgo
python-dotenv
anywidget  # For interactive plots
ta-lib     # For technical indicators in strategies

Create .env for secrets:
textOPENALGO_URL=http://127.0.0.1:5000
OPENALGO_API_KEY=your_api_key_here


Code Structure
The project follows a modular architecture to decouple data, strategies, and execution—key for maintaining and scaling intraday systems.
Directory Layout
textmy_trading_system/
├── venv/                   # Virtual environment (git-ignored)
├── .env                    # Secrets (git-ignored)
├── config.py               # Global settings (e.g., API keys, dates, session times)
├── data_fetcher.py         # Data layer: Fetches historical OHLCV via OpenAlgo SDK
├── strategies/             # Strategy layer: Signal generators
│   ├── __init__.py
│   ├── strat001.py         # Mean reversion using Keltner Channels, ADX, MACD, with risk management (stops, take profits, trailing stops)
│   └── strat001_config.json # JSON configuration file for strat001.py parameters (includes defaults and opt_ranges)
│   ├── strat002.py         # 80-20 reversal strat based on Linda Raschke book
│   └── strat001_config.json # JSON configuration file for strat002.py parameters
├── backtest.py             # Backtesting layer: Runs simulations and optimizations with VectorBT
├── requirements.txt        # Dependencies
└── logs/                   # Output (e.g., heatmaps, logs; git-ignored)
└── db/                     # Test data
File Descriptions

config.py: Central configuration. Loads env vars for OpenAlgo (URL, API key) and sets defaults for backtesting (dates, cash, fees, session times). Includes validation for required keys.
data_fetcher.py: Handles data retrieval. Uses OpenAlgo SDK to fetch pandas DataFrames of OHLCV. Function: fetch_historical_data(symbol, interval, from_date, to_date, exchange). Reorders columns for VectorBT compatibility.
strategies/strat002.py: Generates buy/sell signals for trend following. Uses VectorBT's MA for crossovers (e.g., fast MA > slow MA = entry). Tunable windows for intraday swings.
strategies/strat001.py: Uses ADA, MACD & Keltner for intraday strategies.
strategies/strat001_config.json: JSON file containing default parameter values and optimization ranges for strat001.py. Edit this file to change settings without modifying the code.
backtest.py: Core script for backtesting.

Fetches data.



Usage

Run Backtest:
textpython backtest.py
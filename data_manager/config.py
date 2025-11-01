import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# OpenAlgo connection
OPENALGO_URL = os.getenv('OPENALGO_URL', 'http://127.0.0.1:5000')
API_KEY = os.getenv('OPENALGO_API_KEY')
if not API_KEY:
    raise ValueError('OPENALGO_API_KEY not set in .env')

# Database path: read from environment if provided, else default to <repo>/db/nse.db
DB_PATH: Path = Path(os.getenv('DB_PATH', Path(__file__).resolve().parents[1] / 'db' / 'nse.db'))

# Behavior flags
SAVE_TO_LOGS = os.getenv('SAVE_TO_LOGS', 'true').lower() in ('1', 'true', 'yes')

# Exchange configuration
EXCHANGE = os.getenv('EXCHANGE', 'NSE')

def ensure_directories():
    """Ensure required folders (db/, logs/) exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path.cwd().joinpath('logs').mkdir(parents=True, exist_ok=True)
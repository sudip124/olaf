import os
import datetime
from dotenv import load_dotenv

load_dotenv()

OPENALGO_URL = os.getenv('OPENALGO_URL', 'http://127.0.0.1:5000')  # Use 127.0.0.1 for Windows local
API_KEY = os.getenv('OPENALGO_API_KEY')
if not API_KEY:
    raise ValueError("OPENALGO_API_KEY not set in .env")

BROKER = 'Zerodha'
# SYMBOLS = ['SIRCA', 'VMART']
SYMBOLS = [
    "ADANIPOWER", "SUZLON", "BEL", "EXIDEIND", "HINDZINC", "CROMPTON", "ALOKINDS", "MSUMI", "SJVN", "GMDCLTD",
    "REDINGTON", "ABLBL", "FIRSTCRY", "HAL", "WOCKPHARMA", "TARIL", "LTFOODS", "RKFORGE", "ASTERDM", "ATGL",
    "HAPPSTMNDS", "DIVISLAB", "COHANCE", "TATACHEM", "AADHARHFC", "MGL", "BLUESTARCO", "MANKIND", "LUPIN", "KEC",
    "SAILIFE", "KEI", "CLEAN", "AAVAS", "CHOLAHLDNG", "UTIAMC"
]
INTERVAL = '15m'  # Changed to '15m' for intraday data
DEFAULT_STRATEGY = 'strat80_20'  # Updated to match file naming (underscore instead of hyphen)

EXCHANGE = 'NSE'
INIT_CASH = 1000000
FEES = 0.00076  # Approximately 0.076% per trade
SLIPAGE = 0.0005  # 0.05% slippage
FROM_DATE = (datetime.date.today() - datetime.timedelta(days=200)).isoformat()
TO_DATE = datetime.date.today().isoformat()
SESSION_START = "09:15:00"
SESSION_END = "15:30:00" # NSE trading hours for retail investors

STRATEGIES = ['strat001', 'trend_following', 'mean_reversion', 'strat80_20']

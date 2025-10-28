from .strat001 import generate_signals as strat001_signals
from .trend_following import generate_signals as trend_signals
from .mean_reversion import generate_signals as mr_signals  # Legacy
from .strat80_20 import generate_signals as strat80_20

STRATEGY_REGISTRY = {
    'strat001': strat001_signals,
    'trend_following': trend_signals,
    'mean_reversion': mr_signals,
    'strat80_20': strat80_20,
    'strat80-20': strat80_20,  # Add hyphenated version for backwards compatibility
    # Add more as needed
}
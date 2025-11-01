from .strat01.strat001 import generate_signals as strat001_signals
from .strat80_20.strat80_20 import generate_signals as strat80_20

STRATEGY_REGISTRY = {
    'strat001': strat001_signals,
    'strat80_20': strat80_20,
    'strat80-20': strat80_20,  # Add hyphenated version for backwards compatibility
}
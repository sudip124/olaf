STRATEGY_REGISTRY = {}

# Optional import: strat80_20 (no external deps beyond pandas/numpy)
try:
    from .strat80_20.strat80_20 import generate_signals as strat80_20
    STRATEGY_REGISTRY['strat80_20'] = strat80_20
    STRATEGY_REGISTRY['strat80-20'] = strat80_20  # backwards compatibility
except Exception:
    pass

# Optional import: strat001 (may require talib). Only register if available.
try:
    from .strat01.strat001 import generate_signals as strat001_signals
    STRATEGY_REGISTRY['strat001'] = strat001_signals
except Exception:
    # talib or other dependency missing; ignore unless explicitly used
    pass
"""
Strategy loader utility for dynamically loading strategy classes.

This module provides functions to load backtest and live strategy classes
based on strategy name from run_config.json.
"""

import importlib
from typing import Optional
from strategies.base_backtest import BacktestStrategy
from strategies.base_live import LiveStrategy
from strategies.base_scanner import Scanner


# Strategy registry mapping strategy names to module paths and function/class names
STRATEGY_REGISTRY = {
    'strat80_20': {
        'backtest_module': 'strategies.strat80_20.backtest_8020',
        'backtest_class': 'Backtest8020',
        'backtest_function': 'run_backtest',  # Fallback to function if class not found
        'live_module': 'strategies.strat80_20.live_8020',
        'live_class': 'Live8020',
        'live_function': 'start_trading',  # Use function directly
        'scanner_module': 'strategies.strat80_20.scanner_8020',
        'scanner_class': 'Scanner8020',
    },
    # Add more strategies here as they are developed
    # 'strat_name': {
    #     'backtest_module': 'strategies.strat_name.backtest_module',
    #     'backtest_class': 'BacktestClassName',
    #     'live_module': 'strategies.strat_name.live_module',
    #     'live_class': 'LiveClassName',
    # },
}


def load_backtest_strategy(strategy_name: str) -> Optional[BacktestStrategy]:
    """
    Dynamically load a backtest strategy class by name.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'strat80_20')
    
    Returns:
        Instance of the backtest strategy class, or None if not found
    
    Raises:
        ValueError: If strategy is not registered
        ImportError: If module cannot be imported
        AttributeError: If class cannot be found in module
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{strategy_name}' not found in registry. Available strategies: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_config = STRATEGY_REGISTRY[strategy_name]
    module_path = strategy_config['backtest_module']
    class_name = strategy_config['backtest_class']
    
    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)
        
        # Get the class from the module
        strategy_class = getattr(module, class_name)
        
        # Instantiate and return
        return strategy_class()
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}")


def load_live_strategy(strategy_name: str) -> Optional[LiveStrategy]:
    """
    Dynamically load a live trading strategy class by name.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'strat80_20')
    
    Returns:
        Instance of the live strategy class, or None if not found
    
    Raises:
        ValueError: If strategy is not registered
        ImportError: If module cannot be imported
        AttributeError: If class cannot be found in module
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{strategy_name}' not found in registry. Available strategies: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_config = STRATEGY_REGISTRY[strategy_name]
    module_path = strategy_config['live_module']
    class_name = strategy_config['live_class']
    
    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)
        
        # Get the class from the module
        strategy_class = getattr(module, class_name)
        
        # Instantiate and return
        return strategy_class()
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}")


def get_available_strategies():
    """Get list of available strategy names."""
    return list(STRATEGY_REGISTRY.keys())


def load_scanner_strategy(strategy_name: str) -> Optional[Scanner]:
    """
    Dynamically load a scanner strategy class by name.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{strategy_name}' not found in registry. Available strategies: {list(STRATEGY_REGISTRY.keys())}")

    strategy_config = STRATEGY_REGISTRY[strategy_name]
    module_path = strategy_config.get('scanner_module')
    class_name = strategy_config.get('scanner_class')

    if not module_path or not class_name:
        raise AttributeError(f"Scanner not configured for strategy '{strategy_name}'")

    try:
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        return strategy_class()
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}")

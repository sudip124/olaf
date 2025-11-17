# Quantity Refactoring Summary

## Overview
Refactored the quantity setup for live trading in strat80_20 to move from a global `fixed_qty` parameter to per-symbol `qty` in each setup JSON.

## Changes Made

### 1. `utilities/filter_scanner_setups.py`
- **Added**: `DEFAULT_QTY = 5` constant (line 40)
- **Modified**: Added `qty` field to each filtered setup before updating run_config (lines 94-96)
- **Purpose**: When filtering scanner setups, automatically inject a default quantity for each symbol

### 2. `strategies/strat80_20/live_symbol_state.py`
- **Changed**: Renamed parameter `fixed_qty` → `qty` in `__init__` (line 20)
- **Changed**: Renamed instance variable `self.fixed_qty` → `self.qty` (line 38)
- **Updated**: All order placement calls now use `self.qty` instead of `self.fixed_qty` (lines 386, 470, 496, 574, 617)
- **Purpose**: SymbolState now uses per-symbol quantity instead of global fixed quantity

### 3. `strategies/strat80_20/live_8020.py`
- **Modified `load_setups` function**:
  - Removed `fixed_qty` parameter from function signature (line 303)
  - Added extraction of `qty` from each setup row: `qty = int(setup.get('qty', 1))` (line 312)
  - Pass `qty` to SymbolState instead of `fixed_qty` (line 323)
  
- **Modified `Live8020.start_trading` method**:
  - Removed `fixed_qty` parameter from method signature (line 467)
  - Updated database save call to use placeholder value (line 519)
  - Removed `fixed_qty` from `load_setups` call (line 532)

### 4. `run.py`
- **Removed**: `fixed_qty = int(live_cfg.get("fixed_qty", 1))` (previously line 133)
- **Updated**: Print statement no longer shows fixed_qty (line 177)
- **Updated**: Added `'qty'` to required columns validation (line 185)
- **Removed**: `fixed_qty` parameter from `strategy.start_trading()` call (line 208)

### 5. `run_config.json`
- **Removed**: `"fixed_qty": 5` from live config section (previously line 62)
- **Added**: `"qty": 5` to each setup object (lines 74, 84, 94, 104)

## Benefits

1. **Flexibility**: Each symbol can now have a different quantity based on:
   - Price per share
   - Volatility
   - Risk tolerance
   - Position sizing strategy

2. **Maintainability**: Quantity is now part of the setup data, making it easier to:
   - Track which quantity was used for each trade
   - Adjust quantities per symbol without changing global config
   - Use different quantities for different market conditions

3. **Consistency**: The `filter_scanner_setups.py` utility automatically adds the default quantity when filtering setups, ensuring all setups have the required field

## Usage

### Setting Default Quantity
Edit `utilities/filter_scanner_setups.py`:
```python
DEFAULT_QTY = 5  # Change this value as needed
```

### Per-Symbol Quantity Override
Manually edit `run_config.json` to set different quantities for specific symbols:
```json
{
  "symbol": "ABSLAMC",
  "entry_price": 731.15,
  "trigger_price": 730.65,
  "tick_size": 0.05,
  "true_range": 20.05,
  "qty": 10  // Override for this specific symbol
}
```

## Backward Compatibility
- If a setup is missing the `qty` field, the code defaults to `qty=1` (see `live_8020.py` line 312)
- This ensures the system won't break if old setup data is used

## Testing Recommendations
1. Run `python utilities/filter_scanner_setups.py` to verify qty is added to setups
2. Check that `run_config.json` has `qty` in all setup objects
3. Verify live trading uses the correct quantity per symbol
4. Test with different qty values for different symbols

## Date
November 17, 2025

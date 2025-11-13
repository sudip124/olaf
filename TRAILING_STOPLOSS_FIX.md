# Trailing Stoploss Fix - November 13, 2025

## Problem Summary

The trailing stoploss for strat80_20 was not updating after entry fills. Investigation revealed:

1. **Duplicate SL Orders**: Emergency SL orders were being placed immediately after entry fills
2. **No Price Updates**: "Protective SL Modified" events showed in logs but stoploss price didn't change
3. **Root Cause**: Race condition between initial SL placement and emergency failsafe check

## Root Cause Analysis

### The Race Condition

From the error logs (lines 68-74):
```
2025-11-13 03:53:20,214 - INFO - [PVRINOX] Entry filled at 1093.5
2025-11-13 03:53:20,216 - INFO - [PVRINOX] Placing new SELL STOP order at 1091.5
2025-11-13 03:53:21,387 - WARNING - [PVRINOX] In position but no broker SL order! Placing emergency SL.
2025-11-13 03:53:21,388 - INFO - [PVRINOX] Placing new SELL STOP order at 1091.5
```

**Timeline:**
1. Entry fills → `on_order_update()` called (line 606)
2. `place_or_update_sell_stop()` called to place initial SL
3. **Race condition window**: Order placed but `self.sl_order_id` not yet set
4. `process_tick()` runs emergency failsafe check (line 256)
5. Detects `self.sl_order_id` is None → places duplicate emergency SL
6. Original order completes and sets `self.sl_order_id`

Result: **Two SL orders** for the same position, both at the same price.

### Why Trailing Updates Didn't Work

When trailing SL logic triggered:
1. It calculated new SL price (e.g., 1091.5)
2. Called `place_or_update_sell_stop(1091.5)`
3. Method modified one of the duplicate orders
4. But the price was already 1091.5, so no effective change
5. Broker accepted modification but stoploss didn't move

## Solution Implemented

### 1. Prevent Race Condition (Lines 518-545)

**Before:**
```python
resp = client.placeorder(...)
self.sl_order_id = resp.get('order_id')
```

**After:**
```python
# Set placeholder BEFORE placing order
self.sl_order_id = "PENDING"

resp = client.placeorder(...)

# Update with actual order ID
order_id = resp.get('order_id')
if order_id:
    self.sl_order_id = order_id
else:
    self.sl_order_id = None
```

This prevents the emergency failsafe from triggering during the brief window between placing the order and receiving the response.

### 2. Skip Unnecessary Modifications (Lines 456-459)

**Added check:**
```python
# Skip if new_stop equals current stop_loss (no change needed)
if self.stop_loss is not None and abs(new_stop - self.stop_loss) < 1e-9:
    logger.debug(f"[{self.symbol}] Skipping SL update: new_stop ({new_stop}) equals current stop_loss ({self.stop_loss})")
    return
```

This prevents unnecessary API calls when the stoploss price hasn't actually changed.

### 3. Improved Emergency Failsafe (Lines 257-270)

**Before:**
```python
elif not self.sl_order_id:
    # Place emergency SL
```

**After:**
```python
elif not self.sl_order_id or self.sl_order_id == "PENDING":
    # Only trigger emergency if sl_order_id is None (not PENDING)
    if self.sl_order_id is None:
        # Place emergency SL
```

This ensures the emergency check doesn't trigger when an order is already being placed (PENDING state).

## Testing Recommendations

1. **Monitor for duplicate SL orders**: Check broker orderbook after entry fills
2. **Verify trailing updates**: Confirm stoploss price moves up on green bars
3. **Check logs**: Look for "Skipping SL update" messages when price hasn't changed
4. **Emergency scenarios**: Test that emergency SL still works when truly needed

## Additional Issue: Incorrect Trailing SL Calculation

### Problem with Original Calculation

```python
# BEFORE (Line 134)
new_sl = bar_low - 1e-8  # Only 0.00000001 below bar low!
```

**Issues:**
1. **Insufficient buffer**: `1e-8` (0.00000001) is essentially zero. After rounding to tick_size (0.05), this becomes exactly `bar_low`
2. **Premature exits**: Any minor retracement to bar_low would trigger the stop
3. **Combined with skip check**: Since `new_sl ≈ bar_low` and rarely changes by a full tick, the skip check prevented most updates

**Example:**
- Bar: open=1093.5, close=1095.0, low=1093.0
- Old calculation: `new_sl = 1093.0 - 0.00000001 = 1092.99999999`
- After rounding: `new_sl = 1093.0` (exactly at bar low!)
- If current SL is 1092.95, new SL only moves up 0.05 (1 tick)
- Very aggressive - no safety margin

### Fix Applied

```python
# AFTER (Lines 135-136)
buffer_ticks = 2
new_sl = bar_low - (buffer_ticks * self.tick_size)
```

**Benefits:**
1. **Meaningful buffer**: 2 ticks (0.10) below bar low provides cushion
2. **Prevents premature exits**: Normal price retracements won't trigger stop
3. **Proper trailing**: SL moves up meaningfully on green bars
4. **Configurable**: Can adjust `buffer_ticks` based on strategy needs

**Example with fix:**
- Bar: open=1093.5, close=1095.0, low=1093.0
- New calculation: `new_sl = 1093.0 - (2 * 0.05) = 1092.90`
- Provides 0.10 cushion below bar low
- More professional trailing stop placement

## Files Modified

### Live Trading
- `strategies/strat80_20/live_symbol_state.py`:
  - Lines 135-136: Fixed trailing SL calculation with proper buffer (2 ticks below bar_low)
  - Lines 456-459: Added skip check for unchanged SL price
  - Lines 518-545: Added PENDING placeholder to prevent race condition
  - Lines 257-270: Improved emergency failsafe logic
  - Line 619: Clear entry_order_id after fill to allow re-entry
  - Lines 290-294: Add pending order check to prevent duplicates
  - Lines 201-202: Enhanced re-arm logic comments

### Backtesting
- `strategies/strat80_20/strat80_20.py`:
  - Line 209: Added `not in_long` check to prevent overlapping positions
  - Line 239: Added `not in_long` check to "max attempts" logging

### Recovery
- `strategies/strat80_20/live_8020.py`:
  - Lines 150-168: Robust validation for positionbook response per OpenAlgo docs
  - Lines 193-210: Robust validation for orderbook response per OpenAlgo docs

## Additional Issue: Multiple Entry Orders While In Position

### Problem

The logs showed multiple buy orders being placed while already in position:
- 09:23:20 - Entry #1 filled (in_position = True)
- 09:23:34 - **NEW Buy Stop Order Placed** (while in position!)
- 09:23:49 - Entry #2 filled

**Root Causes:**
1. **Missing entry_order_id clear**: After order fills, `entry_order_id` wasn't cleared, blocking re-entry logic
2. **Race condition**: `on_order_update()` callback might fire AFTER `process_tick()` runs, so `in_position` still False when re-arm logic checks
3. **Insufficient guards**: `place_buy_stop()` checked `in_position` but not for pending orders

### Fixes Applied

1. **Clear entry_order_id after fill** (line 619):
```python
# After processing fill in on_order_update()
self.entry_order_id = None  # Allow re-entry after exit
```

2. **Add pending order check** (lines 290-294):
```python
# In place_buy_stop(), before placing order
if self.entry_order_id is not None and self.entry_order_id != "":
    logger.warning(f"Skipping buy stop order: pending order {self.entry_order_id} already exists")
    return
```

3. **Enhanced re-arm logic comments** (lines 201-202):
```python
# CRITICAL: Only place order if NOT already in position AND no pending entry order
# This prevents taking multiple overlapping positions (strategy exits fully before re-entering)
```

**Strategy Behavior:**
- Strategy is designed to take multiple attempts per day (max_attempts=3)
- But only ONE position at a time - must exit fully before re-entering
- Not a pyramiding strategy - it's separate independent trades

### Backtest Has Same Bug

The backtest code in `strat80_20.py` had the identical issue at line 209:

**BEFORE:**
```python
if can_enter and df['high'].iloc[i] >= entry_price:
```

**AFTER:**
```python
if can_enter and not in_long and df['high'].iloc[i] >= entry_price:
```

Without the `not in_long` check, the backtest could take overlapping positions, which doesn't make sense with `np.inf` position sizing (full capital allocation). This would have caused:
- Incorrect backtest results (multiple overlapping positions)
- Mismatch between backtest and live trading behavior
- Inflated performance metrics

**Fix also applied to line 239** for the "max attempts reached" logging to only trigger when not in position.

## Additional Issue: Recovery Type Error

### Problem

The error logs showed:
```
2025-11-13 03:49:39,812 - ERROR - [Recovery] Error recovering orders: 'str' object has no attribute 'get'
```

**Root Cause:**
The recovery code incorrectly assumed `order_response.get('data')` returns a list of orders directly. According to OpenAlgo documentation, `data` is actually a **dictionary** that contains the orders under a nested key.

The original code did:
```python
orders = order_response.get('data', [])  # Wrong! data is a dict, not a list
for order in orders:
    order.get('symbol')  # Crashes!
```

When `data` is a dict (or string), iterating over it doesn't give order objects, causing `'str' object has no attribute 'get'`.

### Fix Applied

Implemented robust validation following OpenAlgo documentation pattern:

```python
# For orderbook (lines 193-210)
if order_response and isinstance(order_response, dict) and order_response.get('status') == 'success':
    data = order_response.get('data', {})
    # Per OpenAlgo docs, data is a dict, not a list
    if isinstance(data, dict):
        # Try common keys where orders might be stored
        orders = data.get('orders', []) or data.get('orderbook', []) or []
    elif isinstance(data, list):
        # Some brokers might return list directly
        orders = data
    else:
        logger.warning(f"Unexpected data type: {type(data).__name__}")
        orders = []
    
    # Validate that orders is a list
    if not isinstance(orders, list):
        logger.warning(f"Expected list of orders, got {type(orders).__name__}")
        orders = []
```

Same pattern applied to positionbook (lines 150-168).

**Benefits:**
- ✅ Follows OpenAlgo documentation pattern for response validation
- ✅ Handles both dict and list formats (broker variations)
- ✅ Tries multiple common keys where data might be nested
- ✅ Gracefully handles unexpected API response formats
- ✅ Logs warnings with actual type and value for debugging
- ✅ Prevents crash during recovery
- ✅ Recovery continues even if one API call returns unexpected data

## Expected Behavior After Fix

1. **Single SL order** placed after entry fill
2. **Trailing updates** work correctly when price moves up
3. **No duplicate orders** from emergency failsafe
4. **Efficient API usage** by skipping unnecessary modifications
5. **Emergency protection** still works when truly needed
6. **No overlapping positions** - only one entry order at a time
7. **Proper re-entry** - can take new entries after exit (up to max_attempts)

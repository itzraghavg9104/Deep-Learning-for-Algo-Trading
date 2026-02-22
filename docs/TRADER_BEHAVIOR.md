# Trader Behavior Integration

This document explains how trader behavior is modeled and where it plugs into the system.

## Risk Profiler

File: `backend/app/trader_behavior/risk_profiler.py`

Why it exists

- Converts subjective risk answers into a numeric score.
- Standardizes risk categories for consistent UI and decision logic.

Key functions

- `calculate_risk_score(answers)`
  - Input: list of integers 1 to 4.
  - Output: float between 0.0 and 1.0.
- `get_risk_category(risk_tolerance)`
  - Input: risk tolerance score.
  - Output: category and description.
- `get_position_size_multiplier(risk_tolerance)`
  - Output: multiplier from 0.5 to 1.5.
- `get_stop_loss_percentage(risk_tolerance)`
  - Output: stop loss percentage from 5% to 15%.
- `get_take_profit_percentage(risk_tolerance)`
  - Output: take profit percentage from 10% to 30%.

Integration points

- `POST /api/v1/profile/risk-assessment` uses these functions to compute score and recommendations.
- The frontend displays the score and category on the profile page.

## Position Sizer

File: `backend/app/trader_behavior/position_sizer.py`

Why it exists

- Converts risk tolerance and confidence into position sizing decisions.

Integration points

- Planned for live trading and recommendation sizing.
- Can be added to signal generation to scale trade size.

## Breakeven Tracker

File: `backend/app/trader_behavior/breakeven_tracker.py`

Why it exists

- Tracks breakeven levels for open positions.

Integration points

- Planned for trade management UI and live portfolio tracking.

## Current Limitations

- Position sizing and breakeven tracking are not yet wired into live signals.
- Risk tolerance currently influences backtest environment only when provided by user input.

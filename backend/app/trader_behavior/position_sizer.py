"""
Position sizing algorithms based on risk tolerance.

Implements multiple sizing strategies including Kelly Criterion.
"""
from typing import Optional
import numpy as np


def fixed_percentage_size(
    portfolio_value: float,
    risk_tolerance: float,
    base_pct: float = 0.10
) -> float:
    """
    Calculate position size as fixed percentage of portfolio.
    
    Args:
        portfolio_value: Total portfolio value
        risk_tolerance: Risk score (0.0 to 1.0)
        base_pct: Base percentage (default 10%)
    
    Returns:
        Position size in currency units
    """
    # Conservative: 5%, Aggressive: 20%
    adjusted_pct = base_pct * (0.5 + risk_tolerance)
    return portfolio_value * adjusted_pct


def kelly_criterion_size(
    portfolio_value: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_tolerance: float = 0.5
) -> float:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Kelly Formula: f* = (bp - q) / b
    where:
        b = odds (avg_win / avg_loss)
        p = probability of winning
        q = probability of losing (1 - p)
    
    Args:
        portfolio_value: Total portfolio value
        win_rate: Historical win rate (0.0 to 1.0)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)
        risk_tolerance: Risk score for adjustment
    
    Returns:
        Position size in currency units
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    b = avg_win / abs(avg_loss)  # Odds
    p = win_rate
    q = 1 - p
    
    # Kelly fraction
    kelly_fraction = (b * p - q) / b
    
    # Clamp to valid range
    kelly_fraction = max(0, min(kelly_fraction, 1))
    
    # Apply risk tolerance modifier
    # Conservative traders use 25% Kelly, Aggressive use full Kelly
    adjustment = 0.25 + (0.75 * risk_tolerance)
    adjusted_fraction = kelly_fraction * adjustment
    
    # Cap at 25% of portfolio
    max_fraction = 0.25
    final_fraction = min(adjusted_fraction, max_fraction)
    
    return portfolio_value * final_fraction


def volatility_adjusted_size(
    portfolio_value: float,
    atr: float,
    current_price: float,
    risk_tolerance: float,
    risk_per_trade: float = 0.02
) -> float:
    """
    Calculate position size adjusted for market volatility.
    
    Higher volatility = smaller position size.
    
    Args:
        portfolio_value: Total portfolio value
        atr: Average True Range (14-period)
        current_price: Current stock price
        risk_tolerance: Risk score (0.0 to 1.0)
        risk_per_trade: Base risk per trade (default 2%)
    
    Returns:
        Position size in shares
    """
    if atr == 0 or current_price == 0:
        return 0.0
    
    # Risk amount based on portfolio and tolerance
    adjusted_risk = risk_per_trade * (0.5 + risk_tolerance)
    risk_amount = portfolio_value * adjusted_risk
    
    # Position size based on ATR as stop-loss reference
    # Assuming stop-loss at 2x ATR
    stop_distance = 2 * atr
    position_size_shares = risk_amount / stop_distance
    
    # Convert to value
    position_value = position_size_shares * current_price
    
    # Cap at 25% of portfolio
    max_value = portfolio_value * 0.25
    
    return min(position_value, max_value)


def calculate_position_size(
    portfolio_value: float,
    current_price: float,
    risk_tolerance: float,
    atr: Optional[float] = None,
    win_rate: Optional[float] = None,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    method: str = "fixed"
) -> dict:
    """
    Calculate position size using specified method.
    
    Args:
        portfolio_value: Total portfolio value
        current_price: Current stock price
        risk_tolerance: Risk score (0.0 to 1.0)
        atr: ATR for volatility adjustment
        win_rate: Historical win rate for Kelly
        avg_win: Average win for Kelly
        avg_loss: Average loss for Kelly
        method: Sizing method ("fixed", "kelly", "volatility")
    
    Returns:
        Dictionary with position size details
    """
    if method == "kelly" and all([win_rate, avg_win, avg_loss]):
        size_value = kelly_criterion_size(
            portfolio_value, win_rate, avg_win, avg_loss, risk_tolerance
        )
    elif method == "volatility" and atr:
        size_value = volatility_adjusted_size(
            portfolio_value, atr, current_price, risk_tolerance
        )
    else:
        size_value = fixed_percentage_size(portfolio_value, risk_tolerance)
    
    # Calculate shares
    shares = int(size_value / current_price) if current_price > 0 else 0
    actual_value = shares * current_price
    
    return {
        "method": method,
        "position_value": round(actual_value, 2),
        "shares": shares,
        "portfolio_percentage": round((actual_value / portfolio_value) * 100, 2),
        "risk_tolerance_used": risk_tolerance,
    }

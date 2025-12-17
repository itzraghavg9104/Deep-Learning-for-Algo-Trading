"""
Sharpe Ratio based reward function for the DRL agent.

Optimizes for risk-adjusted returns.
"""
import numpy as np
from typing import List, Optional


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02 / 252  # Daily risk-free rate
) -> float:
    """
    Calculate Sharpe Ratio from returns.
    
    Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(252)
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Daily risk-free rate (default ~8% annual in India)
    
    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = mean_return / std_return
    annualized_sharpe = sharpe * np.sqrt(252)
    
    return float(annualized_sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02 / 252
) -> float:
    """
    Calculate Sortino Ratio (only penalizes downside volatility).
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Daily risk-free rate
    
    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    
    # Only consider negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 5.0  # Cap at high positive value
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 5.0
    
    sortino = mean_return / downside_std
    annualized_sortino = sortino * np.sqrt(252)
    
    return float(np.clip(annualized_sortino, -5, 5))


def calculate_reward(
    portfolio_values: List[float],
    window_size: int = 20,
    reward_type: str = "sharpe"
) -> float:
    """
    Calculate reward based on portfolio performance.
    
    Args:
        portfolio_values: List of portfolio values over time
        window_size: Rolling window for calculation
        reward_type: "sharpe", "sortino", or "returns"
    
    Returns:
        Reward value (scaled to reasonable range)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # Calculate returns
    values = np.array(portfolio_values[-window_size:])
    returns = np.diff(values) / values[:-1]
    
    if len(returns) == 0:
        return 0.0
    
    if reward_type == "sharpe":
        reward = calculate_sharpe_ratio(returns)
    elif reward_type == "sortino":
        reward = calculate_sortino_ratio(returns)
    else:  # Simple returns
        reward = np.sum(returns) * 100
    
    # Scale reward to reasonable range [-1, 1]
    scaled_reward = np.clip(reward / 2.0, -1, 1)
    
    return float(scaled_reward)


def calculate_step_reward(
    action: int,
    price_change_pct: float,
    position: float,
    risk_tolerance: float,
    transaction_cost: float = 0.001
) -> float:
    """
    Calculate immediate step reward.
    
    Combines:
    - P&L from price movement
    - Transaction cost penalty
    - Risk-adjusted component
    
    Args:
        action: 0=HOLD, 1=BUY, 2=SELL
        price_change_pct: Price change as percentage
        position: Current position (normalized)
        risk_tolerance: User's risk tolerance
        transaction_cost: Transaction cost as fraction
    
    Returns:
        Step reward
    """
    reward = 0.0
    
    # P&L component
    if position > 0:
        reward += position * price_change_pct / 100
    
    # Transaction cost for trades
    if action != 0:  # BUY or SELL
        reward -= transaction_cost
    
    # Risk adjustment (penalize large positions for conservative traders)
    risk_penalty = abs(position) * (1 - risk_tolerance) * 0.001
    reward -= risk_penalty
    
    return float(reward)


class RewardTracker:
    """
    Tracks rewards and portfolio performance over an episode.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.portfolio_values: List[float] = [initial_capital]
        self.returns: List[float] = []
        self.trades: List[dict] = []
    
    def update(self, portfolio_value: float, trade: Optional[dict] = None):
        """Update with new portfolio value."""
        if len(self.portfolio_values) > 0:
            prev_value = self.portfolio_values[-1]
            ret = (portfolio_value - prev_value) / prev_value
            self.returns.append(ret)
        
        self.portfolio_values.append(portfolio_value)
        
        if trade:
            self.trades.append(trade)
    
    def get_episode_reward(self) -> float:
        """Get total episode reward (Sharpe Ratio based)."""
        return calculate_sharpe_ratio(np.array(self.returns))
    
    def get_metrics(self) -> dict:
        """Get performance metrics for the episode."""
        if len(self.portfolio_values) < 2:
            return {
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
            }
        
        # Total return
        total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe ratio
        sharpe = calculate_sharpe_ratio(np.array(self.returns))
        
        # Max drawdown
        peak = self.portfolio_values[0]
        max_dd = 0
        for val in self.portfolio_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        
        # Win rate from trades
        if self.trades:
            profitable = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            win_rate = profitable / len(self.trades) * 100
        else:
            win_rate = 0
        
        return {
            "total_return": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(self.trades),
            "final_value": round(self.portfolio_values[-1], 2),
        }
    
    def reset(self):
        """Reset tracker for new episode."""
        self.portfolio_values = [self.initial_capital]
        self.returns = []
        self.trades = []

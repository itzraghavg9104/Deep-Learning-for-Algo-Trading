"""
Custom Trading Environment for Gymnasium/OpenAI Gym.

Simulates stock trading with the PPO agent.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

from app.layer1_data_processing.technical_indicators import compute_indicators
from app.layer2_decision.reward_function import calculate_step_reward, RewardTracker
from app.layer2_decision.action_space import (
    ACTION_BUY,
    ACTION_HOLD_BUY,
    ACTION_HOLD_SELL,
    ACTION_IDLE,
    ACTION_SELL,
)


class TradingEnv(gym.Env):
    """
    Stock Trading Environment for Reinforcement Learning.
    
    Actions:
        0: HOLD BUY
        1: HOLD SELL
        2: BUY
        3: SELL
        4: IDLE
    
    Observation:
        State vector from Layer 1 (technical indicators, price data, etc.)
    
    Reward:
        Risk-adjusted returns (Sharpe Ratio based)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_shares: int = 100,
        transaction_cost: float = 0.001,
        risk_tolerance: float = 0.5,
        window_size: int = 20,
        behavior_array: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            initial_capital: Starting capital
            max_shares: Maximum shares per trade
            transaction_cost: Transaction cost as fraction
            risk_tolerance: Trader's risk tolerance (0-1)
            window_size: Lookback window for indicators
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.max_shares = max_shares
        self.transaction_cost = transaction_cost
        self.risk_tolerance = risk_tolerance
        self.window_size = window_size
        self.behavior_array = behavior_array or {}
        
        # Action space: HOLD BUY, HOLD SELL, BUY, SELL, IDLE
        self.action_space = spaces.Discrete(5)
        
        # Observation space: state vector dimension
        # Approximate size based on indicators + portfolio state
        self.state_dim = 34
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Start after window_size to have enough history
        self.current_step = self.window_size
        
        # Portfolio state
        self.cash = self.initial_capital
        self.shares = 0
        self.avg_entry_price = 0.0
        
        # Tracking
        self.reward_tracker = RewardTracker(self.initial_capital)
        self.trades = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: see action_space.py
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self.df.loc[self.current_step, 'Close']
        prev_price = self.df.loc[self.current_step - 1, 'Close']
        price_change_pct = (current_price - prev_price) / prev_price * 100
        
        # Execute action
        self._execute_action(action, current_price)
        
        # Calculate portfolio value
        portfolio_value = self.cash + self.shares * current_price
        self.reward_tracker.update(portfolio_value)
        
        # Calculate reward
        position_normalized = (self.shares * current_price) / portfolio_value if portfolio_value > 0 else 0
        reward = calculate_step_reward(
            action=action,
            price_change_pct=price_change_pct,
            position=position_normalized,
            risk_tolerance=self.risk_tolerance,
            transaction_cost=self.transaction_cost
        )
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = portfolio_value < self.initial_capital * 0.5  # Stop if 50% loss
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, current_price: float):
        """Execute trading action."""
        if action == ACTION_HOLD_BUY:
            # Keep existing long bias; if flat, open a small position.
            if self.shares == 0:
                self._buy_shares(current_price, max(1, self.max_shares // 4))
        elif action == ACTION_HOLD_SELL:
            # Keep existing short/exit bias; if long, reduce partially.
            if self.shares > 0:
                shares_to_sell = max(1, self.shares // 2)
                self._sell_shares(current_price, shares_to_sell)
        elif action == ACTION_BUY:
            # Calculate shares to buy
            available_cash = self.cash * (1 - self.transaction_cost)
            shares_to_buy = min(
                int(available_cash / current_price),
                self.max_shares
            )
            self._buy_shares(current_price, shares_to_buy)
        elif action == ACTION_SELL:
            if self.shares > 0:
                self._sell_shares(current_price, self.shares)
        # ACTION_IDLE does nothing

    def _buy_shares(self, current_price: float, shares_to_buy: int):
        if shares_to_buy <= 0:
            return
        cost = shares_to_buy * current_price * (1 + self.transaction_cost)
        if cost > self.cash:
            return
        self.cash -= cost

        if self.shares > 0:
            total_cost = self.avg_entry_price * self.shares + current_price * shares_to_buy
            self.shares += shares_to_buy
            self.avg_entry_price = total_cost / self.shares
        else:
            self.shares = shares_to_buy
            self.avg_entry_price = current_price

        self.trades.append({
            "step": self.current_step,
            "action": "BUY",
            "price": current_price,
            "shares": shares_to_buy,
        })

    def _sell_shares(self, current_price: float, shares_to_sell: int):
        if shares_to_sell <= 0 or self.shares <= 0:
            return
        shares_to_sell = min(shares_to_sell, self.shares)
        proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
        pnl = proceeds - (shares_to_sell * self.avg_entry_price)

        self.trades.append({
            "step": self.current_step,
            "action": "SELL",
            "price": current_price,
            "shares": shares_to_sell,
            "pnl": pnl,
        })

        self.cash += proceeds
        self.shares -= shares_to_sell
        if self.shares == 0:
            self.avg_entry_price = 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state vector)."""
        # Get window of data for indicators
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step + 1
        window_df = self.df.iloc[start_idx:end_idx].copy()
        
        # Compute indicators
        indicators = compute_indicators(window_df)
        
        # Build state vector
        current_price = self.df.loc[self.current_step, 'Close']
        portfolio_value = self.cash + self.shares * current_price
        
        state = []
        
        # Price features
        state.append(current_price / 1000)  # Normalize
        
        # Key indicators
        indicator_keys = [
            'rsi_14', 'macd_line', 'macd_signal', 'bb_pct_b',
            'atr_pct', 'adx', 'stoch_k', 'cci_20', 'volume_ratio'
        ]
        for key in indicator_keys:
            val = indicators.get(key, 0)
            if isinstance(val, (int, float)):
                state.append(float(val) / 100 if 'pct' not in key else float(val))
            else:
                state.append(0.0)
        
        # Portfolio state
        state.append(self.cash / self.initial_capital)
        state.append(self.shares * current_price / portfolio_value if portfolio_value > 0 else 0)
        state.append(self.risk_tolerance)
        state.append(float(self.behavior_array.get("capital_per_trade_pct", 0.1)))
        state.append(float(self.behavior_array.get("tp_sl_ratio_preference", 0.4)))
        state.append(float(self.behavior_array.get("drawdown_sensitivity", 0.2)))
        state.append(float(self.behavior_array.get("post_loss_rest_min", 0.1)))
        
        # P&L state
        if self.shares > 0 and self.avg_entry_price > 0:
            pnl_pct = (current_price - self.avg_entry_price) / self.avg_entry_price
            state.append(pnl_pct)
        else:
            state.append(0.0)
        
        # Pad to fixed size
        while len(state) < self.state_dim:
            state.append(0.0)
        
        return np.array(state[:self.state_dim], dtype=np.float32)
    
    def _get_info(self) -> dict:
        """Get additional info."""
        current_price = self.df.loc[self.current_step, 'Close']
        portfolio_value = self.cash + self.shares * current_price
        
        return {
            'step': self.current_step,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'shares': self.shares,
            'current_price': current_price,
            'total_trades': len(self.trades),
        }
    
    def render(self, mode='human'):
        """Render the environment state."""
        info = self._get_info()
        print(f"Step {info['step']}: Portfolio=${info['portfolio_value']:.2f}, "
              f"Cash=${info['cash']:.2f}, Shares={info['shares']}, "
              f"Price=${info['current_price']:.2f}")
    
    def get_episode_metrics(self) -> dict:
        """Get performance metrics for the episode."""
        return self.reward_tracker.get_metrics()

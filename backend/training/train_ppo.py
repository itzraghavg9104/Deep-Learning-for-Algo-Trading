"""
PPO Reinforcement Learning Agent Training for Trading.

Uses Stable-Baselines3 PPO with custom trading environment.
"""
import os
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for PPO Agent.
    
    Simulates stock trading with BUY, HOLD, SELL actions.
    Optimizes for Sharpe Ratio.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000,
        commission: float = 0.001,
        window_size: int = 30,
        risk_tolerance: float = 0.5
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.risk_tolerance = risk_tolerance
        
        # Feature columns
        self.feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space: window of prices + portfolio state
        n_features = len(self.feature_cols)
        obs_size = window_size * n_features + 3  # +3 for balance, shares, unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.avg_buy_price = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        
        self.portfolio_values = [self.initial_balance]
        self.returns = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Get price window
        start = self.current_step - self.window_size
        end = self.current_step
        
        window_data = self.df[self.feature_cols].iloc[start:end].values
        
        # Normalize prices by dividing by first value
        normalized = window_data / (window_data[0] + 1e-8)
        
        # Flatten
        obs = normalized.flatten()
        
        # Add portfolio state
        current_price = self.df['close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        unrealized_pnl = (current_price - self.avg_buy_price) * self.shares_held if self.shares_held > 0 else 0
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.shares_held * current_price / self.initial_balance,
            unrealized_pnl / self.initial_balance
        ])
        
        full_obs = np.concatenate([obs, portfolio_state]).astype(np.float32)
        return full_obs
    
    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        prev_portfolio_value = self.balance + self.shares_held * current_price
        
        # Execute action
        if action == 1:  # BUY
            # Buy with available balance (considering commission)
            max_shares = int(self.balance / (current_price * (1 + self.commission)))
            shares_to_buy = max(1, int(max_shares * self.risk_tolerance))
            
            if shares_to_buy > 0 and self.balance >= shares_to_buy * current_price * (1 + self.commission):
                cost = shares_to_buy * current_price * (1 + self.commission)
                self.balance -= cost
                
                # Update average buy price
                total_cost = self.avg_buy_price * self.shares_held + current_price * shares_to_buy
                self.shares_held += shares_to_buy
                self.avg_buy_price = total_cost / self.shares_held
                self.total_shares_bought += shares_to_buy
        
        elif action == 2:  # SELL
            if self.shares_held > 0:
                shares_to_sell = max(1, int(self.shares_held * self.risk_tolerance))
                revenue = shares_to_sell * current_price * (1 - self.commission)
                self.balance += revenue
                self.shares_held -= shares_to_sell
                self.total_shares_sold += shares_to_sell
                
                if self.shares_held == 0:
                    self.avg_buy_price = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        current_portfolio_value = self.balance + self.shares_held * self.df['close'].iloc[self.current_step] if self.current_step < len(self.df) else prev_portfolio_value
        self.portfolio_values.append(current_portfolio_value)
        
        # Calculate return
        returns = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns.append(returns)
        
        # Calculate reward (Sharpe-like)
        if len(self.returns) > 1:
            avg_return = np.mean(self.returns[-20:])  # Last 20 steps
            std_return = np.std(self.returns[-20:]) + 1e-8
            reward = avg_return / std_return  # Sharpe ratio approximation
        else:
            reward = returns
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            "portfolio_value": current_portfolio_value,
            "balance": self.balance,
            "shares": self.shares_held,
            "total_return": (current_portfolio_value - self.initial_balance) / self.initial_balance
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def render(self, mode='human'):
        current_price = self.df['close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Portfolio: {portfolio_value:.2f}")


def load_training_data(data_path: str = "./data/training_data.csv") -> pd.DataFrame:
    """Load and prepare training data."""
    df = pd.read_csv(data_path)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    return df


def train_ppo_agent(
    df: pd.DataFrame,
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    model_path: str = "./models"
) -> PPO:
    """
    Train PPO agent on trading environment.
    
    Args:
        df: DataFrame with stock data
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Batch size
        n_epochs: Epochs per update
        model_path: Path to save model
    
    Returns:
        Trained PPO model
    """
    os.makedirs(model_path, exist_ok=True)
    
    # Create environment
    env = TradingEnv(df, initial_balance=100000, risk_tolerance=0.5)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"{model_path}/ppo_tensorboard"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_path,
        name_prefix="ppo_trading"
    )
    
    # Train
    print(f"Training PPO agent for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{model_path}/ppo_trading_final")
    print(f"Model saved to {model_path}/ppo_trading_final.zip")
    
    return model


def evaluate_agent(model: PPO, df: pd.DataFrame, n_episodes: int = 5) -> dict:
    """
    Evaluate trained PPO agent.
    
    Args:
        model: Trained PPO model
        df: Test data
        n_episodes: Number of evaluation episodes
    
    Returns:
        Evaluation metrics
    """
    env = TradingEnv(df, initial_balance=100000)
    
    all_returns = []
    all_sharpe = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        
        total_return = info["total_return"]
        all_returns.append(total_return)
        
        # Calculate Sharpe
        if len(env.returns) > 1:
            sharpe = np.mean(env.returns) / (np.std(env.returns) + 1e-8) * np.sqrt(252)
            all_sharpe.append(sharpe)
    
    return {
        "avg_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "avg_sharpe": np.mean(all_sharpe) if all_sharpe else 0,
        "best_return": max(all_returns),
        "worst_return": min(all_returns)
    }


if __name__ == "__main__":
    print("=" * 50)
    print("PPO Trading Agent Training")
    print("=" * 50)
    
    # Load data
    data_path = "./data/training_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run download_data.py first.")
        exit(1)
    
    print("\nLoading data...")
    df = load_training_data(data_path)
    
    # Use single stock for training (RELIANCE)
    train_df = df[df['symbol'] == 'RELIANCE.NS'].copy()
    print(f"Training on {len(train_df)} samples from RELIANCE.NS")
    
    # Train
    print("\nTraining PPO agent...")
    model = train_ppo_agent(
        train_df,
        total_timesteps=30000,
        learning_rate=3e-4,
        model_path="./models"
    )
    
    # Evaluate
    print("\nEvaluating agent...")
    metrics = evaluate_agent(model, train_df, n_episodes=5)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Average Return: {metrics['avg_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['avg_sharpe']:.2f}")
    print(f"Best Return: {metrics['best_return']*100:.2f}%")
    print(f"Worst Return: {metrics['worst_return']*100:.2f}%")

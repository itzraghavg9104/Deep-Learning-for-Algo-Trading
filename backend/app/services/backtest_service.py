import pandas as pd
import numpy as np
import os
from datetime import date, datetime
from typing import Dict, List, Optional
from stable_baselines3 import PPO

from app.layer2_decision.trading_env import TradingEnv
from app.config import settings

class BacktestService:
    """Service for running historical backtests using the trained PPO agent."""
    
    def __init__(self, data_dir: str = "backend/data/raw"):
        self.data_dir = data_dir
        self.model_path = os.path.join(settings.MODEL_PATH, "ppo_trading_final.zip")
        self._model = None
    
    @property
    def model(self):
        """Lazy load the PPO model."""
        if self._model is None:
            if os.path.exists(self.model_path):
                self._model = PPO.load(self.model_path)
            else:
                print(f"Warning: PPO model not found at {self.model_path}. Using random actions.")
        return self._model
    
    def run(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date, 
        initial_capital: float = 100000.0,
        risk_tolerance: float = 0.5
    ) -> Dict:
        """
        Run a backtest for a specific symbol and date range.
        """
        # 1. Load data
        # Clean symbol for filename (e.g., RELIANCE.NS -> RELIANCE)
        clean_symbol = symbol.split('.')[0]
        file_path = os.path.join(self.data_dir, f"{clean_symbol}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No data found for symbol: {symbol}")
        
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Filter by date range
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        backtest_df = df.loc[mask].reset_index(drop=True)
        
        if len(backtest_df) < 50:  # Minimum data requirement
            raise ValueError(f"Insufficient data for range {start_date} to {end_date}")
        
        # 2. Initialize Environment
        env = TradingEnv(
            df=backtest_df,
            initial_capital=initial_capital,
            risk_tolerance=risk_tolerance
        )
        
        obs, _ = env.reset()
        done = False
        
        # 3. Run Simulation
        while not done:
            if self.model:
                action, _states = self.model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # Random fallback
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        
        # 4. Compile Results
        metrics = env.get_episode_metrics()
        
        return {
            "symbol": symbol,
            "total_return": float(metrics.get("total_return", 0.0)),
            "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "profit_factor": float(metrics.get("profit_factor", 0.0)),
            "total_trades": int(metrics.get("total_trades", 0)),
            "final_value": float(metrics.get("final_value", initial_capital)),
            "trades": env.trades,
            "equity_curve": metrics.get("equity_curve", [])
        }

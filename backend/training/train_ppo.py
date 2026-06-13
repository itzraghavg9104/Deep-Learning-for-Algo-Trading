"""
PPO Reinforcement Learning Agent Training for Trading.

Uses Stable-Baselines3 PPO with custom trading environment.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
import argparse
import json
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

from app.layer2_decision.trading_env import TradingEnv


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
    model_path: str = "./models",
    behavior_array: Optional[Dict[str, float]] = None,
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
    
    # App environment expects OHLCV with capitalized field names.
    env_df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    env = TradingEnv(
        env_df,
        initial_capital=100000,
        risk_tolerance=0.5,
        behavior_array=behavior_array or {
            "capital_per_trade_pct": 0.1,
            "tp_sl_ratio_preference": 0.25,
            "max_profit_close_pct": 0.2,
            "trade_frequency_window_score": 0.15,
            "avg_holding_time_score": 0.2,
            "post_loss_rest_min": 0.1,
            "drawdown_sensitivity": 0.2,
            "streak_risk_adjustment": 0.25,
            "intraday_var_limit": 0.1,
            "entry_slippage_tolerance_bps": 0.1,
            "time_of_day_performance_bias": 0.5,
            "news_proximity_buffer_min": 0.1,
            "partial_tp_preference": 0.5,
            "breakeven_migration_trigger_pct": 0.1,
            "breakeven_migration_time_min": 0.2,
        },
    )
    env = DummyVecEnv([lambda: env])

    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    if not tensorboard_available:
        print("TensorBoard not installed. Continuing PPO training without tensorboard logs.")
    
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
        tensorboard_log=f"{model_path}/ppo_tensorboard" if tensorboard_available else None
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


def evaluate_agent(
    model: PPO,
    df: pd.DataFrame,
    n_episodes: int = 1,
    max_eval_rows: int = 0,
) -> dict:
    """
    Evaluate trained PPO agent.
    
    Args:
        model: Trained PPO model
        df: Test data
        n_episodes: Number of evaluation episodes
    
    Returns:
        Evaluation metrics
    """
    eval_df = df.tail(max_eval_rows).copy() if max_eval_rows and max_eval_rows > 0 else df
    env_df = eval_df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    env = TradingEnv(env_df, initial_capital=100000)
    
    all_returns = []
    all_sharpe = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _info = env.step(int(action))

        metrics = env.get_episode_metrics()
        all_returns.append(metrics.get("total_return", 0.0) / 100.0)
        all_sharpe.append(metrics.get("sharpe_ratio", 0.0))
    
    return {
        "avg_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
        "avg_sharpe": np.mean(all_sharpe) if all_sharpe else 0,
        "best_return": max(all_returns),
        "worst_return": min(all_returns)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO trading model")
    parser.add_argument("--model-path", default="./models")
    parser.add_argument("--data-path", default="./data/training_data.csv")
    parser.add_argument("--symbol", default="ALL")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--behavior-json", default="")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    behavior_array = None
    if args.behavior_json:
        try:
            behavior_array = json.loads(args.behavior_json)
        except Exception:
            behavior_array = None

    print("=" * 50)
    print("PPO Trading Agent Training")
    print("=" * 50)
    
    # Load data
    data_path = args.data_path
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run download_data.py first.")
        exit(1)
    
    print("\nLoading data...")
    df = load_training_data(data_path)
    
    if args.symbol.upper() == "ALL":
        train_df = df.copy()
        print(f"Training on {len(train_df)} samples across {train_df['symbol'].nunique()} symbols")
    else:
        train_df = df[df["symbol"] == args.symbol].copy()
        print(f"Training on {len(train_df)} samples from {args.symbol}")
    
    # Train
    print("\nTraining PPO agent...")
    model = train_ppo_agent(
        train_df,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
        behavior_array=behavior_array,
    )
    
    if args.skip_eval:
        print("\nSkipping evaluation (--skip-eval).")
        metrics = None
    else:
        print("\nEvaluating agent...")
        metrics = evaluate_agent(
            model,
            train_df,
            n_episodes=max(1, args.eval_episodes),
            max_eval_rows=max(0, args.max_eval_rows),
        )
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    if metrics is not None:
        print(f"Average Return: {metrics['avg_return']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['avg_sharpe']:.2f}")
        print(f"Best Return: {metrics['best_return']*100:.2f}%")
        print(f"Worst Return: {metrics['worst_return']*100:.2f}%")

"""
PPO Agent wrapper for trading.

Uses Stable-Baselines3 PPO implementation.
"""
import os
from typing import Optional, Dict, Any
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. PPO agent will not be available.")

from app.layer2_decision.trading_env import TradingEnv


class TradingAgent:
    """
    PPO-based trading agent.
    
    Wraps Stable-Baselines3 PPO with trading-specific configurations.
    """
    
    def __init__(
        self,
        env: Optional[TradingEnv] = None,
        model_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1,
    ):
        """
        Initialize the trading agent.
        
        Args:
            env: Trading environment
            model_path: Path to load existing model
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Mini-batch size
            n_epochs: Epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            verbose: Verbosity level
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PPO agent")
        
        self.env = env
        self.model_path = model_path
        self.verbose = verbose
        
        # PPO hyperparameters
        self.hyperparams = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
        }
        
        self.model: Optional[PPO] = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif env is not None:
            self._create_model()
    
    def _create_model(self):
        """Create new PPO model."""
        if self.env is None:
            raise ValueError("Environment required to create model")
        
        # Wrap in vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        self.model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=self.verbose,
            **self.hyperparams,
            policy_kwargs={
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            }
        )
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[TradingEnv] = None,
        eval_freq: int = 10000,
        save_path: str = "./models",
        log_path: str = "./logs",
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency
            save_path: Path to save models
            log_path: Path for tensorboard logs
        
        Returns:
            Training metrics
        """
        if self.model is None:
            self._create_model()
        
        # Create directories
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=save_path,
            name_prefix="ppo_trading"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if eval_env is not None:
            eval_vec_env = DummyVecEnv([lambda: eval_env])
            eval_callback = EvalCallback(
                eval_vec_env,
                best_model_save_path=save_path,
                log_path=log_path,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        return {"status": "completed", "timesteps": total_timesteps}
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action for given observation.
        
        Args:
            observation: State vector
            deterministic: Use deterministic policy
        
        Returns:
            Tuple of (action, action_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        # Get action probabilities
        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()
        
        return int(action), probs.flatten()
    
    def get_action_with_confidence(
        self,
        observation: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get action with confidence level.
        
        Args:
            observation: State vector
        
        Returns:
            Dictionary with action and confidence
        """
        action, probs = self.predict(observation)
        
        action_names = ["HOLD", "BUY", "SELL"]
        
        return {
            "action": action_names[action],
            "action_code": action,
            "confidence": float(probs[action]),
            "probabilities": {
                "HOLD": float(probs[0]),
                "BUY": float(probs[1]),
                "SELL": float(probs[2]),
            }
        }
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str):
        """Load model from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.model = PPO.load(path)
        self.model_path = path


def create_agent(
    env: TradingEnv,
    risk_tolerance: float = 0.5
) -> TradingAgent:
    """
    Factory function to create agent with risk-adjusted hyperparameters.
    
    Args:
        env: Trading environment
        risk_tolerance: Trader's risk tolerance
    
    Returns:
        Configured TradingAgent
    """
    # Adjust hyperparameters based on risk tolerance
    # Conservative traders: more exploration, lower learning rate
    # Aggressive traders: more exploitation, higher learning rate
    
    learning_rate = 1e-4 + (risk_tolerance * 4e-4)  # 1e-4 to 5e-4
    ent_coef = 0.02 - (risk_tolerance * 0.015)  # 0.02 to 0.005
    
    return TradingAgent(
        env=env,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
    )

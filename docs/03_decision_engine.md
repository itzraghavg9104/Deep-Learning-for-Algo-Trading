# Layer 2: Decision Engine

## Overview

Layer 2 is the "Decision" component that takes the processed state from Layer 1 and outputs optimal trading actions using Deep Reinforcement Learning (PPO algorithm).

---

## Why PPO?

Proximal Policy Optimization (PPO) was chosen based on research recommendations:

| Algorithm | Pros | Cons |
|-----------|------|------|
| **DQN** | Simple, effective | Discrete actions only, unstable |
| **A2C** | Good for continuous | High variance |
| **PPO** | Stable, robust, sample efficient | Slightly slower |

**PPO Advantages**:
- ✅ Stable training (clipped objective prevents large policy updates)
- ✅ Works well with high-dimensional state spaces
- ✅ Balances exploration and exploitation effectively

---

## Trading Environment

**Custom Gym Environment** that simulates the market:

```python
class TradingEnv(gym.Env):
    action_space = Discrete(3)  # 0: HOLD, 1: BUY, 2: SELL
    observation_space = Box(low=-inf, high=inf, shape=(50,))
    
    def step(self, action):
        # Execute action
        # Calculate reward (Sharpe Ratio)
        # Update portfolio state
        return next_state, reward, done, info
```

### Action Space

| Action | Code | Description |
|--------|------|-------------|
| HOLD | 0 | Maintain current position |
| BUY | 1 | Open/increase long position |
| SELL | 2 | Close/reduce position |

### Position Sizing

Position size is determined by the Trader Behavior module based on:
- Risk tolerance
- Kelly Criterion
- Current portfolio allocation

---

## Reward Function: Sharpe Ratio

**Why Sharpe Ratio?**
- Optimizes risk-adjusted returns, not just raw returns
- Penalizes high volatility strategies
- Aligns with professional trading metrics

**Implementation**:
```python
def calculate_reward(returns_window, risk_free_rate=0.02/252):
    """
    Calculate rolling Sharpe Ratio as reward.
    
    Args:
        returns_window: Last N daily returns
        risk_free_rate: Daily risk-free rate
    
    Returns:
        Annualized Sharpe Ratio
    """
    excess_returns = returns_window - risk_free_rate
    
    if len(excess_returns) < 2:
        return 0.0
    
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns) + 1e-8  # Avoid division by zero
    
    sharpe = mean_return / std_return
    annualized_sharpe = sharpe * np.sqrt(252)
    
    return annualized_sharpe
```

**Reward Signal**:
- Calculated over rolling 5-21 day window
- Updated after each action
- Scaled to reasonable range [-2, 2]

---

## PPO Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        STATE INPUT                           │
│                    (50 dimensions)                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│    POLICY NETWORK     │       │    VALUE NETWORK      │
│       (Actor)         │       │       (Critic)        │
├───────────────────────┤       ├───────────────────────┤
│ Input: 50             │       │ Input: 50             │
│ Hidden 1: 256 (ReLU)  │       │ Hidden 1: 256 (ReLU)  │
│ Hidden 2: 256 (ReLU)  │       │ Hidden 2: 256 (ReLU)  │
│ Output: 3 (Softmax)   │       │ Output: 1 (Linear)    │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            ▼                               ▼
    π(a|s) = P(action)              V(s) = State Value
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | Adam optimizer LR |
| `n_steps` | 2048 | Steps per update |
| `batch_size` | 64 | Mini-batch size |
| `n_epochs` | 10 | Epochs per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE parameter |
| `clip_range` | 0.2 | PPO clipping |
| `ent_coef` | 0.01 | Entropy coefficient |

---

## Training Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Historical │────▶│   Training   │────▶│  Trained PPO    │
│    Data     │     │   Loop       │     │     Model       │
└─────────────┘     └──────────────┘     └─────────────────┘
      │                    │                      │
      │            ┌───────┴───────┐              │
      │            │  Evaluation   │              │
      │            │  (Sharpe,     │              │
      │            │   Drawdown)   │              │
      │            └───────────────┘              │
      │                                           │
      └───────────────────┬───────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │    Backtesting      │
              │    Validation       │
              └─────────────────────┘
```

### Training Steps

1. **Data Preparation**: Load historical data, compute indicators
2. **Environment Setup**: Initialize TradingEnv with training data
3. **Model Training**: Run PPO for N episodes
4. **Evaluation**: Calculate Sharpe, Drawdown, Win Rate
5. **Checkpointing**: Save best models

### Training Command

```bash
python training/train_ppo.py \
    --symbol AAPL \
    --start-date 2018-01-01 \
    --end-date 2023-01-01 \
    --total-timesteps 1000000 \
    --eval-freq 10000
```

---

## Inference Pipeline

```python
# Load trained model
model = PPO.load("models/ppo_trading_agent.zip")

# Get current state from Layer 1
state = state_builder.build_state(
    market_data=current_ohlcv,
    trader_profile=user_profile,
    current_portfolio=portfolio
)

# Get action probabilities
action, _ = model.predict(state, deterministic=True)
action_probs = model.policy.get_distribution(state).distribution.probs

# Generate signal
signal = {
    "action": ["HOLD", "BUY", "SELL"][action],
    "confidence": float(action_probs[action]),
    "position_size": calculate_position_size(user_profile, action_probs)
}
```

---

## Implementation Files

| File | Purpose |
|------|---------|
| `trading_env.py` | Custom Gym environment |
| `ppo_agent.py` | PPO model wrapper |
| `reward_function.py` | Sharpe Ratio calculation |
| `trainer.py` | Training pipeline |
| `evaluator.py` | Model evaluation metrics |

---

## Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Max Drawdown | < 20% | Maximum peak-to-trough |
| Win Rate | > 50% | Profitable trades ratio |
| Profit Factor | > 1.5 | Gross profit / Gross loss |

---

## Next Steps

- See [Trader Behavior](04_trader_behavior.md) for personalization
- See [API Reference](05_api_reference.md) for endpoint details

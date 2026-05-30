# 4. ML Pipeline and Models

## 4.1 Overview of Model Architecture

The system uses three machine learning models in a pipeline arrangement:

1. **LSTM** — Supervised time-series forecasting for price prediction
2. **PPO (Proximal Policy Optimization)** — Reinforcement learning for trading decisions
3. **DeepAR** — Probabilistic forecasting (experimental, trained but not used at runtime)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                             │
│                                                                      │
│  download_data.py    train_lstm.py          train_ppo.py             │
│  ┌──────────────┐    ┌──────────────┐       ┌──────────────┐         │
│  │ yfinance     │───→│ LSTM         │       │ PPO          │         │
│  │ NIFTY 50     │    │ Seq Len: 30  │       │ SB3 PPO      │         │
│  │ 5 years      │    │ Features: 5  │       │ MLP [256,256]│         │
│  │ Daily OHLCV  │    │ Hidden: 64   │       │ 5 Actions    │         │
│  └──────────────┘    │ Epochs: 30   │       │ 30k steps    │         │
│       │              └──────┬───────┘       └──────┬────────┘        │
│       ▼                     ▼                      ▼                  │
│  ┌──────────────┐    ┌──────────────┐       ┌──────────────┐         │
│  │ training_data│    │ lstm_final.pt│       │ ppo_trading_ │         │
│  │ .csv         │    │ lstm_best.pt │       │ final.zip    │         │
│  └──────────────┘    └──────────────┘       └──────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         RUNTIME INFERENCE                            │
│                                                                      │
│  GET /trading/signals/{symbol}                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌────────────────────────────────────────┐                          │
│  │ PredictionService.predict()            │                          │
│  │                                        │                          │
│  │  1. LSTM Prediction                    │                          │
│  │     └─ StockLSTM.forward(sequence)     │                          │
│  │     └─ Returns: predicted_price,       │                          │
│  │        directional_action, confidence  │                          │
│  │                                        │                          │
│  │  2. PPO Signal                         │                          │
│  │     └─ TradingEnv._get_observation()   │                          │
│  │     └─ PPO.predict(observation)        │                          │
│  │     └─ Returns: action, confidence=0.8 │                          │
│  │                                        │                          │
│  │  3. Merge: PPO action overrides LSTM   │                          │
│  │     Returns: LSTM+PPO combined result   │                          │
│  └────────────────────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────────┘
```

## 4.2 Data Download Pipeline (`download_data.py`)

### Purpose
Downloads 5 years of historical OHLCV data for all 52 NIFTY 50 constituent stocks from Yahoo Finance.

### Execution
```bash
python training/download_data.py
```

### Data Sources
- **52 NIFTY 50 stocks** via `get_nifty50_symbols()` from `market_data.py`
- Symbols use `.NS` suffix (e.g., `RELIANCE.NS`)
- **yfinance** library for data retrieval

### Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| start_date | "2020-01-01" | Start of historical data |
| end_date | today (auto) | End of historical data |
| interval | "1d" | Daily data |
| output_dir | "./data/raw" | Per-symbol CSVs |

### Output Files

| File | Content | Rows (approx) |
|------|---------|----------------|
| `data/raw/{Symbol}.csv` | Per-symbol OHLCV | ~1,588 rows per symbol |
| `data/raw/nifty50_combined.csv` | All symbols concatenated | ~79,600 rows |
| `data/training_data.csv` | Cleaned/prepared training set | ~79,600 rows |

### Data Processing Steps

1. **Per-symbol download** — yfinance Ticker.history() for each symbol
2. **Column standardization** — Adds Date, Symbol, time_idx
3. **Individual save** — Each symbol saved to `data/raw/{Symbol}.csv` (strips `.NS` suffix from filename)
4. **Combined save** — All data concatenated to `nifty50_combined.csv`
5. **`prepare_training_data()`** — Creates the ML-ready dataset:
   - Selects columns: date, symbol, open, high, low, close, volume
   - Renames to lowercase
   - Sorts by symbol → date
   - Adds `time_idx` (sequential integer per symbol)
   - Adds `returns` (pct_change per symbol)
   - Adds `log_returns` (log returns per symbol)
   - Drops NA values

## 4.3 LSTM Model (`train_lstm.py`)

### Model Architecture

```python
class StockLSTM(nn.Module):
    LSTM(2 layers, hidden=64, dropout=0.2) →
    FC(64 → 32, ReLU, Dropout(0.2)) →
    FC(32 → 1)
```

| Component | Value |
|-----------|-------|
| Input size | 5 (open, high, low, close, volume) |
| Sequence length | 30 days |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 (between LSTM layers, after FC) |
| Output | 1 (predicted next close) |
| Total parameters | ~39,000 |

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Loss function | MSELoss |
| Optimizer | Adam (lr=0.001) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Gradient clipping | max_norm=1.0 |
| Batch size | 64 |
| Epochs | 30 (configurable) |
| Train/Val split | 80/20 (chronological) |
| Device | CUDA if available, else CPU |

### Data Preparation

1. Loads `data/training_data.csv`
2. Groups by symbol, applies per-symbol `MinMaxScaler` to features [open, high, low, close, volume]
3. Creates sequences of length 30: `seq = scaled_data[i:i+30]`, `target = scaled_data[i+30, 3]` (close price index)
4. Splits 80/20 chronologically (not randomly — respects time series order)

### Training Loop

```python
for epoch in range(epochs):
    # Training
    model.train()
    for sequences, targets in train_loader:
        loss = MSELoss(model(sequences), targets)
        loss.backward()
        clip_grad_norm_(1.0)
        optimizer.step()
    
    # Validation
    model.eval()
    for sequences, targets in val_loader:
        val_loss += MSELoss(model(sequences), targets)
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), './models/lstm_best.pt')
```

### Model Checkpoints

| File | Format | Contents |
|------|--------|----------|
| `models/lstm_best.pt` | state_dict only | Best validation loss weights |
| `models/lstm_final.pt` | Full dict | `{'model_state_dict': ..., 'history': {'train_loss': [...], 'val_loss': [...]}, 'config': {'seq_length': 30, 'hidden_size': 64, 'features': ['open','high','low','close','volume']}}` |

### Inference Flow (in `PredictionService`)

1. Fetch 3 months of daily data for the symbol
2. Lowercase column names
3. Create fresh `MinMaxScaler`, fit on last `seq_length` data points
4. Scale features, create tensor of shape `(1, seq_length, 5)`
5. Model forward pass → scaled prediction
6. Inverse transform via scaler to get price
7. Compute `change_pct = (predicted - current) / current * 100`
8. Map to action:
   - `change_pct > +1%` → BUY (confidence = min(0.5 + change_pct/10, 0.95))
   - `change_pct < -1%` → SELL (confidence = min(0.5 + |change_pct|/10, 0.95))
   - Otherwise → IDLE (confidence = 0.5)

## 4.4 PPO Model (`train_ppo.py`)

### Algorithm

Proximal Policy Optimization (PPO) via Stable-Baselines3, a policy gradient method for reinforcement learning that maintains training stability by clipping policy updates.

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| policy | MlpPolicy | Multi-layer perceptron policy |
| net_arch | [256, 256] | Shared layers for policy and value |
| learning_rate | 3e-4 | Adam optimizer learning rate |
| n_steps | 2048 | Steps per environment update |
| batch_size | 64 | Mini-batch size for SGD |
| n_epochs | 10 | PPO optimization epochs per update |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE smoothing parameter |
| clip_range | 0.2 | PPO clipping range |
| ent_coef | 0.01 | Entropy bonus coefficient |
| vf_coef | 0.5 | Value function loss coefficient |
| max_grad_norm | 0.5 | Gradient clipping |

### Training Environment

The PPO agent is trained in the `TradingEnv` Gymnasium environment:

| Feature | Value |
|---------|-------|
| Observation space | Box(34,) — 34-dimensional state vector |
| Action space | Discrete(5) — HOLD BUY, HOLD SELL, BUY, SELL, IDLE |
| Initial capital | 100,000 |
| Max shares/trade | 100 |
| Transaction cost | 0.1% per trade |
| Risk tolerance | 0.5 (moderate) |
| Window size | 20 days |

**Training Data:**
- Uses `data/training_data.csv` with O/H/L/C/V columns capitalised for TradingEnv compatibility
- Column mapping: open→Open, high→High, low→Low, close→Close, volume→Volume
- By default trained on ALL symbols (52 stocks) over 5 years

**Default Behavior Array (used in training):**
```python
{
    "capital_per_trade_pct": 0.1,
    "tp_sl_ratio_preference": 0.4,
    "drawdown_sensitivity": 0.2,
    "post_loss_rest_min": 0.1,
}
```

### Training Flow

```bash
python training/train_ppo.py
                  [--model-path ./models]
                  [--data-path ./data/training_data.csv]
                  [--symbol ALL]
                  [--timesteps 30000]
                  [--learning-rate 3e-4]
                  [--behavior-json '{"capital_per_trade_pct": 0.1, ...}']
```

1. Loads training data, capitalizes columns
2. Creates TradingEnv with behavior defaults (or from --behavior-json)
3. Wraps in DummyVecEnv for SB3 compatibility
4. Creates PPO model with configured hyperparameters
5. Sets up CheckpointCallback (every 10k steps)
6. Train for 30,000 timesteps (default)
7. Save final model to `{model_path}/ppo_trading_final.zip`
8. Evaluate over 5 episodes: reports avg_return, std_return, avg_sharpe

### Inference Flow (in `PredictionService`)

1. Fetch 1 month of daily data
2. Create temporary TradingEnv with the data + user's risk tolerance + behavior array
3. Reset env → get initial observation (34-dim state)
4. Load PPO model (user-specific: `models/users/{user_id}/ppo_trading_final.zip` if exists, else global: `models/ppo_trading_final.zip`)
5. `ppo_model.predict(observation, deterministic=True)` → action
6. Map action index to label via `ACTION_LABELS`
7. Return action with confidence=0.8 (hardcoded — not derived from policy)

### Combined LSTM+PPO Inference

```python
def predict(symbol, risk_tolerance, behavior_array, user_id):
    lstm_result = _get_lstm_prediction(symbol)       # Price forecast + action
    ppo_result = get_ppo_signal(symbol, ...)          # RL action
    
    final = lstm_result.copy()
    if ppo_result has no error:
        final["action"] = ppo_result["action"]        # PPO overrides
        final["model"] = "LSTM+PPO"
    return final
```

The LSTM provides the price prediction (direction, magnitude), while the PPO agent provides the trading decision. If PPO is available, its action choice overrides the LSTM's rule-based action.

## 4.5 DeepAR Model (`train_deepar.py`) — Experimental

### Status
**Trained but not integrated into runtime inference.** The `state_builder.py` has placeholder fields for DeepAR predictions (`pred_price_mean`, `pred_price_std`, `pred_change_pct`, `pred_confidence`), but they default to 0 in actual runtime.

### Architecture
- Uses `pytorch_forecasting` DeepAR implementation
- Probabilistic time series forecasting
- Encoder length: 30 days
- Prediction length: 5 days
- Hidden size: 32
- RNN layers: 2
- Dropout: 0.1
- Lightning Trainer with EarlyStopping + ModelCheckpoint

### Output
- `models/deepar_final.pt` — saved but never loaded at runtime

## 4.6 Per-User PPO Retraining (`user_model_training_service.py`)

### Trigger
Triggered by `POST /profile/behavior-assessment` endpoint when behavior answers are submitted.

### Flow
```
POST /profile/behavior-assessment
  → trigger_user_retraining(user_id, behavior_array)
    → Spawns daemon thread
      → _train_user_models(user_id, behavior_array)
        → Acquires per-user lock (non-blocking)
        → If locked: queues behavior, returns
        → Creates models/users/{user_id}/
        → Runs download_data.py if training data missing
        → Runs train_ppo.py --model-path <dir> --behavior-json <json>
        → Saves meta.json
        → Checks for queued updates → loops if found
```

### Thread Safety
- Per-user `threading.Lock` prevents concurrent training for same user
- If training in progress, new behavior assessment is queued (`_pending_behavior`)
- After training completes, checks for queued updates and retrains if needed

### Training Status

| Status | Description |
|--------|-------------|
| idle | No training ever requested |
| queued | Request accepted, waiting for thread execution |
| running | Thread executing train_ppo.py |
| completed | Training finished successfully |
| failed | Exception during training |

Queried via `GET /profile/model-training-status`.

## 4.7 Model Bootstrap at Startup (`model_bootstrap.py`)

### Execution
Called during FastAPI startup event (`main.py` → `ensure_models_ready()`)

### Logic
1. Resolve model directory (supporting relative and absolute paths)
2. Check for `lstm_final.pt` and `ppo_trading_final.zip`
3. If all present → skip
4. If any missing:
   - If `AUTO_TRAIN_IF_MISSING=True`: run download script → train missing models
   - Scripts executed as subprocess with proper PYTHONPATH
   - If `AUTO_TRAIN_STRICT=True` and failure occurs: raise RuntimeError (fails startup)
5. If `AUTO_TRAIN_IF_MISSING=False`: log warning, continue without models

## 4.8 Model File Locations

All model files are gitignored (`*.pt`, `*.zip` patterns in `.gitignore`):

| File | Model | Created By | Used By |
|------|-------|-----------|---------|
| `models/lstm_final.pt` | LSTM final checkpoint | `train_lstm.py` | `PredictionService` |
| `models/lstm_best.pt` | LSTM best validation | `train_lstm.py` | Manual inspection |
| `models/ppo_trading_final.zip` | PPO agent (global) | `train_ppo.py` | `PredictionService`, `BacktestService` |
| `models/ppo_tensorboard/` | TensorBoard logs | `train_ppo.py` | Training monitoring |
| `models/users/{uid}/ppo_trading_final.zip` | PPO agent (per-user) | `user_model_training_service.py` | `PredictionService` (user-specific override) |
| `models/deepar_final.pt` | DeepAR model | `train_deepar.py` | Not used at runtime |

## 4.9 Fallback Behavior

When model files are missing or fail to load:

### Trading Signals
- `PredictionService._get_lstm_prediction()` returns error dict if model not loaded
- `PredictionService.get_ppo_signal()` returns IDLE with 0.5 confidence if model not found
- Trading route falls back to IDLE action with 0.5 confidence
- Watchlist returns "fallback" model indicator

### Backtest
- `BacktestService.model` property returns None if PPO model not found
- Backtest falls back to `env.action_space.sample()` — random actions
- Warning logged but simulation continues

## 4.10 Agent Creation with Risk Adjustment

The `create_agent()` factory in `ppo_agent.py` adjusts PPO hyperparameters based on risk tolerance:

| Risk Tolerance | learning_rate | ent_coef | Behavioral Effect |
|---------------|---------------|----------|-------------------|
| 0.0 (Conservative) | 1e-4 | 0.02 | Slow learning, high exploration |
| 0.5 (Moderate) | 3e-4 | 0.0125 | Balanced |
| 1.0 (Aggressive) | 5e-4 | 0.005 | Fast learning, low exploration |

This enables personalized PPO agents tuned to each user's risk profile.

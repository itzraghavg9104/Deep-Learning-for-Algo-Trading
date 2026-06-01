# 4. ML Pipeline and Models

## 4.1 Overview of Model Architecture

The system uses three machine learning models in a pipeline arrangement:

1. **LSTM** — Supervised time-series forecasting for price prediction
2. **PPO (Proximal Policy Optimization)** — Reinforcement learning for trading decisions
3. **DeepAR** — Probabilistic forecasting (experimental, trained but not used at runtime)

### Why These Models?

- **LSTM**: Chosen for its proven ability to capture long-term dependencies in time series data (Hochreiter & Schmidhuber, 1997). The 30-day sequence window provides context of ~6 trading weeks, enough for short-to-medium term patterns.
- **PPO**: Selected over DQN or A2C for its stability in training (Schulman et al., 2017). The clipped surrogate objective prevents destructive policy updates, making it suitable for the high-variance financial domain.
- **DeepAR**: Included as experimental (Salinas et al., 2020). Probabilistic forecasting would ideally provide confidence intervals around predictions, but integration remains incomplete.

### Training Results (from actual runs)

| Model | Metric | Value |
|-------|--------|-------|
| **LSTM** | Training Samples | 23,167 |
| | Validation Loss (MSE) | **0.000228** |
| | Architecture | 2-layer LSTM (hidden=64), seq_len=30, ~39K params |
| | Optimizer | Adam (lr=0.001) |
| | Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| **PPO** | Training Timesteps | 30,000 |
| | Average Return (5 eval episodes) | **132.28%** |
| | Sharpe Ratio | **0.66** |
| | Policy Network | MlpPolicy [256, 256] |
| | Risk-Free Rate | ~8% annual (Indian market proxy) |

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
| start_date | "2020-01-01" | Start of historical data (~5.5 years) |
| end_date | today (auto) | End of historical data |
| interval | "1d" | Daily data |
| output_dir | "./data/raw" | Per-symbol CSVs |

### Output Files

| File | Content | Rows (approx) |
|------|---------|----------------|
| `data/raw/{Symbol}.csv` | Per-symbol OHLCV | ~1,588 rows per symbol |
| `data/raw/nifty50_combined.csv` | All symbols concatenated | ~82,576 rows |
| `data/training_data.csv` | Cleaned/prepared training set | ~82,576 rows |

### Data Processing Steps

1. **Per-symbol download** — yfinance Ticker.history() for each symbol, 5-year period
2. **Column standardization** — Adds Date, Symbol, time_idx (sequential integer)
3. **Individual save** — Each symbol saved to `data/raw/{Symbol}.csv` (strips `.NS` suffix from filename)
4. **Combined save** — All data concatenated to `nifty50_combined.csv`
5. **`prepare_training_data()`** — Creates the ML-ready dataset:
   - Selects columns: date, symbol, open, high, low, close, volume
   - Renames to lowercase
   - Sorts by symbol → date
   - Adds `time_idx` (sequential integer per symbol, 1-indexed)
   - Adds `returns` (pct_change per symbol)
   - Adds `log_returns` (log returns per symbol: `log(close / close.shift(1))`)
   - Drops NA values (first row per symbol after pct_change)

## 4.3 LSTM Model (`train_lstm.py`)

### Model Architecture

```python
class StockLSTM(nn.Module):
    LSTM(2 layers, hidden=64, dropout=0.2) →
    FC(64 → 32, ReLU, Dropout(0.2)) →
    FC(32 → 1)
```

| Component | Value | Justification |
|-----------|-------|---------------|
| Input size | 5 (open, high, low, close, volume) | OHLCV as minimal required features |
| Sequence length | 30 days | ~6 trading weeks; balances context vs training data |
| Hidden size | 64 | Sufficient capacity for financial time series without overfitting |
| LSTM layers | 2 | Captures hierarchical temporal patterns; 1 is too shallow, 3+ overfits |
| Dropout | 0.2 (between LSTM layers, after FC) | Standard regularization, prevents co-adaptation |
| Output | 1 (predicted next close) | Single-step ahead forecast |
| Total parameters | ~39,000 | Computed: LSTM params (4 * (5*64 + 64*64 + 64) * 2) + FC1 (64*32 + 32) + FC2 (32*1 + 1) |

### Training Configuration

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| Loss function | MSELoss | Standard regression loss for price prediction |
| Optimizer | Adam (lr=0.001) | Adaptive learning rate, good default for RNNs |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) | Reduces LR when validation plateaus |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients in recurrent network |
| Batch size | 64 | Memory-efficient, works well with ~23K training samples |
| Epochs | 30 | Sufficient for convergence; early stopping will halt early |
| Train/Val split | 80/20 (chronological) | Respects temporal order — no data leakage |
| Device | CUDA if available, else CPU | Automatic GPU detection |

### Data Preparation

1. Loads `data/training_data.csv` (~82K rows)
2. Groups by symbol, applies per-symbol `MinMaxScaler` to features [open, high, low, close, volume]
3. Creates sequences of length 30: `seq = scaled_data[i:i+30]`, `target = scaled_data[i+30, 3]` (close price index)
4. Splits 80/20 chronologically (not randomly — respects time series order)
5. Result: ~23,167 training samples (after sequence creation + split)

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

# Save final model with metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'history': {'train_loss': train_hist, 'val_loss': val_hist},
    'config': {'seq_length': 30, 'hidden_size': 64, 'features': features}
}, './models/lstm_final.pt')
```

### Model Checkpoints

| File | Format | Contents |
|------|--------|----------|
| `models/lstm_best.pt` | state_dict only | Best validation loss weights (saved via torch.save) |
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

**Known Issue**: `train_lstm.py` scales per-symbol with symbol-specific MinMaxScaler fitted on all historical data, but `PredictionService` creates a fresh scaler on the last 30 data points. This scaling mismatch can cause prediction distribution shift at inference time.

## 4.4 PPO Model (`train_ppo.py`)

### Algorithm

Proximal Policy Optimization (PPO) via Stable-Baselines3, a policy gradient method for reinforcement learning that maintains training stability by clipping policy updates (Schulman et al., 2017).

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| policy | MlpPolicy | Multi-layer perceptron policy; sufficient for vector observations |
| net_arch | [256, 256] | Two hidden layers of 256; balances expressiveness and training speed |
| learning_rate | 3e-4 | Standard default for PPO; risk-adjusted via create_agent factory |
| n_steps | 2048 | Steps per environment update before policy update |
| batch_size | 64 | Mini-batch size for SGD; 2048/64 = 32 updates per epoch |
| n_epochs | 10 | PPO optimization epochs per update; higher = more reuse |
| gamma | 0.99 | Discount factor; near-1 for long-term trading horizon |
| gae_lambda | 0.95 | GAE smoothing parameter; bias-variance tradeoff |
| clip_range | 0.2 | PPO clipping range; standard value prevents large updates |
| ent_coef | 0.01 | Entropy bonus; encourages exploration |
| vf_coef | 0.5 | Value function loss coefficient; balances policy vs value learning |
| max_grad_norm | 0.5 | Gradient clipping; conservative for financial domain |

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

**Reward Signal:**
The reward function combines:
- **Step reward**: position * price_change_pct - transaction_cost (0.001 for trades) - risk_penalty
- **Episode reward**: Sharpe ratio over portfolio value trajectory (window_size=20), scaled via tanh to [-1, 1]
- Final reward = step_reward + 0.1 * episode_sharpe_reward (blended)

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
7. Save final model to `{model_path}/ppo_training_final.zip`
8. Evaluate over 5 episodes: reports avg_return, std_return, avg_sharpe

### Inference Flow (in `PredictionService`)

1. Fetch 1 month of daily data
2. Create temporary TradingEnv with the data + user's risk tolerance + behavior array
3. Reset env → get initial observation (34-dim state)
4. Load PPO model (user-specific: `models/users/{user_id}/ppo_training_final.zip` if exists, else global: `models/ppo_training_final.zip`)
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

### Risk-Adjusted Agent Creation

The `create_agent()` factory in `ppo_agent.py` adjusts PPO hyperparameters based on risk tolerance:

| Risk Tolerance | learning_rate | ent_coef | Behavioral Effect |
|---------------|---------------|----------|-------------------|
| 0.0 (Conservative) | 1e-4 | 0.02 | Slow learning, high exploration, smaller updates |
| 0.5 (Moderate) | 3e-4 | 0.0125 | Balanced default |
| 1.0 (Aggressive) | 5e-4 | 0.005 | Fast learning, low exploration, larger updates |

This enables personalized PPO agents tuned to each user's risk profile.

## 4.5 DeepAR Model (`train_deepar.py`) — Experimental

### Status
**Trained but not integrated into runtime inference.** The `state_builder.py` has placeholder fields for DeepAR predictions (`pred_price_mean`, `pred_price_std`, `pred_change_pct`, `pred_confidence`), but they default to 0 in actual runtime.

### Purpose
Probabilistic time series forecasting — unlike LSTM which gives a point estimate, DeepAR provides a probability distribution with mean and variance. This would enable confidence-aware state representation.

### Architecture
- Uses `pytorch_forecasting` DeepAR implementation (Salinas et al., 2020)
- Probabilistic forecasting with negative binomial likelihood
- Encoder length: 30 days
- Prediction length: 5 days (multi-step ahead)
- Hidden size: 32
- RNN layers: 2 (LSTM cell type)
- Dropout: 0.1
- Loss: Negative log-likelihood (Gaussian / NegativeBinomial)

### Training Configuration
- PyTorch Lightning Trainer
- EarlyStopping callback (patience=5)
- ModelCheckpoint callback (best validation loss)
- Learning rate finder (LRFinder) before training
- Batch size: 64
- Max epochs: 50

### Output
- `models/deepar_final.pt` — saved but never loaded at runtime

### Why Experimental
1. Longer training time than LSTM
2. Multi-step prediction (5 days) introduces compounding error
3. Integration with state_builder and reward function not completed
4. Decision logic for uncertainty-adjusted trading not designed

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
        → If locked: queues behavior, returns (latest wins)
        → Creates models/users/{user_id}/
        → Runs download_data.py if training data missing
        → Runs train_ppo.py --model-path <dir> --behavior-json <json>
        → Saves meta.json (config, timestamps, status)
        → Checks for queued updates → loops if found
```

### Thread Safety
- Per-user `threading.Lock` (stored in `_training_locks: Dict[str, Lock]`) prevents concurrent training for same user
- If training in progress, new behavior assessment is queued in `_pending_behavior: Dict[str, Dict]`
- After training completes, checks for queued updates and retrains if needed (latest behavior array wins)

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
3. If all present → skip (log info)
4. If any missing:
   - If `AUTO_TRAIN_IF_MISSING=True`: run download script → train missing models
   - Scripts executed as subprocess via `_run_script()` with proper PYTHONPATH
   - If `AUTO_TRAIN_STRICT=True` and failure occurs: raise `RuntimeError` (fails startup)
   - If `AUTO_TRAIN_STRICT=False`: log warning, continue without models
5. If `AUTO_TRAIN_IF_MISSING=False`: log warning, continue without models

## 4.8 Model File Locations

All model files are gitignored (`*.pt`, `*.zip` patterns in `.gitignore`):

| File | Model | Created By | Used By |
|------|-------|-----------|---------|
| `models/lstm_final.pt` | LSTM final checkpoint (~500KB) | `train_lstm.py` | `PredictionService` |
| `models/lstm_best.pt` | LSTM best validation | `train_lstm.py` | Manual inspection |
| `models/ppo_trading_final.zip` | PPO agent (global) | `train_ppo.py` | `PredictionService`, `BacktestService` |
| `models/ppo_tensorboard/` | TensorBoard logs | `train_ppo.py` | Training monitoring (`tensorboard --logdir models/ppo_tensorboard`) |
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

### Complete Fallback Tree
```
Trading Signal Request
  ├── PredictionService loaded?
  │   ├── Yes → LSTM predict
  │   │   ├── Success → combine with PPO
  │   │   │   ├── PPO available → LSTM+PPO signal
  │   │   │   └── PPO missing → LSTM-only signal
  │   │   └── Error → IDLE, 0.5 confidence, model="fallback"
  │   └── No → IDLE, 0.5 confidence, model="fallback"
  └── Not loaded → IDLE, 0.5 confidence, model="fallback"
```

## 4.10 Complete Data Flow from Raw Market to Trading Action

```
yfinance (52 NIFTY 50 stocks, 5 years)
    │
    ▼
download_data.py
    │
    ├── data/raw/{Symbol}.csv (per-symbol OHLCV)
    ├── data/raw/nifty50_combined.csv (all symbols)
    └── data/training_data.csv (cleaned, with features)
           │
           ▼
    train_lstm.py                train_ppo.py
    │                            │
    │ 1. Per-symbol scaling      │ 1. Capitalize columns
    │ 2. Create sequences (30)   │ 2. Create TradingEnv
    │ 3. 80/20 chronological     │ 3. DummyVecEnv wrapper
    │ 4. Train StockLSTM         │ 4. Train PPO (30k steps)
    │ 5. Save lstm_final.pt      │ 5. Save ppo_trading_final.zip
    │                            │
    ▼                            ▼
    ┌─────────────────────────────────────┐
    │       Runtime Inference              │
    │                                     │
    │ GET /trading/signals/{symbol}       │
    │ 1. get_market_data(symbol, "3mo")   │
    │ 2. compute_indicators(df) → 30+     │
    │ 3. _get_lstm_prediction(symbol)     │
    │    └─ MinMaxScaler → 30-day seq     │
    │    └─ StockLSTM.forward() → price   │
    │ 4. get_ppo_signal(symbol, ...)      │
    │    └─ Create TradingEnv → obs       │
    │    └─ PPO.predict(obs) → action     │
    │ 5. Merge: PPO action overrides      │
    │ 6. Return SignalResponse            │
    └─────────────────────────────────────┘
```

# 4. ML Pipeline and Models

## 4.1 Training Data Workflow

`backend/training/download_data.py`:

- downloads historical OHLCV for selected NIFTY symbols via yfinance
- writes per-symbol CSV files to `backend/data/raw/`
- writes combined file `backend/data/raw/nifty50_combined.csv`
- prepares cleaned training frame in `backend/data/training_data.csv`

## 4.2 LSTM Training

`backend/training/train_lstm.py`:

- sequence model over features: `open, high, low, close, volume`
- default sequence length: 30
- target: next-step close price (scaled)
- saves:
  - best weights: `./models/lstm_best.pt`
  - final checkpoint: `./models/lstm_final.pt`

Inference path in app:

- `PredictionService` loads `lstm_final.pt`
- computes predicted price and directional action confidence

## 4.3 PPO Training

`backend/training/train_ppo.py`:

- defines RL environment for BUY/HOLD/SELL simulation
- objective approximates risk-adjusted returns
- uses Stable-Baselines3 PPO
- outputs model: `./models/ppo_trading_final.zip`

Inference path in app:

- trading signal flow can combine LSTM projection with PPO action
- backtest engine uses PPO policy if model exists

## 4.4 DeepAR Script Status

`backend/training/train_deepar.py` exists, but the current online prediction flow does not use DeepAR for serving API responses.

For project reporting, treat DeepAR as an experimental/auxiliary training artifact, not part of primary runtime inference.

## 4.5 Runtime Fallback Behavior

If model files are missing or fail to load:

- signal endpoint returns fallback HOLD-style outputs
- backtest service switches to random policy actions

This ensures API continuity in demo/dev setups.

## 4.6 Model File Conventions

- model directory resolved relative to backend process CWD
- default path in settings: `MODEL_PATH=./models`
- model binaries (`*.pt`, `*.zip`) are gitignored

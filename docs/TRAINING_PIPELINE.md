# Training and Data Pipeline

This document describes how historical data and models are produced.

## Data Download

File: `backend/training/download_data.py`

Why it exists

- Downloads historical OHLCV data for selected symbols.
- Writes CSV files into `backend/data/raw`.

Inputs

- Symbol list and date range in the script.

Outputs

- CSV files per symbol.

## LSTM Training

File: `backend/training/train_lstm.py`

Why it exists

- Trains the LSTM model used for price prediction.

Inputs

- CSV files in `backend/data/raw`.

Outputs

- Model checkpoint in `backend/models/lstm_final.pt`.

## PPO Training

File: `backend/training/train_ppo.py`

Why it exists

- Trains the PPO agent for trading actions.

Inputs

- `TradingEnv` data and reward shaping.

Outputs

- PPO model in `backend/models/ppo_trading_final.zip`.
- Checkpoints at 10k, 20k, 30k steps.

## DeepAR Training (Optional)

File: `backend/training/train_deepar.py`

Why it exists

- Alternative probabilistic forecasting model using DeepAR.

Inputs

- Historical OHLCV CSV data.

Outputs

- Model checkpoint.

**Note:** Requires `pytorch-forecasting` (not in `requirements.txt` — install separately).

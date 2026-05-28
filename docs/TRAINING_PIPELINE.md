# Training and Data Pipeline

This document describes how historical data and models are produced.

## Run Order

Run all training commands from `backend/`:

```bash
python training/download_data.py
python training/train_lstm.py
python training/train_ppo.py
```

## Typical Runtime (Local Machine)

- `download_data.py`: ~1-5 minutes (network dependent).
- `train_lstm.py`: ~8-30 minutes on CPU, usually faster on CUDA.
- `train_ppo.py`: ~10-45 minutes depending on timesteps and hardware.

These are practical ranges, not guarantees; runtime varies with CPU/GPU, background load, and data size.

## Data Download

File: `backend/training/download_data.py`

Why it exists

- Downloads historical OHLCV data for selected symbols.
- Writes CSV files into `backend/data/raw`.

Inputs

- Symbol list and date range in the script.

Outputs

- CSV files per symbol.
- Combined training dataset at `backend/data/training_data.csv`.

## LSTM Training

File: `backend/training/train_lstm.py`

Why it exists

- Trains the LSTM model used for price prediction.

Inputs

- CSV files in `backend/data/raw`.

Outputs

- Model checkpoint in `backend/models/lstm_final.pt`.
- Best-epoch snapshot in `backend/models/lstm_best.pt`.

Why it can take time

- Uses 30 epochs by default.
- Processes many rolling windows across all symbols.
- Trains recurrent layers (LSTM), which are slower than simple feed-forward models.

## PPO Training

File: `backend/training/train_ppo.py`

Why it exists

- Trains the PPO agent for trading actions.

Inputs

- `TradingEnv` data and reward shaping.

Outputs

- PPO model in `backend/models/ppo_trading_final.zip`.
- Checkpoints at 10k, 20k, 30k steps.

Why it can take time

- RL training is simulation-heavy (environment step + policy update loop).
- PPO runs multiple rollout/update cycles (`n_steps`, `n_epochs`, minibatches).
- Runtime scales roughly with `total_timesteps`.

## DeepAR Training (Optional)

File: `backend/training/train_deepar.py`

Why it exists

- Alternative probabilistic forecasting model using DeepAR.

Inputs

- Historical OHLCV CSV data.

Outputs

- Model checkpoint.

**Note:** Requires `pytorch-forecasting` (not in `requirements.txt` — install separately).

## Practical Speed-Ups

- Reduce epochs in `train_lstm.py` (`EPOCHS`), or reduce sequence length.
- Reduce PPO `total_timesteps` in `train_ppo.py` for faster iteration.
- Train on 1-3 symbols while iterating, then scale up for final runs.
- Use GPU (`torch.cuda.is_available()`) for LSTM when available.

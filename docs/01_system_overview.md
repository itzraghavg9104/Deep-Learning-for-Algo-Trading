# 1. System Overview

## 1.1 Project Purpose

This project is an AI-assisted algorithmic trading platform focused on Indian equities (NSE/BSE). It combines:

- live market data retrieval,
- technical analysis,
- deep learning (LSTM) price forecasting,
- reinforcement learning (PPO) trade action selection,
- trader behavior and risk profiling,
- backtesting and visualization through a Next.js frontend.

## 1.2 High-Level Architecture

The backend is organized as a layered design:

- Layer 1: data processing (`backend/app/layer1_data_processing/`)
- Layer 2: decision engine (`backend/app/layer2_decision/`)
- Trader behavior layer (`backend/app/trader_behavior/`)
- API/service layer (`backend/app/api/`, `backend/app/services/`)

Frontend (`frontend/`) provides authenticated dashboards, profile workflows, backtesting UI, and real-time market feed interaction.

## 1.3 Runtime Modes

### Demo mode (default)

- `DEMO_MODE=True` by default in `backend/app/config.py`
- Database engine is not initialized.
- Auth and profile/trade/backtest persistence use in-memory `DemoStore`.
- Data resets on backend restart.

### Persistent mode

- `DEMO_MODE=False`
- SQLAlchemy + PostgreSQL path is enabled.
- Optional Firebase auth/profile storage can be enabled via `FIREBASE_AUTH_ENABLED=True`.

## 1.4 Main Repository Structure

- `backend/app/main.py`: FastAPI app entrypoint and router wiring
- `backend/app/config.py`: runtime configuration and production safety checks
- `backend/app/api/routes/`: REST API routes
- `backend/app/api/websocket.py`: live price WebSocket
- `backend/app/layer1_data_processing/`: market + indicators + state builder
- `backend/app/layer2_decision/`: trading environment + PPO + reward logic
- `backend/app/services/`: prediction, backtest, storage services
- `backend/training/`: data download + model training scripts
- `frontend/src/app/`: Next.js app router pages
- `frontend/src/lib/`: API client, auth store, websocket hook

## 1.5 Data and Market Conventions

- Indian market symbols are normalized through `normalize_symbol()`:
  - NSE: `.NS`
  - BSE: `.BO`
- Raw historical data is stored under `backend/data/raw/`.
- Combined training dataset is `backend/data/training_data.csv`.

## 1.6 What the System Currently Delivers

- Market data endpoint with indicators + OHLCV history.
- Signal endpoint with model-based output when models are available, fallback otherwise.
- Backtest run + retrieval endpoint.
- Risk and behavior profiling workflows.
- Profile preferences and trade evaluation feedback loop.
- Authenticated frontend pages and middleware route protection.
- WebSocket live price updates every ~30 seconds per subscription.

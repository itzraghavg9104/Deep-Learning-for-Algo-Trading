# System Architecture Overview

This document describes the high-level architecture of the AlgoTrading platform and how the major modules interact.

## High-Level Components

- Backend: FastAPI service in `backend/`.
- Frontend: Next.js App Router UI in `frontend/`.
- Data and Models: CSV data in `backend/data/` and models in `backend/models/`.
- Training: model training and data download scripts in `backend/training/`.

## Core Layers

- Layer 1 Data Processing
  - Files: `backend/app/layer1_data_processing/*`
  - Responsibilities: market data fetching, technical indicators, feature preparation.
- Layer 2 Decision
  - Files: `backend/app/layer2_decision/*`
  - Responsibilities: trading environment, PPO policy decisions, backtest logic.
- Trader Behavior
  - Files: `backend/app/trader_behavior/*`
  - Responsibilities: risk profiling, position sizing, breakeven tracking.

## Runtime Overview

- Frontend calls REST endpoints for signals, profile, risk assessment, and backtest.
- Backend fetches market data, computes indicators, and runs model inference.
- WebSocket sends live price updates to the dashboard.
- Demo mode can replace database persistence for auth and backtests.

## End-to-End Request Flow

1. User logs in from the frontend (`/auth/login`).
2. Frontend stores token in localStorage (`auth-storage`) and cookie (`auth_token`).
3. Protected routes are enforced in `frontend/src/middleware.ts`.
4. Frontend sends API requests to `backend/app/api/routes/*` with bearer token.
5. Backend validates token in `get_current_user`:
   - Firebase token if `FIREBASE_AUTH_ENABLED=true`.
   - JWT token + demo/DB lookup otherwise.
6. Trading requests trigger:
   - Market data fetch (`layer1_data_processing/market_data.py`)
   - Indicators (`technical_indicators.py`)
   - Model inference (`services/prediction_service.py`) with fallback when models are unavailable.
7. Backtest/profile updates are stored in Firestore (if enabled) or in-memory demo store.

## Security Runtime Guard

- On backend startup, production config is validated.
- App refuses to start in production if:
  - `DEBUG=true`
  - `DEMO_MODE=true`
  - default `SECRET_KEY` or `JWT_SECRET` are still in use

## Data Stores

- Postgres is planned for user, trade, backtest, and risk profile persistence.
- Redis is planned for caching and live price stream buffering.
- Demo mode replaces both with in-memory storage for local demos.

## File-Level Entry Points

- Backend entry: `backend/app/main.py`
- Frontend entry: `frontend/src/app/layout.tsx`
- Frontend pages: `frontend/src/app/*`

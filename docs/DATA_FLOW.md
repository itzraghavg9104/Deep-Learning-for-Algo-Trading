# End-to-End Data Flow

This document explains how data moves through the system for the main user flows.

## Authentication Flow

1. Frontend submits login form from `frontend/src/app/auth/login/page.tsx`.
2. Frontend calls `POST /api/v1/auth/login` via `frontend/src/lib/api.ts`.
3. Backend validates user credentials in `backend/app/api/routes/auth.py`.
4. Demo mode uses `backend/app/services/demo_store.py` for user lookup.
5. Backend returns JWT token.
6. Frontend stores token in Zustand `frontend/src/lib/auth-store.ts` and sets `auth_token` cookie.

Inputs

- Login form: email, password.
- API: `username`, `password` (form-encoded).

Outputs

- Access token.
- Authenticated session state.

## Watchlist Signals Flow

1. Dashboard loads `frontend/src/app/dashboard/page.tsx`.
2. Frontend calls `GET /api/v1/trading/watchlist`.
3. Backend fetches market data via `backend/app/layer1_data_processing/market_data.py`.
4. Prediction service `backend/app/services/prediction_service.py` loads LSTM and PPO models if available.
5. Backend returns signal list with price, change percent, action, confidence.
6. Frontend renders `SignalCard` components.

Inputs

- Watchlist endpoint has no required parameters.

Outputs

- Array of signals with price and action.

## Signal Detail Modal Flow

1. User clicks a signal card.
2. Frontend calls `GET /api/v1/trading/market/{symbol}?period=3mo`.
3. Backend fetches OHLCV history and indicators.
4. Backend returns market history array and indicator values.
5. Frontend renders `PriceChart` and `TechnicalIndicators`.

Inputs

- Symbol string and optional period.

Outputs

- Market data including `history` array and indicators.

## WebSocket Live Prices Flow

1. Frontend opens WebSocket to `/api/v1/ws/prices`.
2. Frontend sends `{"action":"subscribe","symbols":[...]}`.
3. Backend loops every 30s and sends price updates.
4. Frontend updates signal card prices and flashes.

Inputs

- Symbol list for subscription.

Outputs

- Price updates array with symbol, price, change percent.

## Risk Assessment Flow

1. User completes questionnaire `frontend/src/components/forms/RiskQuestionnaire.tsx`.
2. Frontend calls `POST /api/v1/profile/risk-assessment`.
3. Backend computes risk score in `backend/app/trader_behavior/risk_profiler.py`.
4. Backend returns score, category, and recommendations.

Inputs

- List of integers in range 1 to 4.

Outputs

- Risk tolerance score and derived recommendations.

## Backtest Flow

1. User submits `BacktestConfig` on `frontend/src/app/backtest/page.tsx`.
2. Frontend calls `POST /api/v1/backtest/run`.
3. Backend loads CSV data in `backend/data/raw`.
4. `TradingEnv` in `backend/app/layer2_decision/trading_env.py` simulates trades.
5. PPO model in `backend/models/` is used if present.
6. Backend returns metrics and equity curve.

Inputs

- Symbol, date range, initial capital, risk tolerance.

Outputs

- Backtest metrics, trades list, equity curve.

## Profile Preferences Flow

1. User updates preferences in `frontend/src/app/profile/page.tsx`.
2. Frontend calls `PUT /api/v1/profile/preferences`.
3. Backend returns the updated preferences object.

Inputs

- Preferences object with `use_sentiment`, `preferred_timeframe`, `symbols`.

Outputs

- Updated preferences.

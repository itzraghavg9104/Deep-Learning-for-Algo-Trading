# Backend Reference (Detailed)

This document explains each backend module, why it exists, and how inputs and outputs flow through it.

## Entry Point

`backend/app/main.py`

- Creates the FastAPI app.
- Adds CORS middleware.
- Registers routers:
  - Trading: `/api/v1/trading`
  - Backtest: `/api/v1/backtest`
  - Profile: `/api/v1/profile`
  - Auth: `/api/v1/auth`
  - WebSocket: `/api/v1/ws/prices`
- Health endpoints: `/` and `/health`.

Inputs

- HTTP requests from frontend and external clients.

Outputs

- JSON responses and WebSocket events.

## Configuration

`backend/app/config.py`

- Reads environment variables from `.env`.
- Key settings include:
  - `DEMO_MODE` for in-memory storage.
  - `DATABASE_URL` and `REDIS_URL`.
  - JWT settings.
  - Model paths.

## API Routes

`backend/app/api/routes/auth.py`

- `POST /register` creates a user.
- `POST /login` returns JWT token.
- `GET /me` returns authenticated user.

Inputs

- JSON user registration payload.
- Form-encoded login payload.
- JWT token via `Authorization` header.

Outputs

- User object or JWT token.

Demo Mode

- Uses `backend/app/services/demo_store.py`.
- In demo mode, no Postgres connection is used.

`backend/app/api/routes/trading.py`

- `GET /signals/{symbol}` returns model signal and indicators.
- `GET /market/{symbol}` returns current price, change percent, indicators, and `history`.
- `GET /watchlist` returns a list of signals for tracked symbols.

Inputs

- Symbol string and optional query params.

Outputs

- Signal responses with action, confidence, and indicators.

`backend/app/api/routes/backtest.py`

- `POST /run` runs backtest in `BacktestService`.
- `GET /{backtest_id}` is not implemented.

Inputs

- Symbol, date range, initial capital, risk tolerance.

Outputs

- Metrics, trades, equity curve.

Demo Mode

- Persists backtest summary in `demo_store`.

`backend/app/api/routes/profile.py`

- `POST /risk-assessment` returns risk profile.
- `GET /` returns demo profile.
- `PUT /preferences` echoes updated preferences.

Inputs

- Questionnaire answers.
- Preferences object.

Outputs

- Risk profile and preferences.

`backend/app/api/websocket.py`

- `WS /api/v1/ws/prices` provides live updates.
- Accepts messages:
  - `{"action": "subscribe", "symbols": ["RELIANCE.NS"]}`
  - `{"action": "unsubscribe", "symbols": ["RELIANCE.NS"]}`
  - `{"action": "set", "symbols": ["RELIANCE.NS"]}`
  - `{"action": "ping"}`
- Emits messages:
  - `{"type": "prices", "data": [{symbol, price, change_pct, timestamp}]}`.

## Services

`backend/app/services/prediction_service.py`

- Loads LSTM model from `backend/models/lstm_final.pt`.
- Optionally loads PPO model from `backend/models/ppo_trading_final.zip`.
- Produces:
  - LSTM predicted price and change percent.
  - PPO action override when available.

Inputs

- Symbol string.
- Optional risk tolerance.

Outputs

- Dict with action, confidence, predicted price, and metadata.

`backend/app/services/backtest_service.py`

- Loads CSV data from `backend/data/raw`.
- Runs PPO-based simulation in `TradingEnv`.
- Returns metrics and trade list.

Inputs

- Symbol.
- Date range.
- Initial capital.
- Risk tolerance.

Outputs

- Backtest metrics and equity curve.

`backend/app/services/demo_store.py`

- In-memory users and backtests.
- Used in demo mode for auth and backtest persistence.

Inputs

- User creation and backtest results.

Outputs

- Demo user and backtest objects.

## Layer 1 Data Processing

`backend/app/layer1_data_processing/market_data.py`

- Fetches OHLCV data using `yfinance`.
- Normalizes symbols.
- Exposes async wrapper `get_market_data`.

Inputs

- Symbol, period, interval.

Outputs

- DataFrame of OHLCV data.

`backend/app/layer1_data_processing/technical_indicators.py`

- Computes indicators with `pandas_ta` or fallback.
- Produces SMA, EMA, MACD, RSI, Bollinger, ATR, etc.

Inputs

- OHLCV DataFrame.

Outputs

- Indicator dictionary.

## Layer 2 Decision

`backend/app/layer2_decision/trading_env.py`

- Trading environment used by PPO and backtest.
- Tracks portfolio, trades, equity curve.

Inputs

- OHLCV data.
- Initial capital, risk tolerance.

Outputs

- Step rewards, trade history, episode metrics.

## Trader Behavior

`backend/app/trader_behavior/risk_profiler.py`

- Converts questionnaire answers to a risk tolerance score.
- Produces risk category and descriptive text.

Inputs

- List of integer answers from 1 to 4.

Outputs

- `risk_tolerance`, `category`, `description`.

`backend/app/trader_behavior/position_sizer.py`

- Calculates position size based on confidence and risk.

`backend/app/trader_behavior/breakeven_tracker.py`

- Computes breakeven prices for positions.

## Models and Data

- `backend/models/` holds trained model artifacts.
- `backend/data/raw/` contains CSV data for backtesting.

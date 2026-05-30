# 2. Backend Architecture

## 2.1 Application Entry

`backend/app/main.py` configures:

- FastAPI metadata and docs endpoints (`/docs`, `/redoc`)
- CORS from `settings.CORS_ORIGINS`
- API router prefixes:
  - `/api/v1/trading`
  - `/api/v1/backtest`
  - `/api/v1/profile`
  - `/api/v1/auth`
  - WebSocket under `/api/v1/ws/prices`
- startup safety validation (`production_security_issues()`)

## 2.2 Configuration Model

`backend/app/config.py` (`Settings`) defines:

- app toggles: `APP_ENV`, `DEBUG`, `DEMO_MODE`
- auth: JWT config, Firebase config
- infra: DB URL, Redis URL
- ML paths: `MODEL_PATH`, model filenames

Production guardrails fail startup if:

- `APP_ENV=production` with `DEBUG=True`
- `APP_ENV=production` with `DEMO_MODE=True`
- default insecure secrets remain unchanged

## 2.3 API Route Responsibilities

### Auth (`api/routes/auth.py`)

- `POST /register`
- `POST /login`
- `GET /me`

Behavior:

- Demo mode auto-creates users on login and skips password validation.
- JWT subject is email (`sub`).
- Firebase mode verifies ID token, then upserts user in Firestore path.

### Trading (`api/routes/trading.py`)

- `GET /signals/{symbol}`
- `GET /market/{symbol}`
- `GET /watchlist`

Behavior:

- Pulls market data from yfinance.
- Computes technical indicators.
- Uses prediction service when available.
- Falls back to rule-based HOLD behavior when model prediction is unavailable.

### Backtest (`api/routes/backtest.py`)

- `POST /run`
- `GET /{backtest_id}`

Behavior:

- Runs simulation through `BacktestService`.
- Persists in demo store or DB service based on mode.
- Current implementation uses placeholder `user_id = 1` (auth linkage not completed).

### Profile (`api/routes/profile.py`)

- `POST /risk-assessment`
- `POST /behavior-assessment`
- `GET /`
- `PUT /preferences`
- `GET /trades`
- `POST /trades/evaluate`

Behavior:

- Computes risk tolerance and category.
- Converts richer questionnaire inputs into normalized behavior vectors.
- Stores/returns preferences and behavior profile.
- Evaluates compliance between planned vs executed trade behavior.

## 2.4 WebSocket Runtime

`api/websocket.py` endpoint `/ws/prices`:

- accepts actions: `subscribe`, `unsubscribe`, `set`, `ping`
- tracks per-connection symbol subscriptions
- sends batched `type: "prices"` updates every ~30s
- normalizes incoming symbols before subscription

## 2.5 Layer 1 - Data Processing

### `market_data.py`

- symbol normalization (`normalize_symbol`)
- sync and async yfinance fetch wrappers
- stock info lookup
- predefined top NSE universe map

### `technical_indicators.py`

- computes indicator dictionary from OHLCV
- uses `pandas_ta` when available, else basic fallback calculations
- includes serialization hardening for numpy/pandas scalar outputs

### `state_builder.py`

- merges market features, indicators, behavior traits, portfolio state, optional sentiment
- outputs normalized float32 vector for policy input

## 2.6 Layer 2 - Decision Engine

### `trading_env.py`

- Gymnasium environment with action space:
  - `0 = HOLD`, `1 = BUY`, `2 = SELL`
- simulates portfolio evolution and transaction cost impact
- computes step reward through reward module

### `reward_function.py`

- Sharpe and Sortino calculations
- step reward combining P&L, transaction cost, and risk penalty
- `RewardTracker` computes episode metrics (return, drawdown, win rate, etc.)

### `ppo_agent.py`

- SB3 PPO wrapper for create/train/predict/save/load
- includes risk-tolerance-adjusted hyperparameter helper (`create_agent`)

## 2.7 Services Layer

### `prediction_service.py`

- loads LSTM model from `./models/lstm_final.pt`
- can infer PPO action by loading `./models/ppo_trading_final.zip`
- returns combined result (`LSTM+PPO`) when both are available

### `backtest_service.py`

- reads historical CSV from `backend/data/raw`
- runs trade simulation through env + PPO model (or random fallback)
- returns metrics, trades, and equity curve

### Storage Services

- `demo_store.py`: in-memory user/profile/backtest/trades
- `db_service.py`: SQLAlchemy CRUD operations
- `firestore_store.py`: Firestore persistence with demo fallback
- `firebase_admin_service.py`: Firebase Admin init + token verification

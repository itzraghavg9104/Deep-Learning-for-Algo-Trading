# API Reference (Detailed)

All endpoints are under `/api/v1`.

## Authentication

`POST /auth/register`

Inputs

- JSON body: `{ "email": "user@example.com", "password": "..." }`

Outputs

- User object: `{ "id": 1, "email": "...", "is_active": true }`

`POST /auth/login`

Inputs

- Form body: `username`, `password`

Outputs

- `{ "access_token": "...", "token_type": "bearer" }`

`GET /auth/me`

Inputs

- Header: `Authorization: Bearer <token>`

Outputs

- User object.

## Trading

`GET /trading/signals/{symbol}`

Inputs

- Path: `symbol`
- Query: `use_sentiment`, `use_model`

Outputs

- `{ symbol, timestamp, action, confidence, prediction, indicators }`

`GET /trading/market/{symbol}`

Inputs

- Path: `symbol`
- Query: `period`

Outputs

- `{ symbol, current_price, change_pct, volume, indicators, history }`
- `history` is an array of `{ timestamp, open, high, low, close, volume }`

`GET /trading/watchlist`

Outputs

- `{ signals: [...], model_available: true|false }`

## Backtest

`POST /backtest/run`

Inputs

- JSON body:
  - `symbol`
  - `start_date`
  - `end_date`
  - `initial_capital`
  - `risk_tolerance`

Outputs

- Backtest metrics, trades list, equity curve.

`GET /backtest/{backtest_id}`

- Returns backtest results by ID (stored in demo_store in DEMO_MODE).

## Profile

`POST /profile/risk-assessment`

Inputs

- JSON body: `{ "answers": [1,2,3,4, ...] }`

Outputs

- `{ risk_tolerance, category, description, recommendations }`

`GET /profile`

Outputs

- Demo profile payload.

`PUT /profile/preferences`

Inputs

- JSON body: `{ use_sentiment, preferred_timeframe, symbols }`

Outputs

- Updated preferences echo.

`GET /profile/trades`

Inputs

- Auth required.

Outputs

- `{ trades: [...], total_pnl, win_rate }`

## WebSocket

`WS /api/v1/ws/prices`

Inputs

- Subscribe: `{ "action": "subscribe", "symbols": ["RELIANCE.NS"] }`
- Unsubscribe: `{ "action": "unsubscribe", "symbols": ["RELIANCE.NS"] }`
- Replace set: `{ "action": "set", "symbols": ["RELIANCE.NS"] }`
- Ping: `{ "action": "ping" }`

Outputs

- `{ "type": "prices", "data": [{ "symbol": "...", "price": 123.4, "change_pct": 0.12, "timestamp": "..." }] }`

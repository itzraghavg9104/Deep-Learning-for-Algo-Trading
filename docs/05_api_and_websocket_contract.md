# 5. API and WebSocket Contract

Base prefix: `/api/v1`

## 5.1 Auth

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

Notes:

- Demo mode login is permissive and auto-provisioning.
- Firebase mode expects bearer token verification on backend.

## 5.2 Trading

- `GET /trading/signals/{symbol}`
  - query: `use_sentiment`, `use_model`
  - returns action, confidence, prediction payload, indicators
- `GET /trading/market/{symbol}`
  - query: `period`
  - returns current price, change %, indicators, OHLCV history
- `GET /trading/watchlist`
  - returns watchlist signal summary and model availability flag

## 5.3 Backtest

- `POST /backtest/run`
  - body: `symbol`, `start_date`, `end_date`, `initial_capital`, `risk_tolerance`
  - returns core performance metrics and trade/equity details
- `GET /backtest/{backtest_id}`
  - currently implemented for demo-store IDs

## 5.4 Profile and Behavior

- `POST /profile/risk-assessment`
- `POST /profile/behavior-assessment`
- `GET /profile/`
- `PUT /profile/preferences`
- `GET /profile/trades`
- `POST /profile/trades/evaluate`

Behavior assessment requires at least 24 question inputs and produces:

- normalized behavior array
- risk profile
- recommendations
- timestamped behavior payload

Trade evaluation returns:

- compliance score
- violation list
- feedback deltas between planned vs executed metrics

## 5.5 Health and Root

- `GET /` returns API metadata
- `GET /health` returns service health payload

## 5.6 WebSocket

Endpoint: `/api/v1/ws/prices`

Incoming action payload examples:

- `{ "action": "subscribe", "symbols": ["RELIANCE", "TCS.NS"] }`
- `{ "action": "unsubscribe", "symbols": ["RELIANCE.NS"] }`
- `{ "action": "set", "symbols": ["INFY.NS"] }`
- `{ "action": "ping" }`

Outgoing payload types:

- `{"type":"prices","data":[...]}` every ~30 seconds for active subscriptions
- `{"type":"pong"}` in response to ping

# 5. API and WebSocket Contract

Base URL: `/api/v1`

## 5.1 Authentication

### POST /auth/register

Register a new user account.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response (200):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "is_active": true
}
```

**Errors:**
- `400` — Email already registered, or "Register via Firebase client SDK when FIREBASE_AUTH_ENABLED=true"
- `422` — Validation error (invalid email, etc.)

**Mode-specific behavior:**
- **Firebase mode**: Always returns 400 — registration must happen via Firebase SDK client-side
- **Demo mode**: Stores in DemoStore (in-memory), auto-incrementing integer ID
- **Normal mode**: Stores in PostgreSQL via DBService, UUID or serial ID

---

### POST /auth/login

Authenticate and receive JWT token.

**Request:** (application/x-www-form-urlencoded)
```
username: user@example.com
password: securepassword123
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

**Errors:**
- `400` — "Login via Firebase client SDK when FIREBASE_AUTH_ENABLED=true"
- `401` — Incorrect email or password

**Mode-specific behavior:**
- **Firebase mode**: Always returns 400 — login must happen via Firebase SDK client-side
- **Demo mode**: Auto-creates user if not found, skips password verification entirely, returns JWT with `sub=email`
- **Normal mode**: Verifies password via bcrypt, queries PostgreSQL, returns JWT with `sub=email`

**JWT Claims:**
```json
{
  "sub": "user@example.com",
  "exp": 1717056000
}
```
- Algorithm: HS256
- Default expiry: 24 hours (configurable via `JWT_EXPIRY_HOURS`)
- Signed with `JWT_SECRET`

---

### GET /auth/me

Get current authenticated user profile.

**Headers:** `Authorization: Bearer <token>`

**Response (200):**
```json
{
  "id": "firebase-uid-123",
  "email": "user@example.com",
  "is_active": true
}
```

**Errors:**
- `401` — Invalid or expired token, or Firebase token verification failure

**Mode-specific behavior:**
- **Firebase mode**: Verifies Firebase ID token via `verify_firebase_token()`, upserts user in Firestore, returns Firestore user data
- **Demo mode**: Decodes JWT, extracts `sub` (email), auto-provisions demo user if not found, returns demo user
- **Normal mode**: Decodes JWT, queries PostgreSQL via DBService

---

## 5.2 Trading

### GET /trading/signals/{symbol}

Get AI-powered trading signal for a stock symbol.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| symbol | str | Stock symbol (e.g., `RELIANCE.NS`, `TCS.NS`) |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| use_sentiment | bool | false | Include sentiment analysis (not yet implemented) |
| use_model | bool | true | Use trained LSTM/PPO models |
| user_id | str | null | User ID for user-specific PPO model |
| risk_tolerance | float | 0.5 | Risk tolerance (0.0–1.0) |
| capital_per_trade_pref | float | 0.1 | Capital allocation preference |
| tp_sl_pref | float | 0.4 | Take-profit/stop-loss ratio |
| max_drawdown_pref | float | 0.2 | Max drawdown sensitivity |
| cooldown_pref | float | 0.1 | Post-loss cooldown period |

**Response (200):**
```json
{
  "symbol": "RELIANCE.NS",
  "timestamp": "2026-05-30T10:30:00",
  "action": "BUY",
  "confidence": 0.78,
  "prediction": {
    "current_price": 2850.50,
    "predicted_price": 2910.25,
    "price_change": 59.75,
    "change_pct": 2.10,
    "model": "LSTM+PPO"
  },
  "indicators": {
    "rsi_14": 62.5,
    "macd_line": 12.3,
    "macd_signal": 8.1,
    "macd_histogram": 4.2,
    "sma_20": 2800.0,
    "sma_50": 2750.0,
    "bb_upper": 2950.0,
    "bb_lower": 2700.0,
    "bb_pct_b": 0.65,
    "atr_14": 45.0,
    "atr_pct": 1.58,
    "adx": 28.0,
    "stoch_k": 70.0,
    "stoch_d": 65.0,
    "cci_20": 100.0,
    "williams_r": -30.0,
    "roc_10": 2.5,
    "obv": 1500000,
    "mfi_14": 55.0,
    "volume_ratio": 1.2,
    "above_sma_20": true,
    "above_sma_50": true,
    "trend_bullish": true
  }
}
```

**Errors:**
- `404` — No data found for symbol
- `500` — Internal error (model failure, yfinance error)

**Action values:**
| Action | Code | Meaning |
|--------|------|---------|
| HOLD BUY | 0 | Maintain/increase long position |
| HOLD SELL | 1 | Reduce/exit long position |
| BUY | 2 | Open new long position |
| SELL | 3 | Close existing position |
| IDLE | 4 | Do nothing |

**Model field values:**
| Value | Meaning |
|-------|---------|
| LSTM | Only LSTM prediction available |
| PPO | Only PPO action available |
| LSTM+PPO | Combined prediction (PPO action overrides LSTM) |
| fallback | No models available, rule-based signal |

**Fallback behavior:**
When models are unavailable (not loaded, missing files, or errors):
- Action: IDLE
- Confidence: 0.5
- Model: "fallback"
- predicted_price = current_price, change_pct = 0.0

---

### GET /trading/market/{symbol}

Get market data and technical indicators for a symbol.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| symbol | str | Stock symbol (e.g., `RELIANCE.NS`) |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| period | str | "1mo" | Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y |

**Response (200):**
```json
{
  "symbol": "RELIANCE.NS",
  "current_price": 2850.50,
  "change_pct": 0.75,
  "volume": 1250000,
  "indicators": {
    "rsi_14": 62.5,
    ...
  },
  "history": [
    {
      "timestamp": "2026-05-01T00:00:00",
      "open": 2800.0,
      "high": 2820.0,
      "low": 2785.0,
      "close": 2810.0,
      "volume": 1000000
    },
    ...
  ]
}
```

**History limit:** Last 180 data points returned (approximately 9 months of daily data).

---

### GET /trading/watchlist

Get signals for 20 major NIFTY 50 stocks plus 3 index indicators.

**Response (200):**
```json
{
  "signals": [
    {
      "symbol": "RELIANCE.NS",
      "price": 2850.50,
      "predicted_price": 2910.25,
      "target_price": 2910.25,
      "change_pct": 0.75,
      "action": "BUY",
      "confidence": 0.78,
      "model": "LSTM+PPO"
    },
    ...
  ],
  "top20": [
    ...same signals array...
  ],
  "indices": [
    {
      "label": "NIFTY 50",
      "symbol": "^NSEI",
      "price": 22500.00,
      "change_pct": 0.50
    },
    {
      "label": "NIFTY MIDCAP 150",
      "symbol": "NIFTYMIDCAP150.NS",
      "price": 18500.00,
      "change_pct": 0.60
    },
    {
      "label": "NIFTY SMALLCAP 250",
      "symbol": "NIFTYSMLCAP250.NS",
      "price": 15500.00,
      "change_pct": 0.80
    }
  ],
  "model_available": true
}
```

**Watchlist Symbols (20):**
RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, BHARTIARTL.NS, ITC.NS, KOTAKBANK.NS, LT.NS, HINDUNILVR.NS, AXISBANK.NS, BAJFINANCE.NS, MARUTI.NS, ASIANPAINT.NS, WIPRO.NS, HCLTECH.NS, SUNPHARMA.NS, TITAN.NS, TATAMOTORS.NS

---

## 5.3 Backtest

### POST /backtest/run

Execute a historical backtest simulation.

**Request:**
```json
{
  "symbol": "RELIANCE.NS",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000.0,
  "risk_tolerance": 0.5
}
```

**Response (200):**
```json
{
  "id": "backtest-123",
  "symbol": "RELIANCE.NS",
  "total_return": 12.5,
  "sharpe_ratio": 1.45,
  "max_drawdown": -8.3,
  "win_rate": 0.55,
  "profit_factor": 1.8,
  "total_trades": 45,
  "final_value": 112500.0,
  "trades": [
    {
      "step": 25,
      "action": "BUY",
      "price": 2650.0,
      "shares": 35
    },
    {
      "step": 42,
      "action": "SELL",
      "price": 2780.0,
      "shares": 35,
      "pnl": 4550.0
    }
  ],
  "equity_curve": [100000, 100500, 101200, ...]
}
```

**Errors:**
- `400` — Insufficient data for the specified date range
- `404` — No data file found for symbol
- `422` — Validation error (invalid dates, negative capital, etc.)

**Backend process:**
1. Loads CSV from `data/raw/{symbol}.csv` (strips `.NS` suffix)
2. Filters by date range, requires ≥50 data points
3. Creates TradingEnv with the filtered data
4. Runs PPO model (or random actions if model unavailable)
5. Returns metrics from `env.get_episode_metrics()`

---

### GET /backtest/{backtest_id}

Retrieve a previously run backtest result.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| backtest_id | str | ID returned from POST /backtest/run |

**Response (200):**
Same structure as POST /backtest/run response.

**Errors:**
- `404` — Backtest not found in storage

---

## 5.4 Profile

### POST /profile/risk-assessment (Legacy)

Submit a simple risk assessment questionnaire (6 questions, 1–4 scale).

**Request:**
```json
{
  "answers": [2, 3, 1, 4, 2, 3]
}
```
Minimum 4 answers required.

**Response (200):**
```json
{
  "risk_tolerance": 0.45,
  "category": "Moderate",
  "description": "Balanced approach to risk and reward",
  "recommendations": {
    "max_position_size": 0.12,
    "suggested_stop_loss": 0.10,
    "suggested_take_profit": 0.19
  }
}
```

**Risk Categories:**
| Range | Category | Description |
|-------|----------|-------------|
| < 0.25 | Conservative | Capital preservation |
| 0.25–0.50 | Moderate | Balanced growth |
| 0.50–0.75 | Growth | Growth-focused |
| > 0.75 | Aggressive | High return pursuit |

---

### POST /profile/behavior-assessment

Submit the full behavior assessment questionnaire (30 questions).

**Request:**
```json
{
  "answers": {
    "question_scores": [
      {"id": "q1", "score": 4},
      {"id": "q2", "score": 3},
      ...
    ],
    "capital_per_trade_pct": 10.0,
    "tp_sl_ratio": 3.0,
    "max_profit_close_pct": 20.0,
    "max_trades_per_day": 5,
    "avg_holding_time_min": 240,
    "post_loss_rest_min": 60,
    "max_drawdown_pct": 15.0,
    "loss_streak_reduce_pct": 25.0,
    "intraday_var_pct": 3.0,
    "entry_slippage_bps": 12.0,
    "session_consistency_score": 65.0,
    "news_buffer_min": 30.0,
    "partial_tp_frequency": 2.0,
    "breakeven_trigger_pct": 1.0,
    "breakeven_migration_time_min": 60.0
  }
}
```

At least 24 question scores required (via `question_scores` array or `q_*` keys).

**Response (200):**
```json
{
  "message": "Behavior profile computed (demo). Per-user model retraining started.",
  "behavior_profile": {
    "behavior_array": {
      "capital_per_trade_pct": 0.1,
      "tp_sl_ratio_preference": 0.375,
      "max_profit_close_pct": 0.2,
      "trade_frequency_window_score": 0.125,
      "avg_holding_time_score": 0.024,
      "post_loss_rest_min": 0.042,
      "drawdown_sensitivity": 0.15,
      "streak_risk_adjustment": 0.25,
      "intraday_var_limit": 0.03,
      "entry_slippage_tolerance_bps": 0.04,
      "time_of_day_performance_bias": 0.65,
      "news_proximity_buffer_min": 0.083,
      "partial_tp_preference": 0.5,
      "breakeven_migration_trigger_pct": 0.01,
      "breakeven_migration_time_min": 0.042
    },
    "raw_answers": {...},
    "question_count": 24,
    "risk_profile": {
      "risk_tolerance": 0.45,
      "category": "Moderate",
      "description": "Balanced approach...",
      "recommendations": {...}
    },
    "updated_at": "2026-05-30T10:30:00"
  },
  "model_training": {
    "started": true,
    "scope": "user-specific-ppo",
    "user_id": "user-123"
  }
}
```

**Side effects:**
- Triggers per-user PPO retraining (async, daemon thread)
- In Firebase mode: saves to Firestore under `users/{uid}/behavior_profiles/latest`

---

### GET /profile/

Get current user's full profile including risk assessment, preferences, and behavior data.

**Response (200):**
```json
{
  "id": "1",
  "email": "user@example.com",
  "risk_profile": {
    "tolerance": 0.45,
    "category": "Moderate"
  },
  "preferences": {
    "use_sentiment": false,
    "preferred_timeframe": "swing",
    "symbols": ["RELIANCE.NS", "TCS.NS"]
  },
  "behavior_profile": {
    "behavior_array": {...},
    "risk_profile": {...},
    "updated_at": "..."
  }
}
```

---

### PUT /profile/preferences

Update user trading preferences.

**Request:**
```json
{
  "use_sentiment": false,
  "preferred_timeframe": "swing",
  "symbols": ["RELIANCE.NS", "TCS.NS"]
}
```

**Response (200):**
```json
{
  "message": "Preferences updated",
  "preferences": {...}
}
```

---

### GET /profile/trades

Get trade history for the current user.

**Response (200):**
```json
{
  "trades": [
    {
      "symbol": "RELIANCE.NS",
      "action": "BUY",
      "price": 2750.0,
      "quantity": 10,
      "pnl": 500.0,
      "timestamp": "2026-05-15T10:30:00"
    }
  ],
  "total_pnl": 1500.0,
  "win_rate": 0.55
}
```

---

### GET /profile/model-training-status

Get the status of per-user PPO model training.

**Response (200):**
```json
{
  "status": "running",
  "message": "PPO training in progress.",
  "updated_at": "2026-05-30T10:30:00",
  "queued_update_pending": false
}
```

**Status values:** idle, queued, running, completed, failed

---

### POST /profile/trades/evaluate

Evaluate a planned vs executed trade for compliance.

**Request:**
```json
{
  "trade_id": "trade-123",
  "symbol": "RELIANCE.NS",
  "planned": {
    "capital_per_trade_pct": 10.0,
    "tp_sl_ratio": 3.0,
    "max_profit_close_pct": 20.0
  },
  "executed": {
    "capital_per_trade_pct": 15.0,
    "tp_sl_ratio": 2.5,
    "max_profit_close_pct": 12.0,
    "cooldown_respected": true
  },
  "pnl": 500.0,
  "pnl_pct": 2.5
}
```

**Response (200):**
```json
{
  "message": "Trade evaluated",
  "evaluation": {
    "trade_id": "trade-123",
    "symbol": "RELIANCE.NS",
    "compliance_score": 60.0,
    "violations": [
      "capital_limit_exceeded",
      "tp_sl_ratio_below_plan",
      "premature_profit_close"
    ],
    "planned": {...},
    "executed": {...},
    "feedback_loop": {
      "capital_pct_delta": 5.0,
      "tp_sl_ratio_delta": -0.5,
      "max_profit_close_pct_delta": -8.0
    },
    "pnl": 500.0,
    "pnl_pct": 2.5,
    "status": "warn",
    "evaluated_at": "2026-05-30T10:30:00"
  }
}
```

**Violations:**
| Violation | Condition |
|-----------|-----------|
| capital_limit_exceeded | executed_size > planned_size > 0 |
| tp_sl_ratio_below_plan | executed_tp_sl < planned_tp_sl (when planned > 0) |
| premature_profit_close | executed_max_profit < planned_max_profit (when planned > 0) |
| cooldown_violation | cooldown_respected = false |

**Status:** "pass" (no violations) or "warn" (any violations)

**Compliance Score:** `max(0, 100 - 20 × violations.length)`

---

## 5.5 System

### GET /

API root information.

**Response (200):**
```json
{
  "name": "Algo Trading System API",
  "version": "1.0.0",
  "target_market": "India (NSE/BSE)",
  "docs": "/docs"
}
```

---

### GET /health

Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "services": {
    "api": "running"
  }
}
```

---

## 5.6 WebSocket

### GET /api/v1/ws/prices (WebSocket Upgrade)

Real-time price streaming endpoint. Connection is upgraded from HTTP to WebSocket.

### Incoming Message Format

All messages should be JSON with an `action` field:

#### Subscribe
```json
{
  "action": "subscribe",
  "symbols": ["RELIANCE.NS", "TCS.NS", "INFY"]
}
```
Adds symbols to the connection's subscription set. Symbols are normalized via `normalize_symbol()`.

#### Unsubscribe
```json
{
  "action": "unsubscribe",
  "symbols": ["RELIANCE.NS"]
}
```
Removes symbols from the subscription set.

#### Set
```json
{
  "action": "set",
  "symbols": ["INFY.NS"]
}
```
Replaces the entire subscription set with the given symbols.

#### Ping
```json
{
  "action": "ping"
}
```
Health check — server responds with `{"type": "pong"}`.

### Outgoing Message Format

#### Price Update (broadcast ~every 30s)
```json
{
  "type": "prices",
  "data": [
    {
      "symbol": "RELIANCE.NS",
      "price": 2850.50,
      "change_pct": 0.75,
      "timestamp": "2026-05-30T10:30:00Z"
    },
    {
      "symbol": "TCS.NS",
      "price": 3850.00,
      "change_pct": -0.25,
      "timestamp": "2026-05-30T10:30:00Z"
    }
  ]
}
```

Broadcast occurs once every ~30 seconds (controlled by `asyncio.sleep(30)` loop). Only symbols that are actively subscribed by the connection are included.

#### Pong
```json
{
  "type": "pong"
}
```

### Connection Lifecycle
1. **Connect**: Client opens WebSocket to `/api/v1/ws/prices`
2. **Subscribe**: Client sends `{"action": "subscribe", "symbols": [...]}`
3. **Receive**: Server pushes price updates every ~30s
4. **Unsubscribe/Set**: Client can modify subscriptions at any time
5. **Ping**: Client can verify connection health
6. **Disconnect**: Connection cleanup occurs in `finally` block (no explicit close message)

### Implementation Notes
- Each WebSocket connection maintains its own `set()` of subscribed symbols
- Prices are sourced from yfinance (1d, 5m interval) — this is **simulated real-time**, not a live market feed
- Symbol normalization applies `.NS` suffix automatically via `normalize_symbol()`
- The server loop catches and logs exceptions per-symbol without crashing the connection
- No authentication required on the WebSocket endpoint

## 5.7 Error Response Format

All API errors follow FastAPI's standard format:

```json
{
  "detail": "Human-readable error message"
}
```

HTTP Status Codes Used:
| Code | Meaning | Common Scenarios |
|------|---------|-----------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Validation error, missing data, Firebase-mode auth |
| 401 | Unauthorized | Invalid/expired token |
| 404 | Not Found | Symbol not found, data not found |
| 422 | Validation Error | Pydantic schema validation failure |
| 500 | Internal Server Error | Unexpected exception, model failure |

## 5.8 Authentication Header

All protected endpoints require:
```
Authorization: Bearer <token>
```

- **Demo mode**: Token is JWT signed with `JWT_SECRET` using HS256
- **Firebase mode**: Token is Firebase ID token (JWT signed by Firebase)
- **Normal mode**: Token is JWT signed with `JWT_SECRET` using HS256

## 5.9 API Documentation

Interactive API documentation available when the backend is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These are auto-generated from FastAPI route decorators and Pydantic models.

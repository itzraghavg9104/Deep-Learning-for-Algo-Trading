# 2. Backend Architecture

## 2.1 Application Entrypoint (`main.py`)

```python
# FastAPI app instantiation
app = FastAPI(title="Algo Trading System", version="1.0.0")
```

### Startup Lifecycle

1. **CORS Middleware** — Allows origins from `settings.CORS_ORIGINS` (default: `localhost:3000`, `127.0.0.1:3000`)
2. **Router Registration** — Five routers mounted under `/api/v1`
3. **Security Validation** (startup event) — Calls `settings.production_security_issues()`; raises `RuntimeError` if unsafe config detected in production
4. **Model Bootstrap** (startup event) — Calls `ensure_models_ready()` via `asyncio.to_thread`: checks for LSTM and PPO model files, auto-trains if missing and `AUTO_TRAIN_IF_MISSING=True`

### Router Prefixes

| Prefix | Router File | Tags |
|--------|-------------|------|
| `/api/v1/trading` | `routes/trading.py` | Trading |
| `/api/v1/backtest` | `routes/backtest.py` | Backtest |
| `/api/v1/profile` | `routes/profile.py` | Profile |
| `/api/v1/auth` | `routes/auth.py` | Authentication |
| `/api/v1` | `api/websocket.py` | WebSocket |

### Health & Root Endpoints

- `GET /` — API metadata (name, version, target market, docs link)
- `GET /health` — Simple health status `{"status": "healthy", "services": {"api": "running"}}`

## 2.2 Configuration Model (`config.py`)

Defined as a `pydantic-settings` `Settings` class that reads from `.env` file:

### Application Settings
```
APP_ENV: str = "development"        # Runtime environment
DEBUG: bool = True                  # Debug mode
SECRET_KEY: str = "your-secret-key-change-in-production"
DEMO_MODE: bool = True              # No DB/Redis required
FIREBASE_AUTH_ENABLED: bool = False # Use Firebase Auth
```

### Database & Cache
```
DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/algotrading"
REDIS_URL: str = "redis://localhost:6379/0"
```

### CORS
```
CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
```

### Market Data
```
DEFAULT_MARKET: str = "NSE"
NEWS_API_KEY: str = ""
USE_SENTIMENT: bool = False
```

### ML Models
```
MODEL_PATH: str = "./models"
DEEPAR_MODEL: str = "deepar_v1.pt"
PPO_MODEL: str = "ppo_agent_v1.zip"
AUTO_TRAIN_IF_MISSING: bool = True    # Auto-train missing models at startup
AUTO_TRAIN_STRICT: bool = False       # Fail startup if bootstrap fails
```

### JWT
```
JWT_SECRET: str = "jwt-secret-change-in-production"
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRY_HOURS: int = 24
```

### Firebase
```
FIREBASE_PROJECT_ID: str = ""
FIREBASE_WEB_API_KEY: str = ""
FIREBASE_SERVICE_ACCOUNT_PATH: str = ""
FIRESTORE_DATABASE_ID: str = "(default)"
```

### Production Security Guard
The `production_security_issues()` method checks when `APP_ENV == "production"`:
- `DEBUG` must be `False`
- `DEMO_MODE` must be `False`
- `SECRET_KEY` must not be default
- `JWT_SECRET` must not be default

If any check fails, FastAPI startup raises `RuntimeError`.

## 2.3 API Routes — Deep Dive

### 2.3.1 Auth Routes (`routes/auth.py`)

**Dependencies:** `get_db()` (SQLAlchemy session, or `None` in demo mode), `oauth2_scheme` (OAuth2PasswordBearer)

**Password Handling:**
- `verify_password(plain, hashed)` — bcrypt `checkpw`
- `get_password_hash(password)` — bcrypt `hashpw` + `gensalt()`

**JWT Token:**
- `create_access_token(data, expires_delta)` — Creates HS256 JWT with `sub` claim
- Default expiry: 24 hours (from `JWT_EXPIRY_HOURS`)

**`get_current_user` Dependency:**
- **Firebase mode**: Verifies Firebase ID token via `verify_firebase_token()`, upserts user in Firestore, returns `UserResponse`
- **Demo mode**: Decodes JWT, extracts email from `sub`, auto-provisions user if not found
- **Normal mode**: Decodes JWT, queries database via `DBService`

**Endpoints:**

| Method | Path | Auth | Behavior |
|--------|------|------|----------|
| POST | `/register` | None | Firebase mode: raises 400 (register via Firebase SDK). Demo mode: auto-creates in DemoStore. Normal: DBService.create_user |
| POST | `/login` | None | Firebase mode: raises 400 (login via Firebase SDK). Demo mode: auto-creates user, skips password check, returns JWT. Normal: verifies password, returns JWT |
| GET | `/me` | Bearer JWT | Returns current user from token |

**Data Models:**
- `Token` — `access_token: str`, `token_type: str`
- `TokenData` — `email: Optional[str]`
- `UserCreate` — `email: str`, `password: str`
- `UserResponse` — `id: Union[int, str]`, `email: str`, `is_active: bool`

### 2.3.2 Trading Routes (`routes/trading.py`)

**Dependencies:** None (no auth on trading endpoints currently)

**Module-level:** Attempts to import `PredictionService`; sets `PREDICTION_AVAILABLE` flag

**Endpoints:**

#### `GET /trading/signals/{symbol}`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_sentiment` | bool | false | Include sentiment analysis (not implemented) |
| `use_model` | bool | true | Use trained LSTM model |
| `user_id` | str | null | User-specific PPO model ID |
| `risk_tolerance` | float | 0.5 | 0.0–1.0 risk preference |
| `capital_per_trade_pref` | float | 0.1 | Capital per trade (0–1) |
| `tp_sl_pref` | float | 0.4 | TP/SL ratio preference |
| `max_drawdown_pref` | float | 0.2 | Max drawdown sensitivity |
| `cooldown_pref` | float | 0.1 | Post-loss cooldown |

**Flow:**
1. Fetches 3 months of market data via `get_market_data(symbol, "3mo")`
2. Computes technical indicators via `compute_indicators(data)`
3. If `use_model=True` and prediction service available:
   - Calls `pred_service.predict(symbol, risk_tolerance, behavior_array, user_id)`
   - Returns combined LSTM+PPO prediction
4. Fallback: Returns IDLE action with 0.5 confidence

**Response:** `SignalResponse` — symbol, timestamp, action, confidence, prediction (current_price, predicted_price, price_change, change_pct, model), indicators

#### `GET /trading/market/{symbol}`

**Query Parameters:** `period` (default "1mo")

**Flow:**
1. Fetches market data
2. Computes indicators
3. Returns OHLCV history (last 180 points), current price, change %, volume, indicators

**Response:** `MarketDataResponse` — symbol, current_price, change_pct, volume, indicators, history[]

#### `GET /trading/watchlist`

**Flow:**
1. Iterates over 20 predefined NIFTY 50 symbols
2. For each symbol: fetches market data, runs prediction (if available), computes day change
3. Also returns 3 index signals (`^NSEI`, `NIFTYMIDCAP150.NS`, `NIFTYSMLCAP250.NS`)

**Response:** `{ signals[], top20[], indices[], model_available: bool }`

### 2.3.3 Backtest Routes (`routes/backtest.py`)

**Dependencies:** `get_current_user` (currently uses placeholder `user_id = "1"`)

#### `POST /backtest/run`

**Request Body:** `BacktestRequest` — symbol, start_date, end_date, initial_capital, risk_tolerance

**Flow:**
1. Instantiates `BacktestService`
2. Calls `service.run(symbol, start_date, end_date, initial_capital, risk_tolerance)`
3. Persists result (DemoStore or Firebase or DB)
4. Returns metrics

**Response:** `BacktestResultResponse` — id, symbol, total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades, final_value

#### `GET /backtest/{backtest_id}`

Returns stored backtest result by ID.

### 2.3.4 Profile Routes (`routes/profile.py`)

**Dependencies:** `get_current_user` on all endpoints

**Helper Functions:**
- `_extract_question_scores(raw_answers)` — Extracts 1–5 scores from `question_scores` list or `q_*` keys
- `_build_behavior_array(raw_answers)` — Maps 15 questionnaire answers to normalized (0–1) behavior vector
- `_to_float(value, default)` — Safe float conversion
- `_pct(name, raw_answers, default)` — Normalizes percentage (0–100 → 0–1)
- `_number(name, raw_answers, default, max_value)` — Normalizes value (0–max → 0–1)

#### `POST /profile/risk-assessment` (Legacy)

**Request:** `RiskAssessmentRequest` — `answers: List[int]` (min 4 required)

**Flow:**
1. `calculate_risk_score(answers)` → risk_tolerance (0–1)
2. `get_risk_category(risk_tolerance)` → category + description
3. Generates recommendations (max_position_size, stop_loss, take_profit)
4. Persists to Firestore if Firebase enabled

#### `POST /profile/behavior-assessment`

**Request:** `BehaviorAssessmentRequest` — `answers: Dict[str, Any]` (min 24 questions via `question_scores`)

**Flow:**
1. Extracts question scores (1–5 scale)
2. Normalizes to 1–4 for legacy risk scoring compatibility
3. Computes risk_tolerance, category, recommendations
4. Builds 15-dim behavior array via `_build_behavior_array()`
5. Persists profile (Firestore or DemoStore)
6. Triggers per-user PPO retraining via `trigger_user_retraining()`

**Behavior Array Fields (15 normalized dimensions):**

| Field | Source Key | Normalization | Range |
|-------|-----------|---------------|-------|
| capital_per_trade_pct | capital_per_trade_pct | /100 | 0–1 |
| tp_sl_ratio_preference | tp_sl_ratio | /8 | 0–1 |
| max_profit_close_pct | max_profit_close_pct | /100 | 0–1 |
| trade_frequency_window_score | max_trades_per_day | /40 | 0–1 |
| avg_holding_time_score | avg_holding_time_min | /10080 | 0–1 |
| post_loss_rest_min | post_loss_rest_min | /1440 | 0–1 |
| drawdown_sensitivity | max_drawdown_pct | /100 | 0–1 |
| streak_risk_adjustment | loss_streak_reduce_pct | /100 | 0–1 |
| intraday_var_limit | intraday_var_pct | /100 | 0–1 |
| entry_slippage_tolerance_bps | entry_slippage_bps | /300 | 0–1 |
| time_of_day_performance_bias | session_consistency_score | /100 | 0–1 |
| news_proximity_buffer_min | news_buffer_min | /360 | 0–1 |
| partial_tp_preference | partial_tp_frequency | /4 | 0–1 |
| breakeven_migration_trigger_pct | breakeven_trigger_pct | /100 | 0–1 |
| breakeven_migration_time_min | breakeven_migration_time_min | /1440 | 0–1 |

#### `GET /profile/`

Returns user profile including risk_profile, preferences, behavior_profile.

#### `PUT /profile/preferences`

**Request:** `UserPreferences` — `use_sentiment`, `preferred_timeframe`, `symbols`

#### `GET /profile/trades`

Returns trade history with `total_pnl` and `win_rate` aggregated.

#### `POST /profile/trades/evaluate`

**Request:** `TradeEvaluationRequest` — planned vs executed trade parameters

**Flow:**
1. Compares planned vs executed: capital_per_trade_pct, tp_sl_ratio, max_profit_close_pct, cooldown_respected
2. Generates violations list (capital_limit_exceeded, tp_sl_ratio_below_plan, premature_profit_close, cooldown_violation)
3. Computes compliance_score = max(0, 100 - 20 * violations.count)
4. Persists evaluation report

#### `GET /profile/model-training-status`

Returns per-user PPO training status: idle, queued, running, completed, failed.

## 2.4 WebSocket (`api/websocket.py`)

**Endpoint:** `GET /api/v1/ws/prices` (WebSocket upgrade)

### Accepted Incoming Actions

| Action | Payload | Behavior |
|--------|---------|----------|
| `subscribe` | `{symbols: ["RELIANCE.NS"]}` | Adds symbols to connection's subscription set |
| `unsubscribe` | `{symbols: ["TCS.NS"]}` | Removes symbols from subscription set |
| `set` | `{symbols: ["INFY.NS"]}` | Replaces entire subscription set |
| `ping` | `{}` | Responds with `{type: "pong"}` |

### Outgoing Messages

Every ~30 seconds, for each connection with active subscriptions:
```json
{
  "type": "prices",
  "data": [
    {
      "symbol": "RELIANCE.NS",
      "price": 2850.50,
      "change_pct": 0.75,
      "timestamp": "2026-05-30T10:30:00Z"
    }
  ]
}
```

**Implementation Details:**
- Each connection gets dedicated subscription tracking via a `set()` per WebSocket
- `build_price_update(symbol)` fetches 1-day, 5-minute interval data from yfinance
- Price pushes are simulated real-time (yfinance polled every 30s, not a live feed)
- Symbols are normalized via `normalize_symbol()` before subscription

## 2.5 Layer 1 — Data Processing

### 2.5.1 `market_data.py`

**Symbol Normalization:**
```python
def normalize_symbol(symbol: str) -> str:
    # Adds .NS suffix if missing for known NSE stocks
    # Preserves ^ prefix for indices
    # Preserves .BO suffix for BSE
```

**NSE Stock Universe:** Predefined dict `NSE_STOCKS` with 52 entries including symbol → company name mappings.

**Key Functions:**

| Function | Description | Async |
|----------|-------------|-------|
| `normalize_symbol(symbol)` | Adds proper suffix | No |
| `fetch_market_data_sync(symbol, period, interval)` | yfinance download, capitalizes columns | No |
| `get_market_data(symbol, period, interval)` | Wraps sync fetch in `run_in_executor` | Yes |
| `get_stock_info(symbol)` | yfinance info extraction | No |
| `get_nifty50_symbols()` | Returns `.NS` suffixed list | No |

### 2.5.2 `technical_indicators.py`

**Primary:** `compute_indicators(df)` — Entry point that uses pandas-ta if available, falls back to basic calculations.

**Indicators Computed (30+):**

**Trend Indicators:**
- SMA_20, SMA_50, SMA_200 (Simple Moving Averages)
- EMA_12, EMA_26 (Exponential Moving Averages)
- MACD line, MACD signal, MACD histogram
- ADX (Average Directional Index, 14-period)

**Momentum Indicators:**
- RSI_14 (Relative Strength Index, 14-period)
- Stochastic %K, %D (14,3,3)
- CCI_20 (Commodity Channel Index)
- Williams %R (14-period)
- ROC_10 (Rate of Change)

**Volatility Indicators:**
- BB_upper, BB_middle, BB_lower (Bollinger Bands, 20,2)
- BB_pct_B (Bollinger Band %B)
- ATR_14 (Average True Range)
- ATR_pct (ATR as % of close)

**Volume Indicators:**
- OBV (On-Balance Volume)
- MFI_14 (Money Flow Index)
- Volume_SMA_20
- Volume_Ratio (current / SMA)

**Composite Signals:**
- `above_sma_20`, `above_sma_50` (price relative to MA)
- `trend_bullish` (true if price > SMA_20 > SMA_50)

**Fallback:** `_compute_basic_indicators(df)` — Hand-rolled python implementations of SMA, EMA, RSI, Bollinger Bands, MACD, Volume Ratio when pandas-ta not installed.

**Serialization:** `_to_python_types(value)` — Recursively converts numpy/pandas types to native Python for JSON-safe output.

### 2.5.3 `state_builder.py`

**`build_state(market_data, trader_profile, portfolio_state, prediction, sentiment)`**

Builds a 30-dimensional state vector for the RL agent:

| Dimension Group | Components | Normalization |
|----------------|------------|---------------|
| Price Data | current price, change_pct | log/clip |
| DeepAR Predictions | pred_price_mean, pred_price_std, pred_change_pct, pred_confidence | log/clip |
| Technical Indicators | RSI, MACD, BB%, ATR%, ADX, Stoch_K, Stoch_D, CCI, Volume_Ratio, Williams_%R, OBV, MFI, SMA_20, SMA_50 | linear/clip |
| Trend Signals | above_sma_20 (binary), above_sma_50 (binary), trend_bullish (binary) | none |
| Trader Behavior | risk_tolerance, capital_per_trade_pct, tp_sl_pref, drawdown_sensitivity, post_loss_rest, timeframe | linear |
| Portfolio | position_pct, pnl_pct, cash_ratio, breakeven_distance | clip |
| Sentiment | sentiment_score (default neutral = 0.5) | none |

**`_dict_to_normalized_array(state_dict)`** — Applies normalization config per field.
**`get_state_dim()`** — Returns 30 (note: TradingEnv pads to 34).

## 2.6 Layer 2 — Decision Engine

### 2.6.1 `action_space.py`

Canonical action definitions:

```python
ACTION_HOLD_BUY  = 0  # Maintain/start long bias
ACTION_HOLD_SELL = 1  # Maintain/reduce position
ACTION_BUY       = 2  # Full buy
ACTION_SELL      = 3  # Full sell
ACTION_IDLE      = 4  # Do nothing

ACTION_LABELS = ["HOLD BUY", "HOLD SELL", "BUY", "SELL", "IDLE"]
```

### 2.6.2 `reward_function.py`

**`calculate_sharpe_ratio(returns, risk_free_rate=0.02)`**
- Annualized Sharpe = (mean(returns) - r_f) / std(returns) * sqrt(252)
- Returns 0 if std is 0 (no volatility)

**`calculate_sortino_ratio(returns, risk_free_rate=0.02)`**
- Uses downside deviation only (negative returns)
- Clipped to [-5, 5] range

**`calculate_reward(portfolio_values, window_size=20, reward_type="sharpe")`**
- Rolling window Sharpe/Sortino/returns calculation
- Scaled to [-1, 1] via tanh

**`calculate_step_reward(action, price_change_pct, position, risk_tolerance, transaction_cost)`**
- Step reward = position * price_change_pct (P&L contribution)
- Transaction cost penalty: -0.001 for any non-IDLE action
- Risk penalty applied based on position size and risk_tolerance

**`class RewardTracker`**
- Tracks portfolio values through episode
- `update(portfolio_value)` — records new value
- `get_metrics()` — returns dict with: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades, final_value, equity_curve

### 2.6.3 `trading_env.py`

**`class TradingEnv(gym.Env)`**

**Observation Space:** `Box(-inf, inf, shape=(34,), dtype=float32)`

**Action Space:** `Discrete(5)` — HOLD BUY, HOLD SELL, BUY, SELL, IDLE

**Constructor Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| df | required | OHLCV DataFrame |
| initial_capital | 100,000 | Starting capital |
| max_shares | 100 | Max shares per trade |
| transaction_cost | 0.001 | 0.1% per trade |
| risk_tolerance | 0.5 | 0–1 scale |
| window_size | 20 | Lookback for indicators |
| behavior_array | {} | User behavior preferences |

**`reset()`:**
- Starts at index `window_size`
- Resets cash, shares, avg_entry_price
- Initializes `RewardTracker`
- Returns observation (34-dim state)

**`step(action)`:**
1. Gets current and previous price
2. Calls `_execute_action(action, price)`
3. Calculates portfolio value
4. Updates `RewardTracker`
5. Calculates step reward
6. Advances `current_step`
7. Checks termination: `current_step >= len(df) - 1` or `portfolio_value < initial_capital * 0.5`
8. Returns (observation, reward, terminated, truncated, info)

**Action Execution (`_execute_action`):**

| Action | Behavior |
|--------|----------|
| HOLD_BUY | If flat: buy max_shares//4 shares |
| HOLD_SELL | If long: sell half of shares |
| BUY | Buy max affordable shares (up to max_shares) |
| SELL | Sell all shares |
| IDLE | Do nothing |

**State Construction (`_get_observation`):**
1. Gets window of data for indicator computation
2. Computes indicators on window
3. Builds 34-dim state vector:
   - Normalized current price (price/1000)
   - 9 indicators (RSI, MACD line, MACD signal, BB%B, ATR%, ADX, Stoch_K, CCI, Volume Ratio)
   - Portfolio state (cash_ratio, position_ratio, risk_tolerance)
   - 4 behavior fields (capital_per_trade_pct, tp_sl_ratio, drawdown_sensitivity, post_loss_rest)
   - P&L (pnl_pct)
   - Padded to 34 with zeros

**Episode Metrics:** Delegated to `RewardTracker.get_metrics()`

### 2.6.4 `ppo_agent.py`

**`class TradingAgent`**

**Constructor:** Accepts env or model_path
- If model_path: loads existing model via `PPO.load(path)`
- If env: creates new PPO via `_create_model()`

**PPO Configuration:**
```python
PPO(
    policy="MlpPolicy",
    policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)
```

**Key Methods:**
| Method | Description |
|--------|-------------|
| `train(total_timesteps, eval_env, eval_freq, save_path, log_path)` | Trains with checkpoint + eval callbacks |
| `predict(observation, deterministic)` | Returns (action, probabilities) |
| `get_action_with_confidence(observation)` | Returns action label, code, confidence, all probs |
| `save(path)` / `load(path)` | Serialization |

**`create_agent(env, risk_tolerance)` — Factory:**
- `learning_rate = 1e-4 + (risk_tolerance * 4e-4)` → range 1e-4 to 5e-4
- `ent_coef = 0.02 - (risk_tolerance * 0.015)` → range 0.02 to 0.005
- Conservative traders: more exploration (higher ent_coef), slower learning
- Aggressive traders: more exploitation, faster learning

## 2.7 Trader Behavior Layer

### 2.7.1 `risk_profiler.py`

**`calculate_risk_score(answers)`** — Normalizes 1–4 scale answers to 0–1 score

**`get_risk_category(risk_tolerance)`:**

| Range | Category | Description |
|-------|----------|-------------|
| < 0.25 | Conservative | Capital preservation, minimal risk |
| 0.25–0.50 | Moderate | Balanced growth, controlled risk |
| 0.50–0.75 | Growth | Growth-focused, moderate risk |
| > 0.75 | Aggressive | High return pursuit, high risk tolerance |

**Derived Parameters:**
- `get_position_size_multiplier(risk_tolerance)` → 0.5 to 1.5
- `get_stop_loss_percentage(risk_tolerance)` → 5% to 15%
- `get_take_profit_percentage(risk_tolerance)` → 10% to 30%

**`RISK_QUESTIONNAIRE`** — 6 questions with 4 options each, used by legacy risk assessment.

### 2.7.2 `position_sizer.py`

Three sizing strategies:

| Strategy | Formula | Best For |
|----------|---------|----------|
| `fixed_percentage_size()` | 5–20% of portfolio based on risk_tolerance | Simple allocation |
| `kelly_criterion_size()` | f* = (bp - q)/b, adjusted by 25–100% Kelly, capped at 25% | Optimal growth |
| `volatility_adjusted_size()` | Position = risk_per_trade / (2 × ATR%) × price | Volatility-aware |

**`calculate_position_size()`** — Unified dispatcher that selects strategy.

### 2.7.3 `breakeven_tracker.py`

**`@dataclass Position`** — symbol, quantity, avg_entry_price, total_cost, dates

**`class BreakevenTracker`:**
- `add_trade(symbol, action, price, quantity, timestamp)` — Updates weighted average entry price
- `get_position_info(symbol)` — Current position details
- `calculate_pnl(symbol, current_price)` — Unrealized P&L, distance to breakeven
- `get_all_positions()` — List all active positions

**Global Singleton:** `get_tracker()` returns module-level `BreakevenTracker` instance

## 2.8 Services Layer

### 2.8.1 `prediction_service.py`

**`class StockLSTM(nn.Module)`** — Matches training architecture:
- LSTM(2 layers, hidden=64, dropout=0.2) → FC(64→32, ReLU, Dropout) → FC(32→1)

**`class PredictionService`:**
- **Initialization:** Loads `lstm_final.pt` from `MODEL_PATH`
- Checkpoint format expected: `{"model_state_dict": ..., "config": {"seq_length": 30, "features": [...], "hidden_size": 64}}`
- Falls back to state_dict-only loading if config key missing

**`_get_lstm_prediction(symbol)` — LSTM Inference:**
1. Fetches 3 months daily data via yfinance
2. Lowercases columns, scales [open, high, low, close, volume] via MinMaxScaler
3. Creates sequence of length `seq_length` (default 30)
4. Runs model forward pass
5. Inverse-transforms prediction to price space
6. Determines action:
   - Predicted change > +1% → BUY (confidence: 0.5 + change_pct/10)
   - Predicted change < -1% → SELL (confidence: 0.5 + |change_pct|/10)
   - Otherwise → IDLE (confidence: 0.5)

**`get_ppo_signal(symbol, risk_tolerance, behavior_array, user_id)` — PPO Inference:**
1. Fetches 1 month daily data
2. Creates temporary TradingEnv
3. Resets env to get initial observation
4. Loads PPO model (user-specific if exists, else global)
5. Predicts action deterministically
6. Returns action label, confidence=0.8 (hardcoded), model="PPO"

**`predict(symbol, risk_tolerance, behavior_array, user_id)` — Combined:**
1. Gets LSTM prediction (price forecast + initial action)
2. Gets PPO signal (action from RL policy)
3. If PPO available: overrides action with PPO decision, marks model as "LSTM+PPO"
4. Returns merged result

**Singleton:** `get_prediction_service()` — Module-level lazy singleton

### 2.8.2 `backtest_service.py`

**`class BacktestService`:**
- **Init:** `data_dir` defaults to `backend_root / "data" / "raw"` (resolved from file location)
- **Model:** Lazy-loads PPO from `models/ppo_trading_final.zip`
- Falls back to random actions if model not found

**`run(symbol, start_date, end_date, initial_capital, risk_tolerance)` Flow:**
1. Loads CSV from `data_dir/{symbol}.csv` (strips `.NS` suffix for filename)
2. Filters by date range (requires ≥50 data points)
3. Creates TradingEnv with filtered data
4. Runs simulation: loop predict → step → check done
5. Collects metrics from `env.get_episode_metrics()`
6. Returns: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades, final_value, trades[], equity_curve[]

### 2.8.3 `model_bootstrap.py`

Called at startup via `ensure_models_ready()`:

1. Resolves model directory (absolute or relative to backend root)
2. Checks for `lstm_final.pt` and `ppo_trading_final.zip`
3. If missing and `AUTO_TRAIN_IF_MISSING=True`:
   - Downloads training data if `training_data.csv` missing
   - Runs `train_lstm.py` if LSTM missing
   - Runs `train_ppo.py` if PPO missing
4. Scripts run as subprocess with proper PYTHONPATH
5. If `AUTO_TRAIN_STRICT=True`: raises exception on failure

### 2.8.4 `user_model_training_service.py`

Triggered by behavior assessment submission.

**`trigger_user_retraining(user_id, behavior_array)`:**
1. Sets status to "queued"
2. Spawns daemon thread running `_train_user_models()`

**`_train_user_models(user_id, behavior_array)`:**
1. Acquires per-user lock (non-blocking)
2. If locked: queues latest behavior array, sets status to "queued"
3. Creates `models/users/{user_id}/` directory
4. Runs `download_data.py` if training data missing
5. Runs `train_ppo.py --model-path <dir> --symbol ALL --behavior-json <json>`
6. Saves metadata to `meta.json`
7. Loops to process any queued updates
8. On failure: sets status to "failed"

**Per-user state tracking:**
- `_training_locks`: Dict[str, Lock] — per-user thread safety
- `_pending_behavior`: Dict[str, Dict] — queued behavior arrays
- `_training_status`: Dict[str, Dict] — status for each user

### 2.8.5 Storage Services

**`demo_store.py` — In-Memory Storage:**
- `@dataclass DemoUser`: id, email, hashed_password, is_active, created_at
- `@dataclass DemoBacktest`: id, user_id, symbol, metrics, trades, equity_curve, timestamp
- `@dataclass DemoProfile`: user_id, risk_tolerance, risk_category, preferences, behavior_profile, trades
- `class DemoStore`: thread-safe dict storage with auto-incrementing IDs
- Singleton: `demo_store = DemoStore()`

**`firestore_store.py` — Firebase Firestore:**
- Collections: `users/{uid}`, subcollections: `risk_assessments/latest`, `preferences/latest`, `behavior_profiles/latest`, `trade_events`, `trade_evaluations`
- Falls back to `demo_store` if Firebase not configured
- Singleton: `firestore_store = FirestoreStore()`

**`firebase_admin_service.py`:**
- `is_firebase_ready()` — Checks Firebase config validity
- `initialize_firebase_admin()` — One-time Firebase Admin SDK init
- `verify_firebase_token(id_token)` — Verifies Firebase JWT, returns `FirebasePrincipal`

**Database Models (`models/db/models.py`):**
- `User` — id, email, hashed_password, is_active, created_at. Relationships: trades, backtests, risk_profile
- `Trade` — id, user_id, symbol, action, quantity, price, pnl, timestamp
- `BacktestResult` — id, user_id, symbol, total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, total_trades, final_value, config (JSON), timestamp
- `RiskProfile` — id, user_id, score, answers (JSON), updated_at

**`database.py` — Sync Engine:**
- Demo mode: `SessionLocal = None`, `get_db()` yields None
- Normal mode: Creates sync engine from DATABASE_URL (strips `+asyncpg`)

**`async_database.py` — Async Engine:**
- Creates async engine from DATABASE_URL (adds `+asyncpg` if needed)
- Not currently used by any route (all use sync `get_db()`)

**`db_service.py` — CRUD Operations:**
- `DBService(db_session)` — Typed methods for all model CRUD

## 2.9 CORS Configuration

```python
CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
```
- `allow_credentials=True`
- `allow_methods=["*"]`
- `allow_headers=["*"]`

Must be updated for production deployments with proper domain origins.

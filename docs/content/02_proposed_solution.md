# 2. PROPOSED SOLUTION

## 2.1. Design Philosophy

The proposed solution is designed as a **modular, three-layered, and fail-safe** architecture:

1. **Modularization** improves maintainability and enables independent experimentation with each subsystem.
2. **Three-layer separation** cleanly divides concerns: data processing (Layer 1), RL decision-making (Layer 2), and trader behavior personalization (Layer 3).
3. **Fail-safe defaults** ensure that when model artifacts are unavailable, the system returns valid fallback outputs (IDLE action with 0.5 confidence) rather than crashing — preserving API continuity and user workflow.

## 2.2. System Architecture

The architecture consists of four major planes:

```
+-----------------------------------------------------------------+
|              APPLICATION PLANE (Frontend)                        |
|  Next.js 16 . React 19 . Zustand . Recharts . TailwindCSS 4     |
|  Dashboard . Backtest . Profile . Trades . Auth Pages            |
+---------------------------+-------------------------------------+
                            | HTTP/WS
+---------------------------v-------------------------------------+
|              APPLICATION PLANE (Backend API)                     |
|  FastAPI . Auth Routes . Trading Routes . Backtest Routes        |
|  WebSocket /ws/prices . PredictionService                       |
+-----------------------------------------------------------------+
                            |
+---------------------------v-------------------------------------+
|              LAYER 3: TRADER BEHAVIOR                            |
|  RiskProfiler . PositionSizer (Kelly Criterion) . BreakevenTracker
|  15-dim behavior vector . risk score 0-1 . position sizing      |
+-----------------------------------------------------------------+
                            |
+---------------------------v-------------------------------------+
|              LAYER 2: DECISION ENGINE                            |
|  TradingEnv (Gymnasium) . PPO Agent (SB3) . Sharpe/Sortino Reward
|  34-dim observation . 5-action discrete space                   |
+-----------------------------------------------------------------+
                            |
+---------------------------v-------------------------------------+
|              LAYER 1: DATA PROCESSING                            |
|  Market Data (yfinance) . Technical Indicators (pandas-ta, 30+)  |
|  State Builder (30-dim . padded to 34)                          |
+-----------------------------------------------------------------+
```

## 2.3. Functional Modules

**A. Market Data and Indicators (Layer 1)**

- **Symbol normalization:** `normalize_symbol()` in `market_data.py` handles `.NS` (NSE), `.BO` (BSE), and `^` (indices) prefixes correctly.
- **Data retrieval:** Synchronous `fetch_market_data_sync()` and async `get_market_data()` (via `run_in_executor`) wrapping yfinance.
- **Technical indicators:** `compute_indicators()` computes 30+ indicators via pandas-ta with a numpy fallback path. Indicators span trend (SMA_20/50/200, EMA_12/26, MACD, ADX), momentum (RSI_14, Stochastic, CCI_20, Williams %R, ROC_10), volatility (Bollinger Bands, ATR_14, ATR_pct), volume (OBV, MFI_14, Volume_Ratio), and composite signals (above_sma_20, above_sma_50, trend_bullish).
- **State builder:** `build_state()` constructs a 30-dimensional vector from price data, technical indicators, trader behavior, portfolio state, and sentiment. This is padded to 34 dimensions in TradingEnv.

**B. Predictive Modeling (LSTM)**

- **Architecture:** 2-layer LSTM (hidden=64, dropout=0.2) . FC(64.32, ReLU, Dropout) . FC(32.1). Total parameters: ~39,000.
- **Training:** 23,167 samples from 52 stocks, 80/20 chronological split, MSE loss, Adam optimizer (lr=0.001), ReduceLROnPlateau scheduler, 30 epochs. Validation loss: 0.000228.
- **Inference:** Fetches 3 months of daily data, creates a fresh MinMaxScaler on the last 30 points, produces a next-close price prediction. Maps change_pct to action BUY (>+1%), SELL (<-1%), or IDLE.

**C. Reinforcement Learning Decision Engine (PPO)**

- **Environment:** Custom `TradingEnv` (Gymnasium) with `Box(34,)` observation space, `Discrete(5)` action space (HOLD_BUY=0, HOLD_SELL=1, BUY=2, SELL=3, IDLE=4).
- **Agent:** Stable-Baselines3 PPO with MlpPolicy, net_arch=[256,256], learning_rate=3e-4 (risk-adjusted: 1e-4 to 5e-4), 30,000 timesteps.
- **Reward:** Blended step reward (position * price_change_pct - transaction_cost - risk_penalty) + episode Sharpe ratio scaled via tanh to [-1, 1]. Average return: 132.28%, Sharpe: 0.66.
- **Risk-adjusted creation:** `create_agent()` adjusts learning_rate and ent_coef based on risk tolerance (conservative: slow learning, high exploration; aggressive: fast learning, low exploration).

**D. Trader Behavior and Risk Profiling (Layer 3)**

- **Behavior assessment:** 30+ questions . 15-dimensional normalized behavior vector (0-1 scale). Fields include capital_per_trade_pct, tp_sl_ratio_preference, drawdown_sensitivity, intraday_var_limit, and more.
- **Risk scoring:** `calculate_risk_score()` maps responses to 0-1 scale, categorized as Conservative (<0.25), Moderate (0.25-0.50), Growth (0.50-0.75), or Aggressive (>0.75).
- **Position sizing:** Three strategies — fixed_percentage_size (5-20%), kelly_criterion_size (optimal growth, capped at 25%), volatility_adjusted_size (ATR-aware).
- **Trade evaluation:** Planned vs executed parameters compared; compliance_score = max(0, 100 - 20 * violations.count). Violations include capital_limit_exceeded, tp_sl_ratio_below_plan, premature_profit_close, cooldown_violation.

**E. Application and Delivery Layer**

- **REST APIs:** 15+ endpoints across 5 routers (auth, trading, backtest, profile, WebSocket) under `/api/v1`.
- **WebSocket:** `/api/v1/ws/prices` with subscribe/unsubscribe/set/ping protocol, 30-second polling interval, per-connection subscription sets.
- **Frontend:** Next.js 16 App Router with 9 pages, Zustand auth store with localStorage persistence, Axios interceptors (Bearer token injection + 401 redirect), Recharts charting (ComposedChart, LineChart), React Hook Form + Zod validation.
- **Auth:** Dual-pathway — Firebase SDK (frontend) + Firebase Admin SDK (backend), or JWT (HS256, 24h expiry) with bcrypt password hashing.

## 2.4. Runtime Modes and Reliability

The system supports three runtime modes controlled by environment variables:

| Mode | DEMO_MODE | FIREBASE_AUTH | Storage | Auth Method | Database Required |
|------|-----------|---------------|---------|-------------|-------------------|
| Demo | true | false | In-memory (DemoStore) | JWT (auto-create) | None |
| Firebase | false | true | Firestore | Firebase ID Token | None (Firestore) |
| Production | false | false | PostgreSQL | JWT + bcrypt | PostgreSQL 15 |

**Fallback Behavior:** If model files (lstm_final.pt, ppo_trading_final.zip) are missing or fail to load, the platform returns valid fallback outputs (IDLE action, 0.5 confidence, model="fallback") to avoid service interruption. The `model_bootstrap.py` module auto-trains missing models at startup if `AUTO_TRAIN_IF_MISSING=True`.

**Model File Summary:**
| File | Size (approx) | Purpose | Fallback If Missing |
|------|--------------|---------|-------------------|
| models/lstm_final.pt | ~500 KB | LSTM price prediction | Rule-based BUY/SELL/IDLE by change_pct |
| models/ppo_trading_final.zip | ~2 MB | PPO trading agent | IDLE with 0.5 confidence |
| models/users/{uid}/ppo_trading_final.zip | per-user | Personalized PPO | Global PPO model used |

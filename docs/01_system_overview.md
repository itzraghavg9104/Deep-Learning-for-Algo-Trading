# 1. System Overview

## 1.1 Project Purpose

Deep Learning for Algorithmic Trading is an AI-powered trading platform designed specifically for Indian equity markets (NSE/BSE). The system combines three distinct technological domains:

- **Market Data Processing** — Real-time and historical data ingestion via yfinance with 30+ technical indicators computed through pandas-ta
- **Deep Learning Forecasting** — LSTM neural networks for multi-step price prediction
- **Reinforcement Learning Decision Engine** — Proximal Policy Optimization (PPO) agents trained in a custom Gymnasium trading environment that learn risk-adjusted trading policies

The platform supports personalized trading through a behavior profiling system that adjusts position sizing, risk parameters, and even per-user PPO model retraining based on questionnaire responses.

### Problem Statement

Traditional algorithmic trading systems often fail to adapt to individual trader psychology and risk preferences. Most solutions use either purely technical indicators or rigid rule-based systems. This project addresses the gap by combining deep learning-based price forecasting with reinforcement learning that adapts trading policy to each user's behavioral profile. The system targets the Indian equity market, which has unique characteristics (NSE market hours 9:15-15:30 IST, high volatility, diverse retail participation).

### Objectives

1. Build a complete data pipeline for Indian stock market data (52 NIFTY 50 stocks)
2. Train an LSTM neural network to predict short-term price movements
3. Implement a PPO reinforcement learning agent for trading decisions in a custom Gymnasium environment
4. Create a trader behavior profiling system with questionnaire → behavior vector → personalized policy
5. Develop a full-stack web application with real-time dashboards and backtesting
6. Support three deployment modes: Demo (no infra), Firebase Auth, Full Production

## 1.2 High-Level Architecture (Three-Layer Design)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js 16)                        │
│  Dashboard │ Backtest UI │ Profile │ Trades │ Auth Pages       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/WS
┌───────────────────────────▼─────────────────────────────────────┐
│                   API / SERVICE LAYER                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────────┐  │
│  │  Auth    │ │ Trading  │ │ Backtest │ │  WebSocket        │  │
│  │  Routes  │ │ Routes   │ │ Routes   │ │  /ws/prices       │  │
│  └──────────┘ └────┬─────┘ └────┬─────┘ └───────────────────┘  │
│                    │            │                               │
│  ┌─────────────────▼────────────▼────────────────────────────┐  │
│  │                    SERVICES                               │  │
│  │  PredictionService │ BacktestService │ ModelBootstrap     │  │
│  │  DemoStore │ FirestoreStore │ DBService │ UserModelTraining│  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              LAYER 3: TRADER BEHAVIOR                           │
│  RiskProfiler │ PositionSizer (Kelly) │ BreakevenTracker        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Behavior Array (15-dim normalized vector)               │   │
│  │  → capital_per_trade_pct, tp_sl_ratio, drawdown_sens,   │   │
│  │    post_loss_rest, trade_frequency, holding_time, etc.   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              LAYER 2: DECISION ENGINE                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  TradingEnv  │  │  PPO Agent   │  │  Reward Function     │  │
│  │  (Gymnasium) │  │  (SB3 PPO)   │  │  Sharpe / Sortino    │  │
│  │  5-action    │  │  MLP Policy  │  │  Step + Episode      │  │
│  │  space       │  │  [256,256]   │  │  reward calculation  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
│         │                 │                                      │
│         └─────────────────┘                                      │
│         State: 34-dim observation vector                         │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              LAYER 1: DATA PROCESSING                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │  Market Data   │  │  Technical     │  │  State Builder │    │
│  │  (yfinance)    │──┤  Indicators    │──┤  (30-dim)      │    │
│  │  OHLCV fetch   │  │  (pandas-ta)   │  │  Normalization │    │
│  │  Async/Sync    │  │  30+ features  │  │  + padding→34  │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Rationale

The three-layer separation follows a clear pipeline pattern:
- **Layer 1** handles all data transformation (raw market → structured features → normalized state vector)
- **Layer 2** contains the RL decision logic isolated in a Gymnasium environment for both training and inference
- **Layer 3** encodes trader psychology as mathematical constraints on the decision space

This separation enables independent testing, swapping of components (e.g., different reward functions), and parallel development.

## 1.3 Technology Stack

### Backend

| Component | Technology | Version/Config | Justification |
|-----------|-----------|----------------|---------------|
| Framework | FastAPI | Python 3.12 | Async-native, auto-docs, Pydantic integration |
| ASGI Server | Uvicorn | — | Dev: `--reload`, Prod: `--host 0.0.0.0` |
| Deep Learning | PyTorch | 2.x | LSTM: 2-layer, hidden=64, seq_len=30, ~39K params |
| RL | Stable-Baselines3 | 2.x | PPO, MlpPolicy, net_arch=[256,256] |
| RL Env | Gymnasium | 0.29+ | Box(34,) obs, Discrete(5) actions |
| Data | pandas, numpy, scikit-learn | — | MinMaxScaler, pandas-ta, rolling stats |
| Market Data | yfinance | — | NSE `.NS`, BSE `.BO` suffixes, 1d and 5m intervals |
| Auth (Firebase) | Firebase Admin SDK | — | verify_id_token, Firestore read/write |
| Auth (JWT) | python-jose + bcrypt | — | HS256, 24h expiry, salt rounds |
| Database | SQLAlchemy + asyncpg | — | PostgreSQL 15, ORM + raw async |
| Cache | Redis 7 | — | Via `REDIS_URL`, optional |
| Persistence | In-memory / Firestore / PostgreSQL | — | Three-tier strategy via config flags |

### Frontend

| Component | Technology | Version | Justification |
|-----------|-----------|---------|---------------|
| Framework | Next.js | 16.0.10 | App Router, SSR, middleware |
| UI Library | React | 19.2.1 | Component model, hooks |
| Language | TypeScript | 5.x (strict) | Type safety, `@/*` → `src/*` |
| Styling | TailwindCSS | 4.x | Utility-first, rapid prototyping |
| State | Zustand | 5.x | Lightweight, persist middleware |
| Charts | Recharts | 3.6 | React-native charting, ComposedChart |
| Forms | React Hook Form | 4.x | Performant form state |
| Validation | Zod | 4.x | Schema validation |
| HTTP | Axios | — | Interceptors, base URL config |
| Auth | Firebase JS SDK | 12.x | Email/password + Google OAuth |
| Icons | Lucide React | — | Consistent icon set |
| Build | Turbopack | — | Via Next.js, HMR |

### Infrastructure

| Component | Technology | Notes |
|-----------|-----------|-------|
| Containerization | Docker + Docker Compose | 4 services: db, redis, backend, frontend |
| Database | PostgreSQL 15 | Optional via DEMO_MODE |
| Cache | Redis 7 | Optional via DEMO_MODE |
| API Docs | Swagger (/docs) + Redoc (/redoc) | Auto-generated from Pydantic models |

## 1.4 Runtime Modes

The system operates in three modes controlled by `backend/app/config.py`:

### Mode 1: Demo Mode (Default)

```
DEMO_MODE=True, FIREBASE_AUTH_ENABLED=False
```

- No database or Redis required — runs entirely in-memory
- `DemoStore` provides dict-based persistence for users, profiles, backtests, and trades
- Auth is permissive: login auto-creates users, skips password verification
- All data resets on backend restart
- Perfect for local development, demos, and testing
- State dimension: 34 (TradingEnv)
- Model loading: optional, graceful fallback to rule-based signals
- **Startup behavior**: CORS allows localhost:3000; model bootstrap auto-trains if `AUTO_TRAIN_IF_MISSING=True`

### Mode 2: Firebase Mode

```
DEMO_MODE=false, FIREBASE_AUTH_ENABLED=true
```

- Firebase Admin SDK verifies ID tokens from the frontend
- Firebase Firestore stores user profiles, risk assessments, behavior profiles, trade evaluations
- Per-user PPO retraining enabled (threaded, async)
- Requires Firebase service account JSON at `FIREBASE_SERVICE_ACCOUNT_PATH`
- JWT verification skipped — all auth delegated to Firebase
- **Frontend**: Must configure Firebase project credentials in `.env.local`

### Mode 3: Full Production Mode

```
DEMO_MODE=false, FIREBASE_AUTH_ENABLED=false
```

- PostgreSQL via SQLAlchemy for all CRUD operations
- Redis for caching (optional)
- Standard bcrypt password hashing + JWT token auth
- Full persistent storage across restarts
- Production security checks at startup (fails on DEBUG=true, default secrets, etc.)
- Tables auto-created via SQLAlchemy ORM

### Mode Comparison

| Feature | Demo | Firebase | Production |
|---------|------|----------|------------|
| Database Required | No | No (Firestore) | Yes (PostgreSQL) |
| Redis Required | No | No | Optional |
| User Persistence | In-memory | Firestore | PostgreSQL |
| Auth Method | JWT (auto-create) | Firebase ID Token | JWT (bcrypt verify) |
| Password Verification | Skipped | Handled by Firebase | bcrypt hashpw |
| Data Persistence | None (resets) | Persistent | Persistent |
| Startup Security Checks | No | No | Yes |
| Per-User PPO Training | Yes | Yes | Yes |
| Setup Complexity | Minimal | Medium (Firebase config) | High (DB + Redis) |

## 1.5 Repository Structure

```
Deep-Learning-for-Algo-Trading/
│
├── backend/
│   ├── .env                          # Active env config (gitignored)
│   ├── .env.example                  # Template env vars
│   ├── Dockerfile                    # Python 3.12-slim build
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                 # pydantic-settings Settings class
│   │   ├── main.py                   # FastAPI entrypoint, CORS, routers
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── websocket.py          # /api/v1/ws/prices
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── auth.py           # /api/v1/auth/*
│   │   │       ├── trading.py        # /api/v1/trading/*
│   │   │       ├── backtest.py       # /api/v1/backtest/*
│   │   │       └── profile.py        # /api/v1/profile/*
│   │   │
│   │   ├── layer1_data_processing/
│   │   │   ├── __init__.py
│   │   │   ├── market_data.py        # yfinance fetch, symbol normalization
│   │   │   ├── technical_indicators.py  # 30+ indicators via pandas-ta
│   │   │   └── state_builder.py      # RL state vector construction
│   │   │
│   │   ├── layer2_decision/
│   │   │   ├── __init__.py
│   │   │   ├── action_space.py       # Canonical 5-action definitions
│   │   │   ├── reward_function.py    # Sharpe/Sortino/step rewards
│   │   │   ├── trading_env.py        # Gymnasium TradingEnv
│   │   │   └── ppo_agent.py          # SB3 PPO wrapper
│   │   │
│   │   ├── trader_behavior/
│   │   │   ├── __init__.py
│   │   │   ├── risk_profiler.py      # Questionnaire→risk score
│   │   │   ├── position_sizer.py     # Kelly Criterion + volatility sizing
│   │   │   └── breakeven_tracker.py  # Position tracking
│   │   │
│   │   ├── models/
│   │   │   └── db/
│   │   │       ├── __init__.py
│   │   │       ├── database.py        # SQLAlchemy sync engine
│   │   │       ├── async_database.py  # SQLAlchemy async engine
│   │   │       ├── models.py          # ORM: User, Trade, BacktestResult, RiskProfile
│   │   │       └── db_service.py      # CRUD operations
│   │   │
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── prediction_service.py  # LSTM+PPO inference singleton
│   │       ├── model_bootstrap.py     # Auto-train missing models at startup
│   │       ├── backtest_service.py    # Historical backtest runner
│   │       ├── demo_store.py          # In-memory persistence
│   │       ├── firestore_store.py     # Firebase Firestore persistence
│   │       ├── firebase_admin_service.py  # Firebase Admin SDK init + verify
│   │       └── user_model_training_service.py  # Per-user PPO retraining
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── download_data.py           # NIFTY 50 downloader
│   │   ├── train_lstm.py              # LSTM price predictor training
│   │   ├── train_ppo.py               # PPO trading agent training
│   │   └── train_deepar.py            # DeepAR (experimental, unused at runtime)
│   │
│   ├── data/
│   │   ├── training_data.csv          # Combined training dataset
│   │   └── raw/                       # Per-symbol CSV files
│   │
│   └── models/
│       ├── lstm_final.pt              # LSTM final checkpoint (gitignored)
│       ├── lstm_best.pt               # LSTM best validation checkpoint (gitignored)
│       ├── ppo_trading_final.zip      # PPO agent (gitignored)
│       ├── ppo_tensorboard/           # TensorBoard logs (gitignored)
│       └── users/                     # Per-user PPO models (gitignored)
│
├── frontend/
│   ├── .env.local                    # Firebase + API URL config
│   ├── Dockerfile                    # Node.js build
│   ├── package.json                  # Next.js 16, React 19, etc.
│   ├── next.config.ts                # Next.js configuration
│   ├── tsconfig.json                 # Strict TypeScript
│   └── src/
│       ├── middleware.ts              # Route protection
│       ├── app/
│       │   ├── globals.css            # TailwindCSS v4
│       │   ├── layout.tsx             # Root layout + AuthInitializer
│       │   ├── page.tsx               # Landing page
│       │   ├── auth/
│       │   │   ├── login/page.tsx     # Login form
│       │   │   └── register/page.tsx  # Registration form
│       │   ├── dashboard/
│       │   │   ├── layout.tsx         # Dashboard shell + Sidebar
│       │   │   └── page.tsx           # Main trading dashboard
│       │   ├── backtest/page.tsx      # Backtest runner
│       │   ├── trades/page.tsx        # Trade history
│       │   └── profile/
│       │       ├── page.tsx           # Profile/preferences
│       │       └── risk-assessment/page.tsx  # Risk questionnaire
│       ├── components/
│       │   ├── AuthInitializer.tsx     # Auth state restoration
│       │   ├── dashboard/
│       │   │   ├── index.ts           # Re-exports
│       │   │   ├── Sidebar.tsx        # Navigation + risk card
│       │   │   ├── SignalCard.tsx     # Stock signal card
│       │   │   └── StatsCard.tsx      # KPI metric card
│       │   ├── charts/
│       │   │   ├── PriceChart.tsx     # OHLCV + volume chart
│       │   │   ├── Sparkline.tsx      # Mini trend chart
│       │   │   ├── SignalGauge.tsx    # Confidence bar
│       │   │   ├── EquityCurve.tsx    # Backtest equity curve
│       │   │   └── TechnicalIndicators.tsx  # Indicator grid
│       │   └── forms/
│       │       ├── BacktestConfig.tsx  # Backtest parameters form
│       │       └── RiskQuestionnaire.tsx  # 30-question behavior assessment
│       └── lib/
│           ├── api.ts                # Axios instance + API modules
│           ├── auth-store.ts         # Zustand auth store (Firebase)
│           ├── firebase.ts           # Firebase initialization
│           ├── market-hours.ts       # NSE market hours checker
│           ├── trading-format.ts     # Action formatting utilities
│           └── use-websocket.ts      # WebSocket hook with reconnection
│
├── docker-compose.yml               # PostgreSQL + Redis + Backend + Frontend
├── AGENTS.md                         # Agent guidelines for ML-enhanced coding
├── IMPLEMENTATION_PLAN.md            # Phased implementation plan (6 phases)
├── plan.md                           # Firebase migration + behavior intelligence
├── README.md                         # Project overview and quick start
├── setup.md                          # Setup instructions
├── start.md                          # Quick start guide
├── TODO.md                           # Remaining tasks
├── CLAUDE.md / GEMINI.md             # AI assistant guidelines
├── references/                       # Research papers and resources
└── secrets/                          # Firebase service account (gitignored)
```

## 1.6 Market Data Conventions

### Symbol Format
- **NSE stocks**: `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`)
- **BSE stocks**: `.BO` suffix (e.g., `RELIANCE.BO`)
- **Indices**: `^` prefix (e.g., `^NSEI` for NIFTY 50, `^BSESN` for SENSEX)
- **normalize_symbol()** in `market_data.py` handles suffix normalization automatically

### NSE Stock Universe (52 Constituents)

The platform supports the full NIFTY 50 index. The predefined mapping in `market_data.py`:

| # | Symbol | Company | # | Symbol | Company |
|---|--------|---------|---|--------|---------|
| 1 | ADANIENT.NS | Adani Enterprises | 27 | INFY.NS | Infosys |
| 2 | ADANIPORTS.NS | Adani Ports & SEZ | 28 | ITC.NS | ITC |
| 3 | APOLLOHOSP.NS | Apollo Hospitals | 29 | JIOFIN.NS | Jio Financial Services |
| 4 | ASIANPAINT.NS | Asian Paints | 30 | JSWSTEEL.NS | JSW Steel |
| 5 | AXISBANK.NS | Axis Bank | 31 | KOTAKBANK.NS | Kotak Mahindra Bank |
| 6 | BAJAJ-AUTO.NS | Bajaj Auto | 32 | LT.NS | Larsen & Toubro |
| 7 | BAJFINANCE.NS | Bajaj Finance | 33 | M&M.NS | Mahindra & Mahindra |
| 8 | BAJAJFINSV.NS | Bajaj Finserv | 34 | MARUTI.NS | Maruti Suzuki |
| 9 | BEL.NS | Bharat Electronics | 35 | NESTLEIND.NS | Nestle India |
| 10 | BHARTIARTL.NS | Bharti Airtel | 36 | NTPC.NS | NTPC |
| 11 | BPCL.NS | BPCL | 37 | ONGC.NS | ONGC |
| 12 | BRITANNIA.NS | Britannia Industries | 38 | POWERGRID.NS | Power Grid Corp |
| 13 | CIPLA.NS | Cipla | 39 | RELIANCE.NS | Reliance Industries |
| 14 | COALINDIA.NS | Coal India | 40 | SBILIFE.NS | SBI Life Insurance |
| 15 | DRREDDY.NS | Dr. Reddy's Labs | 41 | SHRIRAMFIN.NS | Shriram Finance |
| 16 | EICHERMOT.NS | Eicher Motors | 42 | SBIN.NS | State Bank of India |
| 17 | ETERNAL.NS | Eternal (Zomato) | 43 | SUNPHARMA.NS | Sun Pharma |
| 18 | GRASIM.NS | Grasim Industries | 44 | TCS.NS | Tata Consultancy Services |
| 19 | HCLTECH.NS | HCL Technologies | 45 | TATACONSUM.NS | Tata Consumer |
| 20 | HDFCBANK.NS | HDFC Bank | 46 | TATAMOTORS.NS | Tata Motors |
| 21 | HDFCLIFE.NS | HDFC Life Insurance | 47 | TATASTEEL.NS | Tata Steel |
| 22 | HEROMOTOCO.NS | Hero MotoCorp | 48 | TECHM.NS | Tech Mahindra |
| 23 | HINDALCO.NS | Hindalco Industries | 49 | TITAN.NS | Titan Company |
| 24 | HINDUNILVR.NS | Hindustan Unilever | 50 | TRENT.NS | Trent |
| 25 | ICICIBANK.NS | ICICI Bank | 51 | ULTRACEMCO.NS | UltraTech Cement |
| 26 | INDUSINDBK.NS | IndusInd Bank | 52 | WIPRO.NS | Wipro |

### Data Sources
- **yfinance**: Primary data source for OHLCV, supports period-based and interval-based queries
- **5-minute caching**: Market data is cached for 5 minutes to avoid repeated yfinance API calls
- **NSE_STOCKS**: Predefined dict of 52 NIFTY 50 constituent symbols with company name mapping

### Data Storage

| File | Purpose | Format |
|------|---------|--------|
| `data/raw/{Symbol}.csv` | Per-symbol raw OHLCV | CSV, Date/Open/High/Low/Close/Volume/Symbol |
| `data/raw/nifty50_combined.csv` | All stocks concatenated | Same + Symbol column |
| `data/training_data.csv` | Cleaned ML training set | date/symbol/open/high/low/close/volume/time_idx/returns |

## 1.7 Auth System Overview

The system implements a dual-pathway authentication architecture with three runtime modes:

### Demo Mode Auth Flow
```
Frontend (login form)
  → POST /api/v1/auth/login (username + password)
  → Backend: auto-creates user if not found, skips password verification
  → Returns JWT (HS256, sub=email, 24h expiry)
  → Frontend stores in localStorage + auth_token cookie
  → Middleware checks cookie for route protection
```

### Firebase Auth Flow
```
Frontend (login form)
  → Firebase SDK (signInWithEmailAndPassword / signInWithPopup)
  → Firebase returns idToken (JWT)
  → Frontend stores idToken in localStorage + auth_token cookie
  → All API calls include Authorization: Bearer <idToken>
  → Backend: Firebase Admin SDK verifies idToken
  → User auto-provisioned in Firestore if new
```

### Full Production Auth Flow
```
Frontend (login form)
  → POST /api/v1/auth/login (username + password)
  → Backend: bcrypt verify password against PostgreSQL hash
  → Returns JWT (HS256, sub=email, 24h expiry)
  → Frontend stores in localStorage + auth_token cookie
  → Middleware checks cookie for route protection
```

### Session Persistence
- Zustand store persisted under key `auth-storage` in localStorage
- `auth_token` cookie set for Next.js middleware route protection (1-day expiry, SameSite=Lax)
- On 401 response: auto-logout, redirect to `/auth/login`
- On app load: `initializeAuth()` restores token from store and re-fetches user

## 1.8 What the System Currently Delivers

### Data & Analysis
- Real-time/live market data for 52 NIFTY 50 stocks via yfinance
- 30+ technical indicators (trend, momentum, volatility, volume)
- OHLCV history for charting (up to 180 data points)
- Stock info (fundamentals) via yfinance

### Predictions & Signals
- LSTM-based price prediction (next-close forecast)
- PPO-based trading action selection (5-action space: HOLD_BUY, HOLD_SELL, BUY, SELL, IDLE)
- Combined LSTM+PPO signal when both models available (PPO action overrides LSTM)
- Rule-based fallback when models unavailable
- Confidence scoring for each signal (0.0–1.0 scale)

### Backtesting
- Historical simulation through TradingEnv (Gymnasium)
- PPO agent trading on historical data
- Random action fallback when no model
- Full metrics: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor

### Personalization
- Risk questionnaire (6 questions, legacy, 1–4 scale)
- Behavior assessment (30+ questions, ~15-dim behavior vector normalized to 0–1)
- User-specific PPO model retraining (async, daemon thread with per-user locking)
- Position sizing via Kelly Criterion, fixed percentage, or volatility-adjusted
- Trade evaluation/compliance scoring (planned vs executed)

### Frontend
- Authenticated dashboard with real-time signals
- Backtest configuration and results visualization
- Trade history with filters and CSV export
- Risk profile management
- Interactive charts (price, equity curve, indicators)
- WebSocket-powered live price updates (30s polling interval)

## 1.9 Methodology

The project follows a sequential pipeline methodology:

1. **Data Acquisition**: Download 5 years of daily OHLCV data for 52 NIFTY 50 stocks via yfinance
2. **Feature Engineering**: Compute 30+ technical indicators, normalize, create sequence windows
3. **Supervised Learning (LSTM)**: Train on 80/20 chronological split to predict next-day closing price using MSE loss, Adam optimizer, ReduceLROnPlateau scheduler
4. **Reinforcement Learning (PPO)**: Train in custom Gymnasium TradingEnv with Sharpe ratio reward, 5-action discrete space, 34-dim observation, 30k-50k timesteps
5. **Behavior Profiling**: Map questionnaire responses (1-5 scale) → 15-dim normalized behavior vector → risk score (0-1) → position sizing constraints
6. **Inference Pipeline**: Fetch live data → compute indicators → LSTM predict → PPO decide → behavior adjust → return signal
7. **Web Application**: FastAPI backend serves REST endpoints + WebSocket; Next.js frontend provides dashboard

## 1.10 Key Challenges Faced

1. **State dimension mismatch**: `state_builder.py` outputs 30-dim but TradingEnv expects 34-dim. Fixed by zero-padding in `_get_observation()`, but this is a design inconsistency.
2. **LSTM inference scaling mismatch**: Training uses per-symbol MinMaxScaler, but inference creates a fresh scaler on query data — can cause prediction distribution shift.
3. **PPO confidence hardcoded**: `get_ppo_signal()` returns confidence=0.8 regardless of actual policy certainty, when `TradingAgent.get_action_with_confidence()` properly extracts action probabilities.
4. **Train-inference skew**: PPO training uses default behavior array `{capital_per_trade=0.1, tp_sl=0.4, drawdown=0.2, cooldown=0.1}`. Inference uses user-specific values — policy may not generalize.
5. **yfinance limitations**: Not a real-time feed; 30s polling introduces latency. Rate limits can cause data gaps.
6. **Docker Node version mismatch**: Frontend Dockerfile uses `node:18-slim` but package.json requires Node 20+ features.
7. **No test coverage**: Both backend and frontend lack automated tests.

## 1.11 Learning Outcomes

1. **Deep Reinforcement Learning**: Implementing and training PPO in a custom Gymnasium environment, understanding policy gradients, reward shaping, and state representation
2. **Time Series Forecasting**: LSTM architecture design, sequence windowing, scaling strategies, and evaluation
3. **Full-Stack Development**: FastAPI async patterns, Next.js App Router, Zustand state management, WebSocket integration
4. **Financial Domain**: Indian market conventions (NSE/BSE), technical analysis, risk management, position sizing
5. **System Architecture**: Three-layer separation, dependency injection, runtime mode switching, service singletons
6. **DevOps**: Docker Compose orchestration, multi-stage builds, environment configuration management

## 1.12 Training Results Summary

| Model | Metric | Value |
|-------|--------|-------|
| **LSTM** | Architecture | 2-layer LSTM (hidden=64), seq_len=30 |
| | Training Samples | 23,167 |
| | Validation Loss (MSE) | **0.000228** |
| | Optimizer | Adam (lr=0.001) with ReduceLROnPlateau |
| **PPO** | Training Timesteps | 30,000 |
| | Average Return (5 eval episodes) | **132.28%** |
| | Sharpe Ratio | **0.66** |
| | Policy Network | MlpPolicy [256, 256] |
| **DeepAR** | Status | Experimental — trained but not in inference pipeline |
| | Architecture | 2-layer RNN, hidden=32, pred_len=5, enc_len=30 |

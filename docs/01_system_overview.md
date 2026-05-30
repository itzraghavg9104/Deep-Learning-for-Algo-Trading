# 1. System Overview

## 1.1 Project Purpose

Deep Learning for Algorithmic Trading is an AI-powered trading platform designed specifically for Indian equity markets (NSE/BSE). The system combines three distinct technological domains:

- **Market Data Processing** — Real-time and historical data ingestion via yfinance with 30+ technical indicators computed through pandas-ta
- **Deep Learning Forecasting** — LSTM neural networks for multi-step price prediction  
- **Reinforcement Learning Decision Engine** — Proximal Policy Optimization (PPO) agents trained in a custom Gymnasium trading environment that learn risk-adjusted trading policies

The platform supports personalized trading through a behavior profiling system that adjusts position sizing, risk parameters, and even per-user PPO model retraining based on questionnaire responses.

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

## 1.3 Technology Stack

### Backend

| Component | Technology | Version/Config |
|-----------|-----------|----------------|
| Framework | FastAPI | Python 3.12 |
| ASGI Server | Uvicorn | dev: `--reload`, prod: `--host 0.0.0.0` |
| Deep Learning | PyTorch | LSTM: 2-layer, hidden=64, seq_len=30 |
| RL | Stable-Baselines3 | PPO, MlpPolicy, net_arch=[256,256] |
| RL Env | Gymnasium | Box(34,) obs, Discrete(5) actions |
| Data | pandas, numpy, scikit-learn | MinMaxScaler, pandas-ta |
| Market Data | yfinance | NSE `.NS`, BSE `.BO` suffixes |
| Auth | Firebase Admin SDK / python-jose + bcrypt | Dual auth paths |
| Database | SQLAlchemy + asyncpg (optional) | PostgreSQL 15 |
| Cache | Redis 7 (optional) | via `REDIS_URL` |
| Persistence | In-memory / Firebase Firestore / PostgreSQL | Three-tier strategy |

### Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Next.js | 16.0.10 (App Router) |
| UI Library | React | 19.2.1 |
| Language | TypeScript | 5.x (strict mode) |
| Styling | TailwindCSS | 4.x |
| State | Zustand | 5.x (with persist middleware) |
| Charts | Recharts | 3.6 |
| Forms | React Hook Form + Zod | 4.x |
| HTTP | Axios | with interceptors |
| Auth | Firebase JS SDK | 12.x (email/password + Google) |
| Icons | Lucide React | latest |
| Build | `npm run build` | lint + typecheck pre-PR |

### Infrastructure

| Component | Technology | Notes |
|-----------|-----------|-------|
| Containerization | Docker + Docker Compose | 4 services |
| Database | PostgreSQL 15 | Optional via DEMO_MODE |
| Cache | Redis 7 | Optional via DEMO_MODE |
| API Docs | Swagger (/docs) + Redoc (/redoc) | Auto-generated |

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

### Mode 2: Firebase Mode

```
DEMO_MODE=false, FIREBASE_AUTH_ENABLED=true
```

- Firebase Admin SDK verifies ID tokens from the frontend
- Firebase Firestore stores user profiles, risk assessments, behavior profiles, trade evaluations
- Per-user PPO retraining enabled (threaded, async)
- Requires Firebase service account JSON
- JWT verification skipped — all auth delegated to Firebase

### Mode 3: Full Production Mode

```
DEMO_MODE=false, FIREBASE_AUTH_ENABLED=false
```

- PostgreSQL via SQLAlchemy for all CRUD operations
- Redis for caching (optional)
- Standard bcrypt password hashing + JWT token auth
- Full persistent storage across restarts
- Production security checks at startup (fails on DEBUG=true, default secrets, etc.)

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
├── AGENTS.md                         # Agent guidelines
├── CLAUDE.md                         # Claude Code guidelines
├── GEMINI.md                         # Gemini guidelines
├── IMPLEMENTATION_PLAN.md
├── plan.md
├── README.md
├── setup.md
├── start.md
├── TODO.md
└── secrets/                          # Firebase service account (gitignored)
```

## 1.6 Market Data Conventions

### Symbol Format
- **NSE stocks**: `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`)
- **BSE stocks**: `.BO` suffix (e.g., `RELIANCE.BO`)
- **Indices**: `^` prefix (e.g., `^NSEI` for NIFTY 50, `^BSESN` for SENSEX)
- **normalize_symbol()** in `market_data.py` handles suffix normalization automatically

### Data Sources
- **yfinance**: Primary data source for OHLCV, supports period-based and interval-based queries
- **NSE_STOCKS**: Predefined dict of 52 NIFTY 50 constituent symbols
- Watchlist endpoint queries 20 major stocks + 3 indices

### Data Storage

| File | Purpose | Format |
|------|---------|--------|
| `data/raw/{Symbol}.csv` | Per-symbol raw OHLCV | CSV, Date/Open/High/Low/Close/Volume/Symbol |
| `data/raw/nifty50_combined.csv` | All stocks concatenated | Same + Symbol column |
| `data/training_data.csv` | Cleaned ML training set | date/symbol/open/high/low/close/volume/time_idx/returns |

## 1.7 Auth System Overview

The system implements a dual-pathway authentication architecture:

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

### Session Persistence
- Zustand store persisted under key `auth-storage` in localStorage
- `auth_token` cookie set for Next.js middleware route protection
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
- PPO-based trading action selection (5-action space)
- Combined LSTM+PPO signal when both models available
- Rule-based fallback when models unavailable
- Confidence scoring for each signal

### Backtesting
- Historical simulation through TradingEnv
- PPO agent trading on historical data
- Random action fallback when no model
- Full metrics: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor

### Personalization
- Risk questionnaire (6 questions, legacy)
- Behavior assessment (30+ questions, ~15-dim behavior vector)
- User-specific PPO model retraining (async, threaded)
- Position sizing via Kelly Criterion
- Trade evaluation/compliance scoring

### Frontend
- Authenticated dashboard with real-time signals
- Backtest configuration and results visualization
- Trade history with filters and CSV export
- Risk profile management
- Interactive charts (price, equity curve, indicators)
- WebSocket-powered live price updates

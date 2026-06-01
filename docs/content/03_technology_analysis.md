# 3. TECHNOLOGY ANALYSIS

## 3.1. Methodology

The implementation follows a pipeline-oriented engineering methodology:

1. **Historical Data Collection:** Download 5 years of daily OHLCV data for 52 NIFTY 50 stocks via yfinance. Output: `data/training_data.csv` (79,628 rows), per-symbol CSVs in `data/raw/`.

2. **Feature Engineering with Indicator Enrichment:** Compute 30+ technical indicators (trend, momentum, volatility, volume) using pandas-ta. Build 30-dimensional RL state vectors combining price data, indicators, behavior profile, and portfolio state.

3. **Supervised Sequence Model Training (LSTM):** Split data 80/20 chronologically per symbol. Create sequences of length 30 days with 5 features (OHLCV). Train StockLSTM for 30 epochs with MSE loss, Adam optimizer, ReduceLROnPlateau scheduler, and gradient clipping (max_norm=1.0). Save best and final checkpoints.

4. **Reinforcement Learning Policy Training (PPO):** Create custom TradingEnv with 34-dim observation, 5-action discrete space, transaction cost 0.1%, initial capital 100,000. Train PPO for 30,000 timesteps with Sharpe ratio reward. Evaluate over 5 episodes reporting avg_return, sharpe ratio, max_drawdown.

5. **Behavior Profiling System:** Design 30+ multi-choice questions across 5 categories (capital allocation, risk parameters, overtrading controls, market context, breakeven strategy). Map responses to 15-dim normalized behavior vector. Apply Kelly Criterion for position sizing.

6. **API and WebSocket Integration:** Build FastAPI REST endpoints (5 routers) and WebSocket handler with subscription management. Implement dual-pathway auth (Firebase/JWT). Add CORS middleware, startup security validation, and model bootstrap.

7. **Frontend Orchestration:** Build Next.js 16 App Router pages with Zustand auth state, Axios interceptors, Recharts visualizations, and middleware-based route protection for /dashboard, /profile, /backtest, /trades.

8. **Deployment Configuration:** Docker Compose with 4 services (postgres:15, redis:7, backend, frontend). Environment-driven mode switching. Production security guardrails (fails on DEBUG=true, default secrets).

## 3.2. Flow Chart Diagrams

The technical documentation provides standardized diagrams for report inclusion (available in `docs/images/`):

- `system_architecture.png` — Complete system architecture showing all four planes and data flow
- `data_flow.png` — Data pipeline from yfinance through indicators, state builder, to TradingEnv
- `training_pipeline.png` — Sequential training workflow: download_data . train_lstm . train_ppo
- `lstm_architecture.png` — LSTM model architecture (2-layer, hidden=64, seq_len=30, FC layers)
- `ppo_training.png` — PPO training loop with TradingEnv interaction, reward calculation, policy update
- `risk_profiler.png` — Behavior profiling flow: questionnaire . behavior array . risk score . position sizing

**Key Data Flow:**

```
yfinance (52 NIFTY 50 stocks)
    . download_data.py (5 years OHLCV)
    . train_lstm.py (30 epochs, 23,167 samples)
        . models/lstm_final.pt
    . train_ppo.py (30,000 timesteps, Sharpe reward)
        . models/ppo_trading_final.zip
    . Runtime Inference (GET /api/v1/trading/signals/{symbol})
        . LSTM predicts next-close . PPO selects action . Merge . SignalResponse
```

## 3.3. Tech Stack Analysis

**Backend and ML:**

| Technology | Version | Role | Key Configuration |
|-----------|---------|------|-------------------|
| Python | 3.12 | Core runtime | Async/await, type hints |
| FastAPI | latest | REST + WebSocket framework | Auto-docs, Pydantic validation |
| Uvicorn | latest | ASGI server | Dev: --reload, Prod: --host 0.0.0.0 |
| PyTorch | 2.x | Deep learning (LSTM) | 2-layer LSTM, hidden=64, ~39K params |
| Stable-Baselines3 | 2.x | Reinforcement learning (PPO) | MlpPolicy, [256,256], 3e-4 lr |
| Gymnasium | 0.29+ | RL environment framework | Box(34,) obs, Discrete(5) actions |
| pandas / numpy | latest | Data processing | MinMaxScaler, rolling statistics |
| yfinance | latest | Market data source | NSE .NS, BSE .BO, 1d/5m intervals |
| pandas-ta | latest | Technical indicators | 30+ indicators, numpy fallback |
| SQLAlchemy + asyncpg | latest | Database ORM | PostgreSQL 15, sync + async engines |
| python-jose + bcrypt | latest | JWT + password hashing | HS256, 24h expiry, salt rounds |
| Firebase Admin SDK | latest | Firebase auth verification | verify_id_token, Firestore CRUD |

**Frontend:**

| Technology | Version | Role | Key Features |
|-----------|---------|------|--------------|
| Next.js | 16.0.10 | React framework | App Router, middleware, SSR |
| React | 19.2.1 | UI library | Server/client components, hooks |
| TypeScript | 5.x | Type safety | Strict mode, @/* . src/* |
| TailwindCSS | 4.x | Styling | Utility-first, responsive design |
| Zustand | 5.x | State management | Persist middleware (localStorage) |
| Recharts | 3.6 | Charts | ComposedChart, LineChart, responsive |
| React Hook Form | 4.x | Forms | Performant form state management |
| Zod | 4.x | Validation | Schema-based form validation |
| Axios | latest | HTTP client | Request/response interceptors |
| Firebase JS SDK | 12.x | Client auth | signInWithPopup, email/password |
| Lucide React | latest | Icons | Consistent icon library |

**Infrastructure:**

| Technology | Version | Purpose |
|-----------|---------|---------|
| Docker | latest | Containerization |
| Docker Compose | latest | Multi-service orchestration (4 services) |
| PostgreSQL | 15 | Primary database |
| Redis | 7 | Caching layer |

**Stack Rationale:**

- **FastAPI over Flask/Django:** Async-native with automatic OpenAPI docs, Pydantic integration, and WebSocket support — ideal for ML API serving.
- **PPO over DQN/A2C:** PPO's clipped surrogate objective provides stable training in high-variance financial domains (Schulman et al., 2017). SB3 provides production-grade implementation.
- **Next.js over plain React:** App Router with server components, built-in middleware for auth, and optimized bundling via Turbopack.
- **Zustand over Redux:** Minimal boilerplate, built-in persist middleware for auth token storage, simpler mental model.

# Deep Learning for Algo Trading — Technical Documentation

Comprehensive, codebase-aligned technical documentation for the AI-powered algorithmic trading platform targeting Indian equity markets (NSE/BSE). This documentation serves as the complete reference for a college final year major project combining Deep Learning, Reinforcement Learning, and Full-Stack Web Development.

## Project Context

| Aspect | Description |
|--------|-------------|
| **Domain** | Algorithmic Trading, Deep Reinforcement Learning, Financial Technology |
| **Target Market** | Indian Equity Markets (NSE/BSE) — 52 NIFTY 50 constituent stocks |
| **Core Pipeline** | LSTM price prediction → PPO trading decision → Behavior-personalized execution |
| **Runtime Modes** | Demo (in-memory), Firebase Auth + Firestore, Full Production (PostgreSQL + Redis) |
| **Institution** | College Final Year Major Project |

## Document Index

| # | Document | Description |
|---|----------|-------------|
| 1 | [System Overview](./01_system_overview.md) | Project purpose, problem statement, objectives, three-layer architecture, technology stack with version rationale, runtime modes comparison, complete repository structure, market data conventions, auth system overview, and deliverables summary |
| 2 | [Backend Architecture](./02_backend_architecture.md) | FastAPI application lifecycle, pydantic-settings configuration, all API routes with deep-dive flows, Layer 1/2/3 modules, services (prediction, backtest, bootstrap, training, storage), complete NSE stocks list, database models with ORM schema, error handling patterns, CORS configuration |
| 3 | [Frontend Architecture](./03_frontend_architecture.md) | Next.js 16 App Router structure, Firebase SDK auth flow with Zustand state management, middleware route protection, WebSocket integration with reconnection strategy, component hierarchy with all props/state, React Hook Form + Zod validation, Recharts charting patterns, utility modules |
| 4 | [ML Pipeline and Models](./04_ml_pipeline_and_models.md) | Complete training pipeline (data download → LSTM → PPO → DeepAR), model architectures with hyperparameters and justification, training results and metrics, inference flows, per-user PPO retraining orchestration, risk-adjusted agent creation, model bootstrap at startup, fallback behavior tree |
| 5 | [API and WebSocket Contract](./05_api_and_websocket_contract.md) | Complete endpoint reference with request/response schemas for all 15+ endpoints, authentication modes (Demo JWT / Firebase Token), WebSocket protocol with subscribe/unsubscribe/ping, error codes, mode-specific behaviors |
| 6 | [Deployment, Configuration, and Limits](./06_deployment_configuration_and_limits.md) | Local setup guide, Docker Compose orchestration, complete environment variable reference table, configuration guide by mode, pre-PR checks, known limitations and implementation gaps (16 items), production readiness checklist, troubleshooting guide for 10 common issues, future scope and roadmap |

## Quick Reference

- **Backend**: Python 3.12, FastAPI, PyTorch (LSTM), Stable-Baselines3 (PPO), Gymnasium, pandas, numpy, scikit-learn, yfinance, pandas-ta, SQLAlchemy, asyncpg, Firebase Admin SDK, python-jose, bcrypt
- **Frontend**: Next.js 16.0.10, React 19.2.1, TypeScript 5 (strict), TailwindCSS 4, Zustand 5, Recharts 3.6, React Hook Form + Zod 4, Axios, Lucide-react, Firebase JS SDK 12.x
- **Infra**: PostgreSQL 15, Redis 7, Docker Compose, Uvicorn
- **Models**: LSTM (2-layer, hidden=64, seq_len=30, ~39K params), PPO (SB3 MlpPolicy [256,256], 5 actions, 30k-50k timesteps), DeepAR (experimental, probabilistic forecasting)
- **API Base**: `/api/v1`, 5 routers + WebSocket, docs at `/docs` (Swagger) and `/redoc` (ReDoc)
- **Auth**: Firebase SDK (frontend) / JWT + Firebase Admin SDK (backend), dual-pathway with demo mode fallback
- **Three-Layer Design**: Layer 1 (market_data → technical_indicators → state_builder) → Layer 2 (trading_env → ppo_agent → reward_function) → Layer 3 (risk_profiler → position_sizer → breakeven_tracker)
- **Key Training Result**: LSTM val loss 0.000228 (23,167 samples), PPO avg return 132.28% (30k timesteps), Sharpe 0.66

## Diagram References

Architecture and flow diagrams are available in the [`images/`](./images/) subdirectory:
- `system_architecture.png`
- `data_flow.png`
- `training_pipeline.png`
- `lstm_architecture.png`
- `ppo_training.png`
- `risk_profiler.png`

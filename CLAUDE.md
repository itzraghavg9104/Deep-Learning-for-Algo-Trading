# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep Learning for Algorithmic Trading** - AI-powered trading platform for Indian Markets (NSE/BSE) using Deep Reinforcement Learning. The system combines LSTM neural networks for price prediction and PPO reinforcement learning for trading decisions.

## Technology Stack

**Backend:**
- FastAPI (Python 3.10+) with async/await patterns
- PyTorch for LSTM model
- Stable-Baselines3 (PPO) for reinforcement learning
- Gymnasium for custom trading environment
- PostgreSQL (asyncpg) + Redis for data/cache
- yfinance + pandas-ta for market data and technical indicators

**Frontend:**
- Next.js 16 with App Router
- React 19, TypeScript
- TailwindCSS 4
- Zustand for state management
- Recharts for visualization

## Development Commands

### Backend
```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate
# Unix
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload        # Run development server
uvicorn app.main:app --host 0.0.0.0 --port 8000   # Production
```

### Frontend
```bash
cd frontend
npm install
npm run dev                          # Development server
npm run build                        # Production build
npm run start                        # Start production server
```

### ML Training Pipeline
```bash
cd backend
.\venv\Scripts\activate

python training/download_data.py     # Download 5 years NIFTY 50 data
python training/train_lstm.py        # Train LSTM price predictor
python training/train_ppo.py         # Train PPO trading agent
```

### Docker (Full Stack)
```bash
docker-compose up --build            # Start all services (PostgreSQL, Redis, Backend, Frontend)
```

## Architecture Overview

The system uses a three-layer architecture:

**Layer 1: Data Processing** (`backend/app/layer1_data_processing/`)
- `market_data.py`: Fetches NSE/BSE data via yfinance with `.NS`/`.BO` suffixes
- `technical_indicators.py`: Computes 30+ indicators via pandas-ta
- `state_builder.py`: Builds state vectors for RL environment

**Layer 2: Decision Engine** (`backend/app/layer2_decision/`)
- `trading_env.py`: Custom Gymnasium environment (actions: 0=HOLD, 1=BUY, 2=SELL)
- `ppo_agent.py`: PPO agent wrapper using Stable-Baselines3
- `reward_function.py`: Sharpe ratio-based reward calculation

**Layer 3: Trader Behavior** (`backend/app/trader_behavior/`)
- `risk_profiler.py`: Risk tolerance assessment (0.0-1.0 scale)
- `position_sizer.py`: Kelly Criterion position sizing
- `breakeven_tracker.py`: Break-even point tracking

## Key Files and Conventions

**Configuration:**
- `backend/app/config.py`: Pydantic-settings configuration (reads from `.env`)
- Required env vars: `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`, `JWT_SECRET`

**API Routes** (`backend/app/api/routes/`):
- `/api/v1/trading/signals/{symbol}` - AI trading signals
- `/api/v1/trading/watchlist` - Top NSE stocks with signals
- `/api/v1/backtest/run` - Strategy backtesting
- `/api/v1/profile/risk-assessment` - Risk profiling

**Models:**
- LSTM model saved to `backend/models/lstm_final.pt`
- PPO agent saved to `backend/models/ppo_trading_final.zip`

**Market Data Conventions:**
- NSE symbols use `.NS` suffix (e.g., `RELIANCE.NS`)
- BSE symbols use `.BO` suffix
- `normalize_symbol()` in `market_data.py` handles suffix normalization

## Development Notes

- All I/O operations in FastAPI must use `async`/`await`
- Use Pydantic models for request/response validation
- The prediction service gracefully falls back to rule-based signals if models are unavailable
- CORS is configured for `http://localhost:3000` in development
- API documentation available at `http://localhost:8000/docs` when running

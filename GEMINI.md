# GEMINI.md

## Project Overview
**Deep Learning for Algorithmic Trading** is an AI-powered trading platform specifically designed for the Indian stock market (NSE/BSE). It employs a sophisticated three-layer architecture to transform raw market data into personalized, risk-adjusted trading decisions.

- **Layer 1: Data Processing**: Fetches real-time data using `yfinance` and `nsepy`, computes 30+ technical indicators via `pandas-ta`, and generates price predictions using LSTM/DeepAR models.
- **Layer 2: Decision Engine**: A Reinforcement Learning (RL) layer using `Stable-Baselines3` (PPO algorithm) and a custom `Gymnasium` environment to optimize trading strategies for maximum risk-adjusted returns (Sharpe Ratio).
- **Layer 3: Trader Behavior**: Personalizes the agent's output based on user risk profiles (0.0-1.0), applies the Kelly Criterion for position sizing, and tracks break-even points.

## Tech Stack
### Backend
- **Framework**: FastAPI (Python 3.10+)
- **Machine Learning**: PyTorch (LSTM), Stable-Baselines3 (PPO), Gymnasium (RL Env)
- **Data Science**: Pandas, NumPy, Pandas-TA, Scikit-learn
- **Database/Cache**: PostgreSQL (SQLAlchemy + asyncpg), Redis
- **Market Data**: yfinance, nsepy

### Frontend
- **Framework**: Next.js 16 (App Router), React 19, TypeScript
- **Styling**: TailwindCSS 4
- **State Management**: Zustand
- **Visualization**: Recharts, Lucide-React
- **Forms**: React Hook Form + Zod

## Building and Running

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL & Redis (optional for core logic, required for full stack)

### Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Unix:
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
- **Application URL**: [http://localhost:3000](http://localhost:3000)

### ML Training Pipeline
1. **Download Data**: `python training/download_data.py` (5 years of NIFTY 50)
2. **Train LSTM**: `python training/train_lstm.py` (Price forecasting)
3. **Train PPO Agent**: `python training/train_ppo.py` (Decision optimization)

## Project Structure
- `backend/app/`: Core FastAPI application.
    - `api/routes/`: Modular API endpoints (trading, backtest, profile).
    - `layer1_data_processing/`: Data ingestion and feature engineering.
    - `layer2_decision/`: RL environment and agent logic.
    - `trader_behavior/`: Risk profiling and position sizing.
- `backend/training/`: Scripts for model training and data acquisition.
- `backend/models/`: Serialized models (LSTM `.pt`, PPO `.zip`).
- `frontend/src/app/`: Next.js pages and layouts.
- `frontend/src/components/`: Reusable React components (charts, forms).
- `docs/`: Comprehensive system documentation and architecture diagrams.

## Development Conventions
- **Asynchronous Code**: Use `async`/`await` for all I/O bound operations in FastAPI.
- **Type Safety**: Use Pydantic models for request/response validation and TypeScript for the frontend.
- **Environment Management**: Configuration is managed via `backend/app/config.py` using `pydantic-settings`.
- **Modularity**: Logic is strictly separated into the three architectural layers to ensure maintainability and testability.
- **Trading Safety**: Always simulate trades in the `TradingEnv` before deploying any logic to production-style endpoints.

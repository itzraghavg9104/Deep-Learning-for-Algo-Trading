# Repository Guidelines

## Stack & Versions
- **Backend**: Python 3.12, FastAPI, PyTorch, Stable-Baselines3 (PPO), Gymnasium.
- **Frontend**: Next.js 16, React 19, TailwindCSS 4, TypeScript strict, Zustand, Recharts, React Hook Form + Zod, Axios, Lucide-React.
- **Infra**: PostgreSQL 15, Redis 7, Docker Compose.

## Quick Start
```bash
# Backend
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload                              # dev :8000
uvicorn app.main:app --host 0.0.0.0 --port 8000            # prod

# Frontend
cd frontend && npm install
npm run dev                                                # dev :3000
npm run lint && npm run build                              # pre-PR checks

# Full stack (Docker)
docker-compose up --build
```

## Demo Mode (No Postgres/Redis)
`backend/app/config.py` sets `DEMO_MODE: bool = True` by default -- the app works without any database or Redis. Override via `backend/.env`:
```env
DEMO_MODE=False
```
Copy `backend/.env.example` to `backend/.env` for custom config. Frontend uses `NEXT_PUBLIC_API_URL` (default `http://localhost:8000/api/v1`).

## Architecture (Three Layers)
1. **`backend/app/layer1_data_processing/`** -- market data (yfinance), technical indicators (pandas-ta), state builder for RL.
2. **`backend/app/layer2_decision/`** -- custom Gymnasium trading env, PPO agent (Stable-Baselines3), Sharpe-ratio reward.
3. **`backend/app/trader_behavior/`** -- risk profiler (0.0-1.0), Kelly Criterion position sizer, breakeven tracker.

Fallback: prediction service (`backend/app/services/prediction_service.py`) uses rule-based signals if models unavailable.

## API Endpoints (prefix `/api/v1`)
| Prefix         | File                                |
|----------------|-------------------------------------|
| `/trading`     | `backend/app/api/routes/trading.py` |
| `/backtest`    | `backend/app/api/routes/backtest.py`|
| `/profile`     | `backend/app/api/routes/profile.py` |
| `/auth`        | `backend/app/api/routes/auth.py`    |
| WebSocket      | `backend/app/api/websocket.py`      |

Docs at `http://localhost:8000/docs`.

## ML Training Pipeline
```bash
cd backend && source venv/bin/activate
python training/download_data.py       # 5y NIFTY 50 -> data/training_data.csv
python training/train_lstm.py          # -> models/lstm_final.pt
python training/train_ppo.py           # -> models/ppo_trading_final.zip
```
Model files (`*.pt`, `*.zip`) are gitignored.

## Market Data Conventions
- NSE symbols: `.NS` suffix (`RELIANCE.NS`)
- BSE symbols: `.BO` suffix
- `normalize_symbol()` in `market_data.py` handles suffix normalization.

## Frontend Routing & Auth
- **Protected pages** (redirect to `/auth/login` if unauthenticated): `/dashboard`, `/profile`, `/backtest`, `/trades` -- defined in `frontend/src/middleware.ts`.
- Auth pages redirect to `/dashboard` if already logged in.
- JWT stored in localStorage under `auth-storage` key; cookie `auth_token` set for middleware.

## Testing
- No tests committed yet (`backend/tests/` is empty, no frontend test runner configured).
- Backend: `pytest` + `pytest-asyncio` available. New tests go in `backend/tests/test_*.py`.
- Frontend: at minimum `npm run lint && npm run build` before PRs.

## Path Aliases
- Frontend: `@/*` maps to `src/*` (tsconfig paths).
- Backend: standard Python imports, all app code under `backend/app/`.

## Notable Config
- CORS origins: `["http://localhost:3000", "http://127.0.0.1:3000"]` -- update in `config.py` for other domains.
- `.gitignore` excludes: `__pycache__/`, `.env`, `venv/`, `node_modules/`, `.next/`, `*.pt`, `*.zip`, `*.pth`, `*.db`.

## Verified Runtime Behavior (Important for Agents)
- **Demo mode defaults ON** (`DEMO_MODE=True` in `backend/app/config.py`), so DB/Redis are optional for local dev.
- **Demo auth behavior is intentionally permissive**:
  - `/auth/login` auto-creates users and skips password verification in demo mode.
  - `/auth/me` can auto-provision token subjects not yet present.
- **Frontend auth contract**:
  - Zustand persistence key: `auth-storage` (token stored in localStorage).
  - Middleware auth cookie: `auth_token` (used for route guarding in `frontend/src/middleware.ts`).
- **WebSocket contract**:
  - Endpoint: `/api/v1/ws/prices`
  - Actions accepted: `subscribe`, `unsubscribe`, `set`, `ping`
  - Price pushes occur about every 30s while symbols are subscribed.

## Working Directory Rules (Avoid Path Bugs)
- Run backend app/training commands from `backend/` unless a script explicitly expects repo root.
- Relative model paths are resolved from backend process CWD (`MODEL_PATH=./models` by default).
- `BacktestService` currently uses `data_dir="backend/data/raw"` internally; if backend is launched from `backend/`, this may resolve incorrectly (double `backend/...`). Prefer validating CWD/path assumptions before editing related code.

## Current Inconsistencies to Be Aware Of
- `backend/.env.example` does **not** currently include `DEMO_MODE`, even though runtime supports it.
- `backend/Dockerfile` has a malformed multiline `RUN apt-get ...` instruction (line continuations missing), so Docker backend build may fail until fixed.
- Frontend Dockerfile uses `node:18-slim` while project docs state Node 20+ for local dev.

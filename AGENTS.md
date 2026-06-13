# AGENTS.md

## Auth Modes (Critical to Understand)

**Two independent auth paths** — the frontend and backend can disagree.

| Config | Backend behavior | Frontend behavior |
|--------|-----------------|-------------------|
| `FIREBASE_AUTH_ENABLED=true` | Verifies Firebase ID token via Admin SDK | Login/register always calls Firebase SDK (`signInWithEmailAndPassword`) |
| `DEMO_MODE=true` (default) | Uses local JWT, auto-creates users, skips password | Sends Firebase ID token → backend `jwt.decode` fails → 401 |

**Practical effect:** to run locally you must either (a) configure Firebase on both sides, or (b) modify `auth-store.ts` to call `authApi.login` (form-post to backend) instead of Firebase SDK in demo mode. The pre-committed `.env.local` has real Firebase keys, and the backend's `.env.example` has `DEMO_MODE=true`.

## Action Space (5 actions, not 3)

`backend/app/layer2_decision/action_space.py`:
- 0=HOLD BUY, 1=HOLD SELL, 2=BUY, 3=SELL, 4=IDLE

The old CLAUDE.md/GEMINI.md claim "0=HOLD, 1=BUY, 2=SELL" is **wrong**.

## Running

```bash
# Backend (run from backend/)
cd backend && source venv/bin/activate
uvicorn app.main:app --reload              # dev on :8000

# Frontend
cd frontend && npm run dev                 # dev on :3000
npm run lint && npm run build              # pre-PR checks

# Training pipeline (from backend/)
python training/download_data.py           # → data/training_data.csv
python training/train_lstm.py              # → models/lstm_final.pt
python training/train_ppo.py               # → models/ppo_trading_final.zip

# Full stack
docker-compose up --build
```

Model files (`*.pt`, `*.zip`) are gitignored.

## Architecture

Three layers under `backend/app/`:
1. `layer1_data_processing/` — yfinance market data, pandas-ta indicators, state builder for RL
2. `layer2_decision/` — Gymnasium trading env, PPO agent (Stable-Baselines3), Sharpe-ratio reward
3. `trader_behavior/` — risk profiler (0.0–1.0), Kelly Criterion position sizer, breakeven tracker

Models used: LSTM (price prediction), PPO (trading decisions), DeepAR (probabilistic forecasting). Fallback in `prediction_service.py` uses rule-based signals if models unavailable.

## Config Defaults vs Reality

| Setting | `config.py` default | What training actually produces |
|---------|-------------------|-------------------------------|
| `PPO_MODEL` | `ppo_agent_v1.zip` | `ppo_trading_final.zip` |
| `DEEPAR_MODEL` | `deepar_v1.pt` | — |

`AUTO_TRAIN_IF_MISSING=True` bootstraps models on backend startup if absent.

## Working Directory

- Always run backend commands from `backend/`. Relative paths (`./models`, `./data/`) break otherwise.
- `BacktestService` resolves `data_dir` via `Path(__file__)` (not CWD), so it's safe from any CWD.

## WebSocket

- Endpoint: `/api/v1/ws/prices`
- Actions: `subscribe`, `unsubscribe`, `set`, `ping`
- Price pushes every ~30s while symbols subscribed.

## Frontend Conventions

| Concern | Detail |
|---------|--------|
| Path alias | `@/*` → `src/*` (tsconfig paths) |
| Auth persistence | Zustand key `auth-storage` (token in localStorage) |
| Middleware cookie | `auth_token` for route guarding |
| Protected routes | `/dashboard`, `/profile`, `/backtest`, `/trades` |
| Firebase init | `src/lib/firebase.ts` reads `NEXT_PUBLIC_FIREBASE_*` env vars |

## Market Data

- NSE: `.NS` suffix (`RELIANCE.NS`), BSE: `.BO` suffix
- `normalize_symbol()` in `market_data.py` handles suffix normalization.

## Known Issues

- **Frontend Dockerfile** uses `node:18-slim`; project docs say Node 20+ for local dev.
- **Frontend auth-store** always uses Firebase SDK — doesn't offer a non-Firebase fallback for demo mode.
- **Backend CORS** from `settings.CORS_ORIGINS` (list), default: `["http://localhost:3000", "http://127.0.0.1:3000"]`.
- No tests exist yet (`backend/tests/` empty). Backend has `pytest` + `pytest-asyncio` available.

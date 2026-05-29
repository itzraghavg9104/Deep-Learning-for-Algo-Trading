# Setup Guide (Complete End-to-End)

This is the complete setup runbook for local development, Firebase mode, model training, and production-ready configuration.

For security and hardening practices, also read `docs/SAFE_GUIDE.md`.

---

## 1) Prerequisites

- Python `3.12+`
- Node.js `20+` (recommended for local)
- npm `10+`
- Git
- Optional: Docker + Docker Compose
- Optional (for Firebase mode): Firebase project with Auth + Firestore

---

## 2) Repository Bootstrap

```bash
git clone <your-repo-url>
cd Deep-Learning-for-Algo-Trading
```

---

## 3) Backend Setup

### 3.1 Create environment and install

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Create backend env file

```bash
cp .env.example .env
```

### 3.3 Minimal local `.env` (Demo Mode)

Use this first for quick start without Postgres/Redis:

```env
APP_ENV=development
DEBUG=true
DEMO_MODE=true
FIREBASE_AUTH_ENABLED=false

SECRET_KEY=dev-only-change-me
JWT_SECRET=dev-only-change-me
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/algotrading
REDIS_URL=redis://localhost:6379/0
MODEL_PATH=./models

FIREBASE_PROJECT_ID=
FIREBASE_WEB_API_KEY=
FIREBASE_SERVICE_ACCOUNT_PATH=
FIRESTORE_DATABASE_ID=(default)
```

### 3.4 Run backend

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

Backend URLs:
- API root: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## 4) Frontend Setup

### 4.1 Install dependencies

```bash
cd frontend
npm install
```

### 4.2 Create frontend env

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_FIREBASE_API_KEY=
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=
NEXT_PUBLIC_FIREBASE_PROJECT_ID=
NEXT_PUBLIC_FIREBASE_APP_ID=
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=
```

Do not commit local env files or secret bundles. This repo ignores `.env`, `.env.local` (including nested folders), and `secrets/`.

### 4.3 Run frontend

```bash
cd frontend
npm run dev
```

Frontend URL:
- `http://localhost:3000`

---

## 5) Auth Modes (Critical)

### Mode A: Demo/Local Auth

Set:
- `DEMO_MODE=true`
- `FIREBASE_AUTH_ENABLED=false`

Behavior:
- `/api/v1/auth/login` and `/api/v1/auth/register` are active.
- Demo mode is intentionally permissive for local demos.

### Mode B: Firebase Auth

Set:
- `DEMO_MODE=false` (recommended)
- `FIREBASE_AUTH_ENABLED=true`
- Fill Firebase env values

Behavior:
- Backend expects Firebase ID token in `Authorization: Bearer <id_token>`.
- `/api/v1/auth/login` and `/api/v1/auth/register` return `400` by design.

---

## 6) Firebase Setup (If Using Mode B)

1. Create/select Firebase project.
2. Enable Authentication provider(s).
3. Enable Firestore in Native mode.
4. Generate service account JSON and store securely outside git.
5. Fill backend:
   - `FIREBASE_PROJECT_ID`
   - `FIREBASE_WEB_API_KEY`
   - `FIREBASE_SERVICE_ACCOUNT_PATH`
6. Fill frontend:
   - `NEXT_PUBLIC_FIREBASE_API_KEY`
   - `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`
   - `NEXT_PUBLIC_FIREBASE_PROJECT_ID`
   - `NEXT_PUBLIC_FIREBASE_APP_ID`
   - `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID`

---

## 7) Optional Infra Setup (Postgres/Redis + Docker)

### 7.1 Docker Compose

From repo root:

```bash
docker-compose up --build
```

Services:
- Backend on `:8000`
- Frontend on `:3000`
- Postgres on `:5432`
- Redis on `:6379`

### 7.2 Notes

- Demo mode can run without DB/Redis.
- For persistent runtime behavior, set `DEMO_MODE=false` and use real stores.

---

## 8) Model Training Pipeline

Run from `backend/`:

```bash
source venv/bin/activate
python training/download_data.py
python training/train_lstm.py
python training/train_ppo.py
```

Expected outputs:
- `backend/data/training_data.csv`
- `backend/models/lstm_final.pt`
- `backend/models/ppo_trading_final.zip`

---

## 9) Smoke Test Checklist

1. Backend starts and `/health` returns healthy.
2. Frontend loads and connects to API.
3. Login works for chosen auth mode.
4. `GET /api/v1/trading/watchlist` returns data.
5. `GET /api/v1/trading/market/RELIANCE.NS` returns history + indicators.
6. `POST /api/v1/profile/risk-assessment` succeeds.
7. `POST /api/v1/backtest/run` returns a result.

---

## 10) Useful API Examples

### 10.1 Behavior assessment

```bash
curl -X POST http://localhost:8000/api/v1/profile/behavior-assessment \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "answers": {
      "capital_per_trade_pct": 5,
      "tp_sl_ratio": 2.0,
      "max_profit_close_pct": 12,
      "max_trades_per_day": 6,
      "post_loss_rest_min": 30,
      "max_drawdown_pct": 15,
      "intraday_var_pct": 3,
      "entry_slippage_bps": 10,
      "news_buffer_min": 45,
      "partial_tp_frequency": 3,
      "breakeven_trigger_pct": 1,
      "breakeven_migration_time_min": 60
    }
  }'
```

### 10.2 Trade evaluation

```bash
curl -X POST http://localhost:8000/api/v1/profile/trades/evaluate \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "trade_id": "T-1001",
    "symbol": "RELIANCE.NS",
    "planned": { "capital_per_trade_pct": 0.05 },
    "executed": {
      "capital_per_trade_pct": 0.07,
      "cooldown_respected": false
    },
    "pnl": -1200.5,
    "pnl_pct": -1.2
  }'
```

---

## 11) Production Configuration (Mandatory)

Use at minimum:

```env
APP_ENV=production
DEBUG=false
DEMO_MODE=false
FIREBASE_AUTH_ENABLED=true
SECRET_KEY=<strong-random-secret>
JWT_SECRET=<strong-random-secret>
```

Important:
- Backend now blocks startup in production if unsafe defaults are detected.
- Set strict `CORS_ORIGINS` to actual frontend domains.
- Keep service account file out of repo and rotate on exposure.

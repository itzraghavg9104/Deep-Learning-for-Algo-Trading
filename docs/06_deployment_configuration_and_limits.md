# 6. Deployment, Configuration, and Current Limits

## 6.1 Local Development

Backend:

- create venv, install `backend/requirements.txt`
- run `uvicorn app.main:app --reload` from `backend/`

Frontend:

- install dependencies in `frontend/`
- run `npm run dev`

## 6.2 Environment and Config

Backend key variables:

- `APP_ENV`, `DEBUG`
- `DEMO_MODE`
- `DATABASE_URL`, `REDIS_URL`
- `JWT_SECRET`, `JWT_ALGORITHM`, `JWT_EXPIRY_HOURS`
- `MODEL_PATH`
- Firebase variables for optional auth/storage

Frontend key variable:

- `NEXT_PUBLIC_API_URL`

## 6.3 Docker Setup

`docker-compose.yml` includes services:

- `db` (PostgreSQL 15)
- `redis` (Redis 7)
- `backend`
- `frontend`

Backend Dockerfile currently installs Python deps and runs uvicorn.
Frontend Dockerfile builds and runs Next.js app.

## 6.4 Operational Notes for This Codebase

- Demo mode defaults ON and bypasses persistent DB usage.
- In demo mode, auth and user data are in-memory only.
- Model availability is optional at runtime due to built-in fallbacks.

## 6.5 Known Implementation Gaps (Current Code)

These are relevant for honest reporting and future work:

- Backtest route uses a placeholder `user_id = 1` (full user-linked persistence pending).
- `BacktestService` default `data_dir="backend/data/raw"` can be sensitive to backend launch CWD.
- `backend/.env.example` does not currently list `DEMO_MODE` even though app uses it.
- Frontend Docker image uses Node 18 while local docs target Node 20+.

## 6.6 Suggested Validation Checklist Before Demo/Submission

- Confirm backend started from `backend/` so relative paths resolve correctly.
- Confirm `NEXT_PUBLIC_API_URL` points to active backend.
- Confirm model files exist in `backend/models/` if model-based signals are expected.
- Confirm route guard behavior by testing unauthenticated access to protected pages.
- Confirm WebSocket stream updates with an active symbol subscription.

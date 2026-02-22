# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI service and ML/trading logic.
- `backend/app/api/routes/`: REST endpoints (`trading.py`, `backtest.py`, `profile.py`, `auth.py`).
- `backend/app/layer1_data_processing/`, `backend/app/layer2_decision/`, `backend/app/trader_behavior/`: core domain modules.
- `backend/training/`: model/data scripts (`download_data.py`, `train_lstm.py`, `train_ppo.py`).
- `backend/models/` and `backend/data/`: trained artifacts and datasets.
- `frontend/src/app/`: Next.js App Router pages; `frontend/src/components/`: UI components; `frontend/src/lib/`: API/store utilities.
- `docs/`: architecture and API documentation; `references/`: research material.

## Build, Test, and Development Commands
- Backend setup: `cd backend && python -m venv venv && .\venv\Scripts\activate && pip install -r requirements.txt`
- Run backend: `cd backend && .\venv\Scripts\activate && uvicorn app.main:app --reload`
- Frontend setup: `cd frontend && npm install`
- Run frontend dev server: `cd frontend && npm run dev`
- Frontend production checks: `cd frontend && npm run lint && npm run build`
- Full stack with containers: `docker-compose up --build` (frontend `:3000`, backend `:8000`, Postgres `:5432`, Redis `:6379`).

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes.
- TypeScript/React: 4-space indentation in current codebase, component files in `PascalCase` (for example `SignalCard.tsx`), hooks/utilities in `camelCase`.
- Keep modules focused by layer (data processing, decision, behavior) and avoid cross-layer leakage.
- Frontend linting uses ESLint (`frontend/eslint.config.mjs` with Next core-web-vitals + TypeScript rules).

## Testing Guidelines
- Backend test dependencies are available (`pytest`, `pytest-asyncio`), but there are currently no committed tests in `backend/tests/`.
- Add backend tests under `backend/tests/` using `test_*.py`; run with `cd backend && .\venv\Scripts\activate && pytest`.
- No frontend test runner is configured yet; at minimum, run `npm run lint` and `npm run build` before PRs.

## Commit & Pull Request Guidelines
- Follow existing history style: short imperative subjects (for example, `Add backend...`, `Update documentation...`).
- Keep commits scoped to one concern (API, frontend, training, docs).
- PRs should include: purpose, key changes, validation steps run, and linked issue/task.
- Include screenshots/GIFs for UI changes and example request/response payloads for API changes.

## Security & Configuration Tips
- Copy `backend/.env.example` to `.env` for local secrets; never commit credentials or model artifacts with sensitive data.
- Verify `NEXT_PUBLIC_API_URL`, `DATABASE_URL`, and `REDIS_URL` per environment before deploying.

# Demo Mode (No Postgres/Redis)

This project can run without Postgres or Redis by enabling `DEMO_MODE`. In demo mode, all data persistence is replaced by a lightweight in-memory store that resets on restart.

## Why Demo Mode Exists

- Local development should not require Postgres or Redis.
- Frontend screens can be demonstrated end-to-end with predictable responses.
- Core model and market data logic can be exercised without infrastructure.

## How Demo Mode Works

- `backend/app/services/demo_store.py` is a process-local, in-memory store.
- It holds users and backtests in Python dictionaries.
- It is used only when `DEMO_MODE=True`.
- It does not persist across server restarts.

## What Is Bypassed

- Postgres connectivity and ORM operations.
- Any Redis-based caching (not currently implemented in code).

## Affected Endpoints

- `POST /api/v1/auth/register`
  - Input: JSON `{"email": "...", "password": "..."}`.
  - Output: User object from demo store.
- `POST /api/v1/auth/login`
  - Input: `application/x-www-form-urlencoded` `username`, `password`.
  - Output: JWT token based on demo store user.
- `GET /api/v1/auth/me`
  - Input: `Authorization: Bearer <token>`.
  - Output: User object from demo store.
- `POST /api/v1/backtest/run`
  - Input: Backtest config.
  - Output: Backtest metrics and `backtest_id` from demo store.

## Behavior That Remains Fully Real

- Market data fetching uses live `yfinance` calls.
- Model inference uses the trained LSTM/PPO files if present.
- Risk questionnaire logic and trader behavior logic are unchanged.

## Enable Demo Mode

`backend/app/config.py` defaults `DEMO_MODE=True`.

To override explicitly in `.env`:

```env
DEMO_MODE=True
```

## Known Limitations

- Restarting the backend clears all demo users and backtest history.
- Multi-process deployments will not share demo data.
- Demo mode is not safe for production use.

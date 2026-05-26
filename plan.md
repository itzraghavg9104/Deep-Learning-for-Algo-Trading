# Implementation Plan — Complete

All items implemented and verified (`npm run lint && npm run build` passes).

## Phase 1 — Backend: Flesh Out Stubs ✅

### 1.1 `GET /backtest/{backtest_id}` ✅
- `backend/app/services/demo_store.py` — added `get_backtest(id)` method
- `backend/app/api/routes/backtest.py:94-100` — implemented lookup using `demo_store.get_backtest()`

### 1.2-1.3 `GET /profile` + `PUT /preferences` ✅
- `backend/app/services/demo_store.py` — added `DemoProfile` dataclass + `get_or_create_profile`/`update_profile` methods
- `backend/app/api/routes/profile.py` — added `get_current_user` dependency, persist to `demo_store`

### 1.4 New `GET /trades` endpoint ✅
- `backend/app/services/demo_store.py` — added `get_user_trades()` collecting trades from stored backtests
- `backend/app/api/routes/profile.py` — added `GET /trades` route with auth

## Phase 2 — Frontend: Wire Up & Polish ✅

### 2.1 Wire trades page ✅
- `frontend/src/lib/api.ts` — added `tradesApi.getTradeHistory()`
- `frontend/src/app/trades/page.tsx` — replaced `SAMPLE_TRADES` with live `GET /profile/trades` API call, added loading/error states

### 2.2 Connect Sidebar risk profile ✅
- `frontend/src/components/dashboard/Sidebar.tsx` — fetches from `profileApi.getProfile()` on mount, shows real category/tolerance

### 2.3 Fix metadata ✅
- `frontend/src/app/layout.tsx` — set proper title + description

### 2.4 Delete dead code ✅
- `frontend/src/lib/store.ts` — removed unused Zustand store

## Phase 3 — Verification ✅
- `npm run lint` — clean
- `npm run build` — compiles and produces all 9 routes

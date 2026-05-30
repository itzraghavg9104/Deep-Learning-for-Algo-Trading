# 3. Frontend Architecture

## 3.1 Stack

- Next.js 16 (App Router)
- React 19
- TypeScript
- TailwindCSS 4
- Zustand (auth state)
- Axios (API client)
- Recharts (charts)
- Firebase JS SDK (auth provider)

## 3.2 Route Structure

Primary pages under `frontend/src/app/`:

- `/` landing page
- `/dashboard`
- `/backtest`
- `/profile`
- `/profile/risk-assessment`
- `/trades`
- `/auth/login`
- `/auth/register`

## 3.3 Route Protection and Session Gate

`frontend/src/middleware.ts`:

- protected prefixes: `/dashboard`, `/profile`, `/backtest`, `/trades`
- redirects unauthenticated users to `/auth/login`
- redirects authenticated users away from auth pages to `/dashboard`
- middleware checks `auth_token` cookie

## 3.4 Client Auth State

`frontend/src/lib/auth-store.ts`:

- Zustand persisted store key: `auth-storage`
- token + auth status persisted in localStorage
- cookie synchronization:
  - set cookie on login/register/google login
  - clear cookie on logout
- `initializeAuth()` restores auth header and re-fetches `/auth/me`

Note: frontend auth flow is Firebase-based. Backend supports JWT demo mode and optional Firebase token verification.

## 3.5 API Integration

`frontend/src/lib/api.ts`:

- base URL: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000/api/v1`)
- request interceptor injects bearer token from `auth-storage`
- response interceptor clears session and redirects to login on 401

API wrappers provided for:

- auth
- trading
- backtest
- profile
- trades

## 3.6 Real-Time Feed

`frontend/src/lib/use-websocket.ts`:

- derives socket URL from API base URL
- endpoint: `/api/v1/ws/prices`
- supports subscribe/unsubscribe and custom payload sending
- includes reconnection with exponential backoff

## 3.7 UI Building Blocks

Component groups:

- `components/dashboard/`: sidebar, KPI cards, signal cards
- `components/charts/`: price chart, technical indicators, equity curve, gauge, sparkline
- `components/forms/`: backtest config and risk questionnaire

These components consume backend contracts directly and power report-ready visuals for strategy and behavior outputs.

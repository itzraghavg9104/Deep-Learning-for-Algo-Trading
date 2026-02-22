# Frontend Reference (Detailed)

This document describes the frontend structure, key files, and the input/output of each page and component.

## App Router Pages

`frontend/src/app/dashboard/page.tsx`

- Loads watchlist signals.
- Subscribes to WebSocket updates.
- Renders signal cards, stats, and a detail modal.

Inputs

- Watchlist API response.
- WebSocket price updates.

Outputs

- Visible signal grid.
- Price chart modal.

`frontend/src/app/backtest/page.tsx`

- Shows a configuration form and results.
- Calls backtest API and renders `EquityCurve`.

Inputs

- Backtest form data.

Outputs

- Metrics grid, trades table, equity curve.

`frontend/src/app/profile/page.tsx`

- Loads user profile and preferences.
- Allows updating preferences.

Inputs

- Profile API response.
- Preferences form fields.

Outputs

- Updated preferences.

`frontend/src/app/profile/risk-assessment/page.tsx`

- Renders risk questionnaire.
- Submits to risk assessment API.

Inputs

- Questionnaire answers.

Outputs

- Risk profile summary.

`frontend/src/app/auth/login/page.tsx`

- Login form.
- Calls `/auth/login`.

Inputs

- Email and password.

Outputs

- Auth token stored in Zustand and cookie.

`frontend/src/app/auth/register/page.tsx`

- Registration form.
- Calls `/auth/register`.

Inputs

- Email and password.

Outputs

- Auto-login and token storage.

`frontend/src/app/trades/page.tsx`

- Demo trade history with filtering and CSV export.
- Uses local demo data.

## Components

`frontend/src/components/dashboard/SignalCard.tsx`

- Displays one signal.
- Optional sparkline.
- Supports flash styling for live updates.

Inputs

- `symbol`, `price`, `change_pct`, `action`, `confidence`, `sparkline`, `flash`.

Outputs

- Card UI.

`frontend/src/components/dashboard/StatsCard.tsx`

- Displays summary metrics.

`frontend/src/components/forms/RiskQuestionnaire.tsx`

- Multi-question form.
- Submits answers to `/profile/risk-assessment`.

`frontend/src/components/forms/BacktestConfig.tsx`

- Backtest configuration form.

`frontend/src/components/charts/EquityCurve.tsx`

- Renders equity curve time series.

`frontend/src/components/charts/PriceChart.tsx`

- Renders price and volume history using Recharts.

`frontend/src/components/charts/TechnicalIndicators.tsx`

- Displays indicator values.

`frontend/src/components/charts/SignalGauge.tsx`

- Shows confidence as a progress bar.

`frontend/src/components/charts/Sparkline.tsx`

- Small line chart for signal cards.

## Lib Utilities

`frontend/src/lib/api.ts`

- Axios client with token injection.
- Exposes `authApi`, `tradingApi`, `backtestApi`, `profileApi`.

Inputs

- Auth token from local storage.

Outputs

- API responses as JSON.

`frontend/src/lib/auth-store.ts`

- Zustand store for auth state.
- `login`, `register`, `logout`, `fetchUser`.

Inputs

- Credentials, JWT tokens.

Outputs

- Auth state, user state.

`frontend/src/lib/use-websocket.ts`

- WebSocket hook with auto reconnect.
- `subscribe`, `unsubscribe`, `sendJson`.

Inputs

- WebSocket messages.

Outputs

- Connection status and callbacks.

`frontend/src/lib/market-hours.ts`

- Computes NSE market open/closed state.

`frontend/src/middleware.ts`

- Protects routes based on `auth_token` cookie.

# 3. Frontend Architecture

## 3.1 Technology Stack

| Component | Technology | Version | Role |
|-----------|-----------|---------|------|
| Framework | Next.js | 16.0.10 | App Router, SSR/SSG, middleware route protection |
| UI Library | React | 19.2.1 | Component model, hooks, server/client components |
| Language | TypeScript | 5.x | Strict mode, `@/*` → `src/*` path aliases |
| Styling | TailwindCSS | 4.x | Utility-first CSS, custom theme |
| State Management | Zustand | 5.x | Auth state with persist middleware (localStorage) |
| HTTP Client | Axios | latest | API communication with request/response interceptors |
| Charts | Recharts | 3.6 | Price, equity, indicator charts (ComposedChart, LineChart) |
| Forms | React Hook Form | 4.x | Form state management with inline validation |
| Validation | Zod | 4.x | Schema-based form validation |
| Auth Provider | Firebase JS SDK | 12.x | Email/password + Google OAuth (signInWithPopup) |
| Icons | Lucide React | latest | UI icon set |
| Build Tool | Turbopack | — | Development bundling via Next.js |

## 3.2 Route Structure

All pages use the App Router pattern in `frontend/src/app/`:

| Route | File | Type | Description |
|-------|------|------|-------------|
| `/` | `page.tsx` | Server | Landing page about the two-stage architecture |
| `/auth/login` | `auth/login/page.tsx` | Client | Email/password + Google OAuth login |
| `/auth/register` | `auth/register/page.tsx` | Client | Registration with password confirmation |
| `/dashboard` | `dashboard/page.tsx` | Client | Main trading signals dashboard |
| `/dashboard` | `dashboard/layout.tsx` | Server | Dashboard shell with Sidebar |
| `/backtest` | `backtest/page.tsx` | Client | Backtest configuration and results |
| `/trades` | `trades/page.tsx` | Client | Trade history with filters |
| `/profile` | `profile/page.tsx` | Client | Risk profile and preferences |
| `/profile/risk-assessment` | `profile/risk-assessment/page.tsx` | Server | Wraps RiskQuestionnaire component |

## 3.3 Middleware & Route Protection (`middleware.ts`)

The middleware runs on every request to matching paths using the `matcher` config:

**Protected Routes** (redirect to `/auth/login?next=<path>` if no `auth_token` cookie):
- `/dashboard/:path*`
- `/profile/:path*`
- `/backtest/:path*`
- `/trades/:path*`

**Auth Routes** (redirect to `/dashboard` if `auth_token` cookie present):
- `/auth/:path*`

**Matcher Configuration:**
```typescript
export const config = {
  matcher: ['/dashboard/:path*', '/profile/:path*', '/backtest/:path*', '/trades/:path*', '/auth/:path*'],
};
```

The middleware checks for the `auth_token` cookie, which is set by the Zustand store on successful authentication. This is a client-side cookie with:
- 1-day expiry
- `SameSite=Lax`
- Path: `/`
- Set via `document.cookie` in the auth store

### Auth State Machine

```
                     ┌─────────────┐
                     │  App Load   │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ AuthInitial │
                     │  izer.run() │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
      ┌───────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
      │ Token exists  │ │ No     │ │ Token      │
      │ + valid      │ │ token  │ │ expired/   │
      └───────┬──────┘ └───┬────┘ │ invalid    │
              │             │      └─────┬──────┘
              ▼             ▼            │
     ┌────────────────┐  ┌──────────┐    │
     │ fetchUser()   │  │ Show     │    │
     │ → /auth/me    │  │ landing/ │    │
     │ → success:    │  │ auth     │    │
     │   dashboard   │  │ page     │    │
     │ → fail:       │  └──────────┘    │
     │   logout()    │                  │
     └────────────────┘                  │
              │                         │
              └─────────────────────────┘
                        │
                 ┌──────▼──────┐
                 │ Authenticated│
                 │   Session   │
                 └──────┬──────┘
                        │
              ┌─────────┼─────────┐
              │         │         │
       ┌──────▼──┐ ┌───▼───┐ ┌───▼──────┐
       │ Logout  │ │ 401   │ │ Token    │
       │ Click   │ │ Resp  │ │ Expiry   │
       └──────┬──┘ └───┬───┘ └───┬──────┘
              └────────┼─────────┘
                       ▼
              ┌────────────────┐
              │ Clear state:  │
              │ - localStorage│
              │ - cookie      │
              │ - Axios header│
              │ Redirect login│
              └────────────────┘
```

## 3.4 Authentication Flow

### Firebase Auth Integration

The frontend uses Firebase as its sole authentication provider. The backend verifies Firebase ID tokens via Firebase Admin SDK.

**Auth Store (Zustand)** — `lib/auth-store.ts`:

```typescript
interface AuthState {
  user: User | null;
  token: string | null;           // Firebase ID token (or JWT in demo mode)
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login(email: string, password: string): Promise<void>;
  register(email: string, password: string): Promise<void>;
  loginWithGoogle(): Promise<void>;
  logout(): void;
  fetchUser(): Promise<void>;
  clearError(): void;
}
```

**Persistence:** Store state persisted to localStorage under key `"auth-storage"`. Only `{ state: { token, isAuthenticated } }` is persisted via Zustand's `persist` middleware.

### Login Flow
```
1. User submits email/password
2. Firebase SDK: signInWithEmailAndPassword(auth, email, password)
3. Extract idToken from UserCredential.user.getIdToken()
4. Store idToken in Zustand state (token)
5. Set auth_token cookie (document.cookie, 1-day expiry)
6. Set Authorization: Bearer <token> on Axios default headers
7. Call GET /auth/me to fetch user profile (auto-provisions in demo)
8. On success: redirect to /dashboard
9. On failure: display error message, clear auth state
```

### Registration Flow
```
1. User submits email/password + confirmation
2. Firebase SDK: createUserWithEmailAndPassword(auth, email, password)
3. Extract idToken from UserCredential
4. Same post-auth flow as login (token → cookie → fetchUser → redirect)
```

### Google OAuth Flow
```
1. User clicks "Continue with Google"
2. Firebase SDK: signInWithPopup(auth, googleProvider)
3. Same post-auth flow as login
```

### Logout Flow
```
1. Clear Authorization header from Axios (api.defaults.headers.common)
2. Firebase SDK: signOut(auth) — catch errors silently
3. Clear auth_token cookie (set expiry to past date)
4. Reset Zustand store (user: null, token: null, isAuthenticated: false)
```

### App Initialization (`AuthInitializer.tsx`)
Called from root layout:
1. Read persisted state from Zustand store (`useAuthStore.getState()`)
2. If token exists: set cookie + Axios header, call `fetchUser()`
3. If fetchUser fails (401): call `logout()` clearing everything
4. Renders children after initialization completes

### Auth Header Injection
Two mechanisms ensure the token is sent with every request:
1. **Axios request interceptor** (`lib/api.ts`): Reads token from localStorage (`auth-storage` → `state.token`) on every request
2. **Store-level header setting**: On login/register, sets `api.defaults.headers.common["Authorization"]`

### 401 Response Handling
The Axios response interceptor in `api.ts`:
1. On 401 response: clears localStorage (`removeItem("auth-storage")`), clears cookie, redirects to `/auth/login`
2. Runs only in browser (`typeof window !== "undefined"`)

## 3.5 API Integration (`lib/api.ts`)

**Shared Axios Instance:**
```typescript
const api = axios.create({
  baseURL: NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1",
  headers: { "Content-Type": "application/json" },
});
```

**Request Interceptor:**
```typescript
api.interceptors.request.use((config) => {
  const stored = localStorage.getItem("auth-storage");
  if (stored) {
    const { state } = JSON.parse(stored);
    if (state?.token) {
      config.headers.Authorization = `Bearer ${state.token}`;
    }
  }
  return config;
});
```

**Response Interceptor:**
```typescript
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && typeof window !== "undefined") {
      localStorage.removeItem("auth-storage");
      document.cookie = "auth_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
      window.location.href = "/auth/login";
    }
    return Promise.reject(error);
  }
);
```

### API Module Structure

| Module | Functions | Endpoints Called |
|--------|-----------|-----------------|
| `authApi` | login, register, getMe | `POST /auth/login`, `POST /auth/register`, `GET /auth/me` |
| `tradingApi` | getSignal, getMarketData, getWatchlist | `GET /trading/signals/{symbol}`, `GET /trading/market/{symbol}`, `GET /trading/watchlist` |
| `backtestApi` | runBacktest | `POST /backtest/run` |
| `profileApi` | getProfile, submitRiskAssessment, submitBehaviorAssessment, getModelTrainingStatus, updatePreferences | `GET /profile`, `POST /profile/risk-assessment`, `POST /profile/behavior-assessment`, `GET /profile/model-training-status`, `PUT /profile/preferences` |
| `tradesApi` | getTradeHistory | `GET /profile/trades` |

Note: `authApi.login`, `authApi.register`, and `authApi.getMe` are **dead code** — the auth store uses Firebase directly and calls `api.get("/auth/me")` inline rather than through `authApi`.

### Complete State Flow Diagram

```
User Action → React Hook Form validation (Zod) ←→ Form State
                  │
                  ▼
          API Module (api.ts)
                  │
          Axios Interceptor (adds Bearer token)
                  │
                  ▼
          FastAPI Backend
                  │
                  ▼
          Response → Zustand Store (auth) → UI re-render
                  │
                  ▼
          Axios Response Interceptor (401 → logout)
```

## 3.6 WebSocket Integration (`lib/use-websocket.ts`)

Custom React hook for real-time price updates:

```typescript
interface UseWebSocketReturn {
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  subscribe: (symbols: string[]) => void;
  unsubscribe: (symbols: string[]) => void;
  sendJson: (data: object) => void;
}
```

**URL Derivation:**
```typescript
const wsBase = NEXT_PUBLIC_API_URL
  .replace('/api/v1', '')
  .replace('http', 'ws');
// → ws://localhost:8000
```

**Endpoint:** `{wsBase}/api/v1/ws/prices`

**Reconnection Strategy:** Exponential backoff with jitter:
- Initial delay: 1s
- Multiplier: 2x
- Cap: 10s maximum
- Sequence: 1s, 2s, 4s, 8s, 10s, 10s, ...

**Lifecycle:**
```
Component Mount → connectWebSocket()
  → status = 'connecting'
  → new WebSocket(url)
  → onopen: status = 'connected', reset retry count
  → onmessage: parse JSON, update signals state
  → onclose: schedule reconnect (setTimeout with backoff)
  → onerror: status = 'error', trigger onclose

Component Unmount → disconnect()
  → close WebSocket
  → clear reconnect timeout
```

**Usage:** Dashboard page subscribes to watchlist symbols on mount, receives price pushes every ~30s, updates SignalCard prices with green/red flash animation.

## 3.7 Component Architecture

### Root Layout (`layout.tsx`)
- Loads Google Fonts (Inter, JetBrains Mono) via next/font
- Mounts `AuthInitializer` component as wrapper
- Global CSS (TailwindCSS v4 directives)
- Root HTML structure with `lang="en"`

### Auth Pages (`auth/login/page.tsx`, `auth/register/page.tsx`)
- Client components with React Hook Form + Zod
- Login: email + password fields with validation
- Register: email + password + confirm password with validation
- Google OAuth button using `loginWithGoogle()`
- Error state displayed below form
- Loading state on submit button
- Links between login/register pages
- Redirect to `/dashboard` if already authenticated

### Dashboard Layout (`dashboard/layout.tsx`)
- Server component wrapper
- Contains `Sidebar` (navigation + risk profile summary)
- Renders child page content via `{children}`
- No data fetching — purely structural

### Dashboard Page (`dashboard/page.tsx`)

The main trading dashboard is a client component with:

**State Management:**
- `signals: Signal[]` — Watchlist signals from API
- `indices: IndexInfo[]` — Index data (NIFTY 50, MIDCAP, SMALLCAP)
- `sparklineBySymbol: Record<string, SparklinePoint[]>` — Cached sparkline data for signal cards
- `marketStatus: 'open' | 'closed'` — NSE market hours (Mon-Fri, 9:15-15:30 IST)

**Data Fetching:**
1. On mount: fetch `/trading/watchlist` → signals + indices
2. Map over top 20 signals: fetch `/trading/market/{symbol}` → extract sparkline data (last 20 close prices)
3. Every 60s: re-check market status via `isMarketOpen()` using `Intl.DateTimeFormat` with `Asia/Kolkata` timezone
4. WebSocket: subscribe to watchlist symbols on mount, unsubscribe on unmount

**Components Rendered:**
- 4 `StatsCard` components (market status, active signals, watchlist count, volatility)
- Multiple `SignalCard` components (one per stock, up to 20)
- Index info cards (3 indices)
- `Modal` (on symbol click) containing `PriceChart`, `SignalGauge`, `TechnicalIndicators`

### Data Model

```typescript
interface Signal {
  symbol: string;
  price: number;
  predicted_price: number;
  target_price: number | null;
  change_pct: number;
  action: string;        // HOLD BUY, HOLD SELL, BUY, SELL, IDLE
  confidence: number;    // 0.0–1.0
  model: string;         // LSTM, PPO, LSTM+PPO, fallback
}

interface SparklinePoint {
  timestamp: string;
  close: number;
}

interface IndexInfo {
  label: string;
  symbol: string;
  price: number;
  change_pct: number;
}
```

## 3.8 Component Hierarchy

```
RootLayout (layout.tsx)
  └── AuthInitializer (restores auth from localStorage, calls fetchUser)
  └── [page content based on route]

DashboardLayout (dashboard/layout.tsx) — Server Component
  ├── Sidebar (client component)
  │   ├── Navigation links (Dashboard, Backtest, Trades, Profile, Logout)
  │   │   └── Lucide icons per link (LayoutDashboard, BarChart4, Receipt, UserCircle, LogOut)
  │   ├── Risk profile card (fetched from /profile/ on mount and auth change)
  │   │   ├── Risk tolerance bar (colored: green/yellow/red)
  │   │   └── Category label (Conservative/Moderate/Growth/Aggressive)
  │   └── Account info (email display)
  └── DashboardPage (client component)
        ├── Header row
        │   ├── StatsCard (Market Status: "Market Open" / "Market Closed")
        │   ├── StatsCard (Active Signals: count of non-IDLE signals)
        │   ├── StatsCard (Watchlist: total stocks tracked)
        │   └── StatsCard (Market Volatility: avg ATR indicator)
        ├── SignalCard (per stock, up to 20)
        │   ├── Symbol + company name
        │   ├── Current price + change_pct (colored)
        │   ├── Action badge (color-coded: green BUY, red SELL, yellow HOLD, gray IDLE)
        │   ├── Confidence progress bar
        │   ├── Sparkline (20-point mini chart, colored green/red)
        │   └── Click handler → opens detail Modal
        ├── Index cards (3 indices: ^NSEI, NIFTYMIDCAP150.NS, NIFTYSMLCAP250.NS)
        └── Modal (on signal card click)
              ├── PriceChart (Recharts ComposedChart: Line for close, Bar for volume)
              ├── SignalGauge (horizontal bar, 0-100% with color gradient)
              └── TechnicalIndicators (grid of indicator cards: RSI, MACD, BB, etc.)

BacktestPage (client component)
  ├── BacktestConfig (React Hook Form + Zod)
  │   ├── Symbol select (text input, must be valid NSE symbol)
  │   ├── Start date / End date (date inputs, validated)
  │   ├── Initial capital (number, must be positive)
  │   └── Risk tolerance (select: 0.25/0.5/0.75/1.0)
  ├── MetricCard × 8 (Total Return, Sharpe, Drawdown, Win Rate, etc.)
  ├── EquityCurve (Recharts LineChart with gradient fill)
  └── Trade table (scrollable, action/price/shares/pnl columns)

TradesPage (client component)
  ├── Filters (symbol search, date range picker)
  ├── StatCard × 3 (Total P&L, Win Rate, Total Trades)
  └── Trade table (sortable by date, symbol, P&L)

ProfilePage (client component)
  ├── InfoCard × 3 (Risk Tolerance, Category, Behavior Profile)
  ├── Risk tolerance bar (colored indicator)
  └── Preferences form
      ├── Use sentiment toggle
      ├── Preferred timeframe select (intraday/swing/position/longterm)
      └── Symbols multi-select (tag input)

RiskAssessmentPage
  └── RiskQuestionnaire
        ├── Feedback Loop Constraints inputs
        ├── 30 categorized questions (5 categories: capital, risk, overtrading, market, breakeven)
        │   └── 5-level selection buttons per question (1-5 scale)
        └── Result display with training status (polls /profile/model-training-status)
```

## 3.9 Key Components Detail

### `Sidebar.tsx`
- Fixed left sidebar (w-[250px]) with slate-900 dark background
- Logo/brand at top
- Navigation links using Lucide icons with active route highlighting via `usePathname()`
- Risk tolerance bar (green/yellow/red based on score)
- Fetches user profile from `/profile/` on mount and when `isAuthenticated` changes
- No caching — re-fetches on every page navigation / component mount

### `SignalCard.tsx`
- Displays stock symbol, current price, change %, signal action with color coding
- Green background (emerald-500/100) for BUY signals, red (rose-500/100) for SELL, yellow (amber-500/100) for HOLD, gray (slate-500/100) for IDLE
- Shows model confidence as progress bar (0-100%)
- Renders `Sparkline` component for price trend visualization
- Clickable → opens detail modal with full chart
- WebSocket updates: price flash animation on change

### `PriceChart.tsx`
- Recharts `ComposedChart` with responsive container
- Line series for close price (blue stroke)
- Bar series for volume (with opacity, bottom axis)
- Tooltip with date, price, volume
- Time range buttons integrated in modal

### `Sparkline.tsx`
- Recharts `LineChart` (tiny, no axes, no grid)
- Animated stroke with configurable color (green/red based on trend direction)
- 20 data points from recent market data
- `isAnimationActive={false}` for performance

### `StatsCard.tsx`
- Simple card with:
  - Icon (optional)
  - Label text
  - Value (large text)
  - Optional sub-text
- Uses `rounded-xl shadow-sm border` styling

### `RiskQuestionnaire.tsx`
- 30 questions organized into 5 categories:
  1. **Capital Allocation** (capital_per_trade_pct, max_profit_close_pct, tp_sl_ratio)
  2. **Risk Parameters** (max_drawdown_pct, intraday_var_pct, loss_streak_reduce_pct)
  3. **Overtrading Controls** (max_trades_per_day, post_loss_rest_min, avg_holding_time_min)
  4. **Market Context** (entry_slippage_bps, session_consistency_score, news_buffer_min)
  5. **Breakeven Strategy** (partial_tp_frequency, breakeven_trigger_pct, breakeven_migration_time_min)
- 5-level selection buttons per question (1-5 scale)
- Submits via `profileApi.submitBehaviorAssessment()`
- Displays results with training status polling (every 5s)

### `BacktestConfig.tsx`
- React Hook Form with Zod validation
- Fields:
  - `symbol`: string, validated as non-empty
  - `start_date`: string (date), must be before end_date
  - `end_date`: string (date), must be after start_date
  - `initial_capital`: number, must be > 0
  - `risk_tolerance`: number, must be 0.0–1.0
- Submits via `backtestApi.runBacktest()`
- Displays results in `MetricCard` components + equity curve chart

### `AuthInitializer.tsx`
- Mounted in root layout
- Reads persisted Zustand state on mount
- If token exists: calls `fetchUser()` (GET /auth/me)
- If fetch fails or no token: ensures clean state
- Renders `{children}` after initialization

## 3.10 Styling Conventions

- TailwindCSS v4 utility classes throughout
- Custom CSS in `globals.css` (minimal)
- Color scheme:
  - BUY signals: green (emerald-500/100)
  - SELL signals: red (rose-500/100)
  - HOLD signals: yellow (amber-500/100)
  - IDLE: gray (slate-500/100)
- Dark sidebar with slate-900 background
- Main content area: white/slate-50 background
- Cards use `rounded-xl` with `shadow-sm` and `border`
- Responsive layout: sidebar collapses on mobile (hidden)

## 3.11 Utility Modules

### `lib/market-hours.ts`
- `isMarketOpen()` — Checks if NSE is currently trading
- Market hours: 9:15 AM – 3:30 PM IST, Monday–Friday
- Uses `Intl.DateTimeFormat` with `Asia/Kolkata` timezone
- Returns boolean

### `lib/trading-format.ts`
- `ACTIONS` constant object mapping action strings to display text
- `normalizeAction(action)` — Standardizes action string format
- `isBuyishAction(action)` — True for BUY, HOLD BUY
- `isSellishAction(action)` — True for SELL, HOLD SELL
- `formatChangePct(value)` — Formats with +/− sign and 2 decimal places

### `lib/firebase.ts`
```typescript
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};
// Initialize Firebase app only if API key exists
const app = firebaseConfig.apiKey ? initializeApp(firebaseConfig) : null;
export const firebaseAuth = app ? getAuth(app) : null;
export const googleProvider = new GoogleAuthProvider();
```

Exports: `firebaseAuth` (getAuth instance or null), `googleProvider` (new GoogleAuthProvider)

## 3.12 Known Frontend Implementation Notes

- **authApi login/register are dead code** — The frontend uses Firebase SDK directly, never calls `POST /auth/login` or `POST /auth/register` on the backend
- **Registration doesn't create backend user record** — Firebase user created but `POST /auth/register` on backend not called, so no backend user record exists in Firebase mode
- **Sidebar fetches profile on every navigation** — No caching layer between renders; makes GET /profile/ on every mount
- **Dockerfile uses Node 18** while package.json targets features requiring Node 20+
- **No loading skeletons** for trades/profile pages (show plain "Loading..." text via React Suspense)
- **No test files or test runner configured** (consistent with backend)
- **WebSocket reconnect is handled but not persistent across page navigation** — new connection established on each dashboard mount
- **No error boundaries** — unhandled React errors may crash the page

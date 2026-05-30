# 3. Frontend Architecture

## 3.1 Technology Stack

| Component | Technology | Version | Role |
|-----------|-----------|---------|------|
| Framework | Next.js | 16.0.10 | App Router, SSR/SSG |
| UI Library | React | 19.2.1 | Component model |
| Language | TypeScript | 5.x | Strict mode, `@/*` → `src/*` |
| Styling | TailwindCSS | 4.x | Utility-first CSS |
| State Management | Zustand | 5.x | Auth state with persist middleware |
| HTTP Client | Axios | latest | API communication with interceptors |
| Charts | Recharts | 3.6 | Price, equity, indicator charts |
| Forms | React Hook Form | latest | Form state management |
| Validation | Zod | 4.x | Schema validation |
| Auth Provider | Firebase JS SDK | 12.x | Email/password + Google OAuth |
| Icons | Lucide React | latest | UI icon set |
| Build Tool | Turbopack (via Next.js) | — | Development bundling |

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

The middleware runs on every request to matching paths:

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

The middleware checks for the `auth_token` cookie, which is set by the Zustand store on successful Firebase authentication. This is a client-side cookie with 1-day expiry and `SameSite=Lax`.

## 3.4 Authentication Flow

### Firebase Auth Integration

The frontend uses Firebase as its sole authentication provider. The backend verifies Firebase ID tokens via Firebase Admin SDK.

**Auth Store (Zustand)** — `lib/auth-store.ts`:

```typescript
interface AuthState {
  user: User | null;
  token: string | null;           // Firebase ID token
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  login(email, password): Promise<void>;
  register(email, password): Promise<void>;
  loginWithGoogle(): Promise<void>;
  logout(): void;
  fetchUser(): Promise<void>;
  clearError(): void;
}
```

**Persistence:** Store state is persisted to localStorage under key `"auth-storage"`. Only `{ token, isAuthenticated }` is persisted.

### Login Flow
```
1. User submits email/password
2. Firebase SDK: signInWithEmailAndPassword(email, password)
3. Extract idToken from UserCredential
4. Store idToken in Zustand + set auth_token cookie
5. Set Authorization: Bearer <token> on Axios default headers
6. Call GET /auth/me to fetch user profile
7. On success: redirect to /dashboard
8. On failure: display error message, clear auth state
```

### Registration Flow
```
1. User submits email/password + confirmation
2. Firebase SDK: createUserWithEmailAndPassword(email, password)
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
1. Clear Authorization header from Axios
2. Firebase SDK: signOut() (catch errors silently)
3. Clear auth_token cookie
4. Reset Zustand store (user: null, token: null, isAuthenticated: false)
```

### App Initialization (`initializeAuth()`)
Called from `AuthInitializer` component mounted in root layout:
1. Read persisted state from Zustand store
2. If token exists: set cookie + Axios header, call `fetchUser()`
3. If fetch fails: call `logout()` (clears everything)

### Auth Header Injection
Two mechanisms ensure the token is sent with every request:
1. **Axios request interceptor** (`lib/api.ts`): Reads token from localStorage (`auth-storage` → `state.token`) on every request
2. **Store-level header setting**: On login/register, sets `api.defaults.headers.common["Authorization"]`

### 401 Response Handling
The Axios response interceptor in `api.ts`:
1. On 401 response: clears localStorage, clears cookie, redirects to `/auth/login`

## 3.5 API Integration (`lib/api.ts`)

**Shared Axios Instance:**
```typescript
const api = axios.create({
  baseURL: NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1",
  headers: { "Content-Type": "application/json" },
});
```

**Request Interceptor:** Injects Bearer token from localStorage `auth-storage`

**Response Interceptor:** On 401, clears session and redirects to login

### API Module Structure

| Module | Functions | Endpoints Called |
|--------|-----------|-----------------|
| `authApi` | login, register, getMe | `POST /auth/login`, `POST /auth/register`, `GET /auth/me` |
| `tradingApi` | getSignal, getMarketData, getWatchlist | `GET /trading/signals/{symbol}`, `GET /trading/market/{symbol}`, `GET /trading/watchlist` |
| `backtestApi` | runBacktest | `POST /backtest/run` |
| `profileApi` | getProfile, submitRiskAssessment, submitBehaviorAssessment, getModelTrainingStatus, updatePreferences | `GET /profile`, `POST /profile/risk-assessment`, `POST /profile/behavior-assessment`, `GET /profile/model-training-status`, `PUT /profile/preferences` |
| `tradesApi` | getTradeHistory | `GET /profile/trades` |

Note: `authApi.login`, `authApi.register`, and `authApi.getMe` are **dead code** — the auth store uses Firebase directly and calls `api.get("/auth/me")` inline rather than through `authApi`.

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

**Reconnection:** Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 10s

**Usage:** Dashboard page subscribes to symbols on mount, receives price pushes every ~30s.

## 3.7 Component Architecture

### Root Layout (`layout.tsx`)
- Loads Google Fonts (Inter, JetBrains Mono)
- Mounts `AuthInitializer` component
- Global CSS (TailwindCSS directives)

### Dashboard Layout (`dashboard/layout.tsx`)
- Server component wrapper
- Contains `Sidebar` (navigation + risk profile summary)
- Renders child page content

### Dashboard Page (`dashboard/page.tsx`)

The main trading dashboard is a client component that:

**State Management:**
- `signals: Signal[]` — Watchlist signals
- `indices: IndexInfo[]` — Index data
- `sparklineBySymbol: Record<string, SparklinePoint[]>` — Cached sparkline data
- `marketStatus: 'open' | 'closed'` — NSE market hours

**Data Fetching:**
1. On mount: fetch `/trading/watchlist` → signals + indices
2. Map over top 20 signals: fetch `/trading/market/{symbol}` → extract sparkline data (last 20 close prices)
3. Every 60s: re-check market status (NSE hours: Mon-Fri, 9:15 AM – 3:30 PM IST)
4. WebSocket: subscribe to watchlist symbols

**Components Rendered:**
- 4 `StatsCard` components (market status, active signals, watchlist count, volatility)
- Multiple `SignalCard` components (one per stock)
- Index info cards
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
  confidence: number;
  model: string;         // LSTM, PPO, LSTM+PPO, fallback
}

interface SparklinePoint {
  timestamp: string;
  close: number;
}
```

## 3.8 Component Hierarchy

```
RootLayout (layout.tsx)
  └── AuthInitializer (restores auth from localStorage)
  └── [page content]

DashboardLayout (dashboard/layout.tsx)
  ├── Sidebar
  │   ├── Navigation links (Dashboard, Backtest, Trades, Profile, Logout)
  │   ├── Risk profile card (fetched from /profile/)
  │   └── Account info
  └── DashboardPage
        ├── StatsCard (Market Status)
        ├── StatsCard (Active Signals)
        ├── StatsCard (Watchlist)
        ├── StatsCard (Market Volatility)
        ├── SignalCard (per stock, up to 20)
        │     └── Sparkline (20-point mini chart)
        ├── Index cards (3 indices)
        └── Modal (on signal click)
              ├── PriceChart (Recharts ComposedChart: bar + line)
              ├── SignalGauge (confidence visualization)
              └── TechnicalIndicators (indicator grid)

BacktestPage
  ├── BacktestConfig (React Hook Form)
  ├── MetricCard (×8: Return, Sharpe, Drawdown, Win Rate, etc.)
  ├── EquityCurve (Recharts LineChart)
  └── Trade table

TradesPage
  ├── Filters (symbol, date range)
  ├── StatCard (×3: Total P&L, Win Rate, Total Trades)
  └── Trade table

ProfilePage
  ├── InfoCard (×3: Risk Tolerance, Category, Behavior Profile)
  ├── Risk tolerance bar
  └── Preferences form

RiskAssessmentPage
  └── RiskQuestionnaire
        ├── Feedback Loop Constraints inputs
        ├── 30 categorized questions (5 categories: capital, risk, overtrading, market, breakeven)
        └── Result display with training status
```

## 3.9 Key Components Detail

### `Sidebar.tsx`
- Fixed left sidebar with logo, navigation links using Lucide icons
- Fetches user profile from `/profile/` on mount and when `isAuthenticated` changes
- Displays risk tolerance as colored bar
- No caching — re-fetches on every page navigation

### `SignalCard.tsx`
- Displays stock symbol, current price, change %, signal action with color coding
- Green background for BUY signals, red for SELL, yellow for HOLD, gray for IDLE
- Shows model confidence as progress bar
- Renders `Sparkline` component for price trend visualization
- Clickable → opens detail modal

### `PriceChart.tsx`
- Recharts `ComposedChart`
- Line series for close price
- Bar series for volume (with opacity)
- Tooltip with date, price, volume
- Responsive container

### `Sparkline.tsx`
- Recharts `LineChart` (tiny, no axes)
- Animated stroke with configurable color (green/red based on trend)
- 20 data points from recent market data

### `RiskQuestionnaire.tsx`
- 30 questions organized into 5 categories:
  1. **Capital Allocation** (capital_per_trade_pct, max_profit_close_pct, tp_sl_ratio)
  2. **Risk Parameters** (max_drawdown_pct, intraday_var_pct, loss_streak_reduce_pct)
  3. **Overtrading Controls** (max_trades_per_day, post_loss_rest_min, avg_holding_time_min)
  4. **Market Context** (entry_slippage_bps, session_consistency_score, news_buffer_min)
  5. **Breakeven Strategy** (partial_tp_frequency, breakeven_trigger_pct, breakeven_migration_time_min)
- 5-level selection buttons per question
- Submits via `profileApi.submitBehaviorAssessment()`
- Displays results with training status polling

### `BacktestConfig.tsx`
- React Hook Form with Zod validation
- Fields: symbol, start_date, end_date, initial_capital, risk_tolerance
- Submits via `backtestApi.runBacktest()`
- Displays results in `MetricCard` components + equity curve

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

## 3.11 Utility Modules

### `lib/market-hours.ts`
- `isMarketOpen()` — Checks if NSE is currently trading
- Market hours: 9:15 AM – 3:30 PM IST, Monday–Friday
- Uses `Intl.DateTimeFormat` with `Asia/Kolkata` timezone

### `lib/trading-format.ts`
- `ACTIONS` constant object mapping
- `normalizeAction(action)` — Standardizes action string format
- `isBuyishAction(action)` — True for BUY, HOLD BUY
- `isSellishAction(action)` — True for SELL, HOLD SELL
- `formatChangePct(value)` — Formats with +/− sign and 2 decimal places

## 3.12 Firebase Configuration (`lib/firebase.ts`)

```typescript
const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};
```

Exports: `firebaseAuth` (getAuth instance), `googleProvider` (new GoogleAuthProvider)

## 3.13 Known Frontend Implementation Notes

- **authApi login/register are dead code** — The frontend uses Firebase SDK directly, never calls `POST /auth/login` or `POST /auth/register` on the backend
- **Registration doesn't create backend user record** — Firebase user created but `POST /auth/register` on backend not called
- **Sidebar fetches profile on every navigation** — No caching layer between renders
- **Dockerfile uses Node 18** while package.json targets features requiring Node 20+
- **No loading skeletons** for trades/profile pages (show plain "Loading..." text)
- **No test files or test runner configured** (consistent with backend)

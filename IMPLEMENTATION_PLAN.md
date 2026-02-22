# AlgoTrading Platform - Implementation Plan

This document outlines the complete plan to finish the AlgoTrading platform.

---

## Phase 1: Critical Backend Fixes (Week 1)

### 1.1 Register Auth Router
**File:** `backend/app/main.py`
**Task:** Add auth router to FastAPI app
**Changes:**
```python
from app.api.routes import trading, backtest, profile, auth  # Add auth

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])  # Add this line
```

### 1.2 Database Async Support
**File:** `backend/app/models/db/database.py`
**Task:** Add async database support using asyncpg
**Changes:**
- Create `get_async_session()` dependency
- Update connection pooling for async

**New File:** `backend/app/models/db/async_database.py`
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session
```

### 1.3 Environment Configuration
**File:** `backend/.env` (create if missing)
**Task:** Add required environment variables
```env
APP_ENV=development
DEBUG=True
SECRET_KEY=your-secret-key-here-change-in-production
DATABASE_URL=postgresql://postgres:password@localhost:5432/algotrading
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=your-jwt-secret-here
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
DEFAULT_MARKET=NSE
MODEL_PATH=./models
```

---

## Phase 2: Frontend Authentication (Week 1-2)

### 2.1 Auth Store (Zustand)
**New File:** `frontend/src/lib/auth-store.ts`
**Features:**
- JWT token storage (localStorage or httpOnly cookies)
- User state management
- Login/logout actions
- Auth status checking

```typescript
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}
```

### 2.2 Login Page
**New File:** `frontend/src/app/auth/login/page.tsx`
**Features:**
- Email/password form (React Hook Form + Zod)
- Error handling
- Redirect to dashboard on success
- Link to register page

### 2.3 Register Page
**New File:** `frontend/src/app/auth/register/page.tsx`
**Features:**
- Email/password registration form
- Password confirmation
- Error handling
- Redirect to login on success

### 2.4 Auth Middleware
**New File:** `frontend/src/middleware.ts`
**Features:**
- Protect dashboard routes
- Redirect unauthenticated users to login
- Handle token expiration

### 2.5 Update API Client
**File:** `frontend/src/lib/api.ts`
**Changes:**
- Add auth token to request headers
- Handle 401 errors (redirect to login)
- Add auth API methods

---

## Phase 3: Core Feature Pages (Week 2-3)

### 3.1 Risk Assessment Page
**New File:** `frontend/src/app/profile/risk-assessment/page.tsx`
**Features:**
- Display 6-question questionnaire from backend
- Multiple choice answers (1-4 scale)
- Submit to `/api/v1/profile/risk-assessment`
- Show results: risk tolerance score, category, recommendations
- Save to user profile

**Components needed:**
- `frontend/src/components/forms/RiskQuestionnaire.tsx`
- Progress indicator
- Results display card

### 3.2 Backtest Page
**New File:** `frontend/src/app/backtest/page.tsx`
**Features:**
- Form: symbol, date range, initial capital, risk tolerance
- Submit to `/api/v1/backtest/run`
- Display results:
  - Total return, Sharpe ratio, max drawdown
  - Win rate, profit factor
  - Trade list table
  - Equity curve chart

**Components needed:**
- `frontend/src/components/forms/BacktestConfig.tsx`
- `frontend/src/components/charts/EquityCurve.tsx`
- Results summary cards
- Trades table

### 3.3 User Profile Page
**New File:** `frontend/src/app/profile/page.tsx`
**Features:**
- Display current user info
- Show risk profile summary
- Edit preferences:
  - Use sentiment toggle
  - Preferred timeframe (intraday/swing/position/longterm)
  - Watchlist symbols (multi-select)
- Save to `/api/v1/profile/preferences`

### 3.4 Trade History Page
**New File:** `frontend/src/app/trades/page.tsx`
**Features:**
- List user's past trades (when implemented)
- Filter by symbol, date range
- Show P&L summary
- Export to CSV

---

## Phase 4: Visualization & Charts (Week 3-4)

### 4.1 Price Chart Component
**New File:** `frontend/src/components/charts/PriceChart.tsx`
**Features:**
- Candlestick or line chart using Recharts
- Show price history
- Volume bars
- Time range selector (1D, 1W, 1M, 3M, 1Y)

### 4.2 Technical Indicators Overlay
**New File:** `frontend/src/components/charts/TechnicalIndicators.tsx`
**Features:**
- SMA/EMA lines
- Bollinger Bands
- Volume chart
- RSI sub-chart
- MACD sub-chart

### 4.3 Equity Curve Chart
**New File:** `frontend/src/components/charts/EquityCurve.tsx`
**Features:**
- Portfolio value over time
- Drawdown visualization
- Benchmark comparison (optional)

### 4.4 Signal Strength Indicator
**New File:** `frontend/src/components/charts/SignalGauge.tsx`
**Features:**
- Visual gauge for confidence score
- Color-coded (red=sell, yellow=hold, green=buy)

### 4.5 Update Dashboard
**File:** `frontend/src/app/dashboard/page.tsx`
**Add:**
- Price chart modal when clicking signal card
- Real-time price sparklines

---

## Phase 5: Real-time Features (Week 4)

### 5.1 WebSocket Backend
**New File:** `backend/app/api/websocket.py`
**Features:**
- WebSocket endpoint for live price updates
- Subscribe/unsubscribe to symbols
- Push updates every 30 seconds during market hours

**Update:** `backend/app/main.py` to include WebSocket router

### 5.2 WebSocket Frontend Hook
**New File:** `frontend/src/lib/use-websocket.ts`
**Features:**
- Connect to WebSocket
- Auto-reconnect on disconnect
- Subscribe to symbol updates
- Handle connection status

### 5.3 Live Price Updates
**Update:** `frontend/src/app/dashboard/page.tsx`
**Changes:**
- Replace polling with WebSocket
- Show live price ticks
- Visual indicators for price changes (green/red flash)

### 5.4 Market Hours Detection
**New File:** `frontend/src/lib/market-hours.ts`
**Features:**
- Detect if NSE is currently open (9:15 AM - 3:30 PM IST, Mon-Fri)
- Show market status indicator
- Disable real-time updates when market closed

---

## Phase 6: Testing & Polish (Week 5)

### 6.1 Backend Tests
**New Directory:** `backend/tests/`
**Files:**
- `test_auth.py` - Login, register, JWT validation
- `test_trading.py` - Signal generation, market data
- `test_backtest.py` - Backtest execution
- `test_profile.py` - Risk assessment
- `conftest.py` - Fixtures, test database

**Run:** `pytest backend/tests/`

### 6.2 Frontend Integration Tests
**New Directory:** `frontend/src/__tests__/`
**Files:**
- `api.test.ts` - API client tests
- `auth-store.test.ts` - Auth state management
- Component tests for critical UI

### 6.3 Error Handling
**Backend:**
- Global exception handler
- Consistent error response format
- Logging with structured output

**Frontend:**
- Error boundaries
- Toast notifications for errors
- Retry logic for failed requests

### 6.4 Performance Optimization
- API response caching (Redis)
- Frontend data caching (React Query or SWR)
- Image optimization
- Lazy loading for heavy components

### 6.5 Documentation Updates
- Update API docs (Swagger auto-generates)
- Add JSDoc comments to complex functions
- Create user guide

---

## Implementation Order

### Sprint 1 (Days 1-7)
1. Register auth router
2. Create auth store
3. Build login/register pages
4. Add auth middleware
5. Update API client with auth headers

### Sprint 2 (Days 8-14)
1. Build risk assessment page
2. Create questionnaire form component
3. Build backtest page
4. Create backtest config form
5. Add equity curve chart

### Sprint 3 (Days 15-21)
1. Build profile page
2. Create preferences form
3. Build trade history page
4. Create price chart component
5. Add technical indicators overlay

### Sprint 4 (Days 22-28)
1. Implement WebSocket backend
2. Create use-websocket hook
3. Add live price updates to dashboard
4. Market hours detection
5. Polish UI/UX

### Sprint 5 (Days 29-35)
1. Write backend tests
2. Write frontend tests
3. Error handling improvements
4. Performance optimization
5. Final documentation

---

## Dependencies

```
Phase 1 (Backend fixes)
    ↓
Phase 2 (Auth) - depends on 1.1
    ↓
Phase 3 (Features) - depends on 2
    ↓
Phase 4 (Charts) - can parallel with 3
    ↓
Phase 5 (Real-time) - depends on 3, 4
    ↓
Phase 6 (Testing) - depends on all above
```

---

## Estimated Effort

| Phase | Duration | Complexity |
|-------|----------|------------|
| Phase 1 | 2-3 days | Low |
| Phase 2 | 4-5 days | Medium |
| Phase 3 | 5-7 days | Medium |
| Phase 4 | 4-5 days | Medium-High |
| Phase 5 | 3-4 days | High |
| Phase 6 | 3-4 days | Medium |
| **Total** | **21-28 days** | |

---

## Quick Wins (Do These First)

1. **Register auth router** (5 minutes) - Currently auth endpoints exist but aren't accessible
2. **Create login page** (2-3 hours) - Unlocks protected features
3. **Add risk assessment page** (3-4 hours) - Core feature mentioned in marketing
4. **Create backtest UI** (4-5 hours) - API already works, just needs frontend

---

## Notes

- Use existing components as templates (SignalCard, StatsCard)
- Follow existing code patterns (async/await, error handling)
- Recharts is already installed for charts
- Zustand is already installed for state management
- React Hook Form + Zod already installed for forms

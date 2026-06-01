# 6. Deployment, Configuration, and Current Limits

## 6.1 Local Development Setup

### Prerequisites
- Python 3.12+
- Node.js 20+
- npm or yarn

### Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env as needed

# Run development server (with auto-reload)
uvicorn app.main:app --reload
# Server at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
# Server at http://localhost:3000
```

### ML Training Pipeline (Optional)

```bash
cd backend
source venv/bin/activate

# Download 5 years of NIFTY 50 data
python training/download_data.py

# Train LSTM price predictor
python training/train_lstm.py

# Train PPO trading agent
python training/train_ppo.py

# Models saved to backend/models/

# View PPO training logs
tensorboard --logdir models/ppo_tensorboard
```

## 6.2 Docker Deployment

### Docker Compose (Full Stack)

```bash
# From repository root
docker-compose up --build
```

**Services:**

| Service | Image | Port | Description | Depends On |
|---------|-------|------|-------------|------------|
| db | postgres:15 | 5432 | PostgreSQL database | — |
| redis | redis:7 | 6379 | Cache and message broker | — |
| backend | ./backend/Dockerfile | 8000 | FastAPI application | db, redis |
| frontend | ./frontend/Dockerfile | 3000 | Next.js application | backend |

**Service Dependencies:**
```
frontend → backend → db, redis
```

**Environment Variables (docker-compose.yml):**
- Backend: `DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/algotrading`, `REDIS_URL=redis://redis:6379/0`
- Frontend: `NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1`

### Dockerfile Details

#### Backend (Python 3.12-slim)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Note:** Backend Docker build may fail due to a malformed multiline `RUN apt-get` instruction (missing line continuations). If building fails, check the Dockerfile line continuations.

#### Frontend (Node.js)
```dockerfile
FROM node:18-slim
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

**Note:** Frontend Dockerfile currently uses `node:18-slim`. For local development, Node 20+ is recommended per package.json requirements (Next.js 16, React 19). Update to `node:20-slim` or `node:22-slim` for production.

## 6.3 Environment Variable Reference

### Backend (`backend/.env`)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `APP_ENV` | "development" | No | Runtime environment: development/production |
| `DEBUG` | true | No | Debug mode (must be false in production) |
| `SECRET_KEY` | "your-secret-key-change-in-production" | Yes* | Application secret key |
| `DEMO_MODE` | true | No | Run without database/Redis (in-memory storage) |
| `FIREBASE_AUTH_ENABLED` | false | No | Enable Firebase authentication |
| `DATABASE_URL` | postgresql+asyncpg://postgres:password@localhost:5432/algotrading | If DEMO_MODE=false | PostgreSQL connection string |
| `REDIS_URL` | redis://localhost:6379/0 | No | Redis connection string |
| `CORS_ORIGINS` | ["http://localhost:3000", "http://127.0.0.1:3000"] | No | Allowed CORS origins (JSON array) |
| `DEFAULT_MARKET` | "NSE" | No | Default market (NSE/BSE) |
| `NEWS_API_KEY` | "" | No | News API key for sentiment analysis |
| `USE_SENTIMENT` | false | No | Enable sentiment analysis |
| `MODEL_PATH` | "./models" | No | Directory for ML model files |
| `DEEPAR_MODEL` | "deepar_v1.pt" | No | DeepAR model filename (unused) |
| `PPO_MODEL` | "ppo_agent_v1.zip" | No | PPO model filename |
| `AUTO_TRAIN_IF_MISSING` | true | No | Auto-train missing models at startup |
| `AUTO_TRAIN_STRICT` | false | No | Fail startup if bootstrap fails |
| `JWT_SECRET` | "jwt-secret-change-in-production" | Yes* | JWT signing secret |
| `JWT_ALGORITHM` | "HS256" | No | JWT signing algorithm |
| `JWT_EXPIRY_HOURS` | 24 | No | JWT token expiry in hours |
| `FIREBASE_PROJECT_ID` | "" | If FIREBASE_AUTH_ENABLED=true | Firebase project ID |
| `FIREBASE_WEB_API_KEY` | "" | If FIREBASE_AUTH_ENABLED=true | Firebase web API key |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | "" | If FIREBASE_AUTH_ENABLED=true | Path to Firebase service account JSON |
| `FIRESTORE_DATABASE_ID` | "(default)" | If FIREBASE_AUTH_ENABLED=true | Firestore database ID |

\* Required in production — must not use default values. The `production_security_issues()` method in `config.py` checks these.

### Frontend (`frontend/.env.local`)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | http://localhost:8000/api/v1 | Yes | Backend API base URL |
| `NEXT_PUBLIC_FIREBASE_API_KEY` | — | If using Firebase | Firebase web API key |
| `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN` | — | If using Firebase | Firebase auth domain |
| `NEXT_PUBLIC_FIREBASE_PROJECT_ID` | — | If using Firebase | Firebase project ID |
| `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET` | — | If using Firebase | Firebase storage bucket |
| `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID` | — | If using Firebase | Firebase messaging sender ID |
| `NEXT_PUBLIC_FIREBASE_APP_ID` | — | If using Firebase | Firebase app ID |

## 6.4 Configuration Guide by Mode

### Demo Mode (No Database Required)

**.env settings:**
```env
DEMO_MODE=true
FIREBASE_AUTH_ENABLED=false
```

**Behavior:**
- No PostgreSQL or Redis required
- All data stored in-memory (resets on restart)
- Auth is permissive: auto-creates users, skips password verification
- Perfect for local testing, demo, and development
- No Firebase configuration needed
- Default mode — works immediately after pip install

**Known issue:** `.env.example` does not include `DEMO_MODE` variable despite being the primary config toggle.

### Firebase Mode (Firebase Auth + Firestore)

**.env settings:**
```env
DEMO_MODE=false
FIREBASE_AUTH_ENABLED=true
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_WEB_API_KEY=your-web-api-key
FIREBASE_SERVICE_ACCOUNT_PATH=/path/to/service-account.json
```

**Behavior:**
- Frontend handles authentication via Firebase SDK
- Backend verifies Firebase ID tokens via Firebase Admin SDK
- User profiles, risk assessments, behavior profiles stored in Firestore
- Per-user PPO retraining enabled
- PostgreSQL not required for user data (Firestore handles it)
- Firebase service account JSON file must be accessible at the specified path
- `secrets/` directory in gitignore can store the service account

**Frontend `.env.local` for Firebase mode:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_FIREBASE_API_KEY=your-web-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=123456789
NEXT_PUBLIC_FIREBASE_APP_ID=1:123456789:web:abc123
```

### Full Production Mode (PostgreSQL)

**.env settings:**
```env
DEMO_MODE=false
FIREBASE_AUTH_ENABLED=false
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/algotrading
JWT_SECRET=your-production-secret
SECRET_KEY=your-production-secret
APP_ENV=production
DEBUG=false
```

**Behavior:**
- All CRUD operations via SQLAlchemy + PostgreSQL
- Standard bcrypt password hashing
- JWT tokens for authentication
- Production security checks at startup:
  - DEBUG must be false
  - DEMO_MODE must be false
  - Default secrets not allowed
- Tables auto-created via SQLAlchemy ORM models on first startup

## 6.5 Pre-commit / Pre-PR Checks

### Backend
```bash
cd backend
source venv/bin/activate
# Lint (if configured)
ruff check .
# Type check (if configured)
mypy app/
```

### Frontend
```bash
cd frontend
npm run lint
npm run build
```

## 6.6 Known Limitations & Implementation Gaps

### Runtime Issues

1. **BacktestService data_dir path sensitivity** — The `data_dir` default is now resolved from file location (`backend_root / "data" / "raw"`), but if launched from a different working directory, path resolution may behave unexpectedly (double `backend/...` prefix).

2. **No tests committed** — Both `backend/tests/` and frontend test infrastructure are empty. The project has no test coverage whatsoever.

3. **WebSocket is simulated real-time** — Price updates poll yfinance every 30 seconds. This is not a live market data feed and introduces latency. Consider upgrading to a WebSocket market data provider (e.g., Alpha Vantage, Polygon.io) for real-time trading.

### Feature Gaps

4. **DeepAR model unused** — `train_deepar.py` trains a model but it is never loaded or used at runtime. The `state_builder.py` has placeholder fields for DeepAR predictions (`pred_price_mean`, `pred_price_std`, `pred_change_pct`, `pred_confidence`) that always default to 0.

5. **Sentiment analysis not implemented** — The `use_sentiment` parameter exists in all trading API routes and `config.py`, but the actual sentiment analysis pipeline (news API integration) is not wired up. `NEWS_API_KEY` env var is available but unused.

6. **State dimension mismatch at architectural level** — `state_builder.py` builds a 30-dimensional vector, but `TradingEnv` uses `state_dim=34`. The gap is filled by zero-padding in `_get_observation()`, but this means the state builder's output is not used directly. The padding dilutes the learned representation.

7. **PPO confidence hardcoded** — `get_ppo_signal()` returns confidence=0.8 regardless of actual policy certainty. The `TradingAgent.get_action_with_confidence()` method properly extracts action probabilities from the policy, but this is not used by `PredictionService`.

8. **PostgreSQL async database not used** — `async_database.py` defines an async engine and `get_async_db()` dependency, but all routes use the sync `get_db()` from `database.py`. The async setup is ready but unused.

### Auth & Frontend

9. **Frontend dead code** — `authApi.login()`, `authApi.register()`, and `authApi.getMe()` in `lib/api.ts` are defined but never called. The auth store uses Firebase SDK directly.

10. **Registration bypasses backend** — Firebase user is created via `createUserWithEmailAndPassword()`, but `POST /auth/register` on the backend is never called, so no backend user record is created during registration in Firebase mode. The user only exists in Firebase Auth, not in Firestore, until `GET /auth/me` auto-provisions them.

11. **Sidebar profile re-fetching** — `Sidebar.tsx` fetches user profile from `/profile/` on every mount/navigation with no caching. This results in redundant API calls.

12. **`.env.example` missing `DEMO_MODE`** — The example env file (`backend/.env.example`) does not include the `DEMO_MODE` variable, even though it's the primary configuration toggle.

13. **Frontend Docker Node version** — Frontend Dockerfile uses `node:18-slim` while package.json is designed for Node 20+ (Next.js 16 requires Node 18.18+, but 18-slim may lack features needed by React 19).

### Model & Training

14. **LSTM inference scaling mismatch** — `train_lstm.py` scales per-symbol with symbol-specific MinMaxScalers fitted on all historical data for that symbol. But `PredictionService._get_lstm_prediction()` creates a fresh MinMaxScaler on the last 30 data points of the query data. These scalers may have different ranges (full history vs last 30 days), causing prediction distribution shift at inference time.

15. **No model versioning or experiment tracking** — Model files are overwritten without version history. No MLflow, W&B, or other tracking integrated. Training runs cannot be compared or rolled back.

16. **Train-inference skew for PPO** — Training uses default behavior array `{capital_per_trade=0.1, tp_sl=0.4, drawdown=0.2, cooldown=0.1}`. Inference uses user-specific values from query parameters. If user behavior differs significantly from defaults, the PPO policy may not generalize to the inference environment.

### Model Performance Context

17. **LSTM training uses combined multi-stock data** — All 52 stocks are concatenated with per-symbol scaling. The model learns cross-stock temporal patterns but may miss stock-specific seasonality.

18. **PPO training on single environment** — The `DummyVecEnv` wraps a single `TradingEnv`. Training on a single environment instance limits diversity of experience compared to vectorized environments.

## 6.7 Production Readiness Checklist

- [ ] Replace default `SECRET_KEY` and `JWT_SECRET` with strong random values
- [ ] Set `DEMO_MODE=false` and `DEBUG=false` in production
- [ ] Configure `CORS_ORIGINS` for production domains (not localhost)
- [ ] Set up PostgreSQL with proper credentials and connection pooling
- [ ] Configure Redis (optional but recommended for caching)
- [ ] Set up Firebase or an alternative auth provider
- [ ] Train models and verify `lstm_final.pt` and `ppo_trading_final.zip` exist
- [ ] Update `.env.example` with all documented variables including `DEMO_MODE`
- [ ] Fix frontend Dockerfile Node version (`node:18-slim` → `node:20-slim`)
- [ ] Add proper error monitoring (Sentry, Datadog, etc.)
- [ ] Add rate limiting to API endpoints (slowapi or custom middleware)
- [ ] Add HTTPS termination (reverse proxy: Nginx, Caddy, Traefik, or cloud LB)
- [ ] Add health check monitoring and automated restart on failure
- [ ] Set up CI/CD pipeline (GitHub Actions, GitLab CI, etc.)
- [ ] Add automated tests (pytest for backend, Vitest/Jest for frontend)
- [ ] Set up structured logging (JSON format, log aggregation)
- [ ] Add database migration system (Alembic for SQLAlchemy)
- [ ] Add backup strategy for PostgreSQL and model files

## 6.8 Troubleshooting

### Backend won't start
```
Error: Unsafe production configuration: ...
```
→ Set `APP_ENV=development` or fix the security issues listed in the error message

### ModuleNotFoundError: No module named 'app'
```
python training/download_data.py
```
→ Run from `backend/` directory (training scripts now auto-add backend root to sys.path)

### Model not found warnings
```
Warning: Model not found at ./models/lstm_final.pt
```
→ Run training pipeline, or set `AUTO_TRAIN_IF_MISSING=true` and restart the backend

### Database connection error
```
sqlalchemy.exc.OperationalError: connection to server at "localhost"
```
→ Set `DEMO_MODE=true` in `.env` to bypass PostgreSQL, or start PostgreSQL service (`sudo systemctl start postgresql`)

### CORS error in browser
```
Access to fetch at 'http://localhost:8000/...' from origin 'http://localhost:3000'
```
→ Verify `CORS_ORIGINS` in backend `.env` includes `http://localhost:3000`. The default in `config.py` already includes it.

### Firestore credential error
```
google.auth.exceptions.DefaultCredentialsError
```
→ Set correct `FIREBASE_SERVICE_ACCOUNT_PATH` pointing to a valid Firebase service account JSON file. If not using Firebase, set `FIREBASE_AUTH_ENABLED=false`.

### WebSocket connection fails
```
WebSocket connection to 'ws://localhost:8000/api/v1/ws/prices' failed
```
→ Verify backend is running on port 8000. Check `NEXT_PUBLIC_API_URL` in frontend `.env.local` matches the backend URL.

### Frontend build errors
```
Error: Cannot find module '...'
```
→ Run `rm -rf node_modules .next && npm install` to get a fresh dependency install.

### Docker build fails (backend)
```
The command '/bin/sh -c apt-get update && apt-get install -y build-essential libpq-dev ...'
```
→ The Dockerfile `RUN` command has malformed line continuations. Check that each line ends with `\` properly. Alternatively, build locally without Docker.

### Docker build fails (frontend)
```
error: Next.js >= 16 requires Node.js >= 18.18
```
→ Update frontend Dockerfile from `FROM node:18-slim` to `FROM node:20-slim` or `FROM node:22-slim`.

## 6.9 Future Scope / Roadmap

### Short Term (Implementation Plan Phases)
1. **Add auth router registration** (if not already) — connects frontend login to backend
2. **Fix LSTM inference scaler** — use per-symbol saved scalers instead of creating fresh ones
3. **Remove hardcoded PPO confidence** — use actual action probabilities from policy
4. **Add packaging/versioning for models** — prevent overwrite, enable rollback

### Medium Term (Product Plan)
1. **Firebase migration completion** — full Firestore backend for user data
2. **Behavior intelligence system** — compliance scoring, feedback loop, adaptive retraining
3. **Multi-user model retraining** — queuing, priority, resource limits
4. **Trade evaluation endpoint** — complete the planned vs executed compliance system
5. **Testing infrastructure** — pytest backend tests, Vitest frontend components
6. **Sentiment analysis** — integrate news API for market sentiment signals

### Long Term
1. **Live market data feed** — replace yfinance polling with WebSocket data provider
2. **Actual trade execution** — broker API integration (Zerodha, Upstox, Angel Broking)
3. **Model marketplace** — community-shared PPO policies for different strategies
4. **Multi-asset support** — commodities, forex, cryptocurrencies
5. **Portfolio optimization** — correlated position sizing across symbols
6. **Mobile app** — React Native or Flutter companion app
7. **Backtesting enhancements** — Walk-forward analysis, Monte Carlo simulation, benchmark comparison

## 6.10 Technical Debt Summary

The following items represent outstanding technical debt that should be addressed systematically (see also `IMPLEMENTATION_PLAN.md`):

| Priority | Issue | Area | Impact |
|----------|-------|------|--------|
| Critical | `.env.example` missing `DEMO_MODE` | Config | New users may not discover demo mode toggle |
| High | Dockerfile `RUN` line continuations broken | Docker | Backend Docker build fails |
| High | Frontend Dockerfile Node version mismatch | Docker | Frontend build may fail with Node 18 |
| High | LSTM inference scaler mismatch | ML | Reduced prediction accuracy at inference |
| Medium | PPO confidence hardcoded | ML | Users see 0.8 confidence regardless of true certainty |
| Medium | State dimension mismatch (30 vs 34) | Architecture | Unused padding in state vector |
| Medium | Dead code in authApi | Frontend | Misleading API module exports |
| Medium | Sidebar profile re-fetching | Frontend | Redundant network calls |
| Low | BacktestService path sensitivity | Backend | Potential file-not-found errors |
| Low | Async database not used | Backend | Unused code path |
| Low | No model versioning | ML | Cannot roll back or compare training runs |
| Low | No test coverage | All | Cannot verify regressions |

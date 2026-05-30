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
```

## 6.2 Docker Deployment

### Docker Compose (Full Stack)

```bash
# From repository root
docker-compose up --build
```

**Services:**

| Service | Image | Port | Description |
|---------|-------|------|-------------|
| db | postgres:15 | 5432 | PostgreSQL database |
| redis | redis:7 | 6379 | Cache and message broker |
| backend | ./backend/Dockerfile | 8000 | FastAPI application |
| frontend | ./frontend/Dockerfile | 3000 | Next.js application |

**Service Dependencies:**
```
frontend → backend → db, redis
```

**Environment Variables (docker-compose.yml):**
- Backend: `DATABASE_URL`, `REDIS_URL`
- Frontend: `NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1`

### Dockerfile Details

#### Backend (Python 3.12-slim)
```
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend (Node.js)
```
FROM node:18-slim
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

**Note:** Frontend Dockerfile currently uses `node:18-slim`. For local development, Node 20+ is recommended per package.json requirements.

## 6.3 Environment Variable Reference

### Backend (`backend/.env`)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `APP_ENV` | "development" | No | Runtime environment: development/production |
| `DEBUG` | true | No | Debug mode |
| `SECRET_KEY` | "your-secret-key-change-in-production" | Yes* | Application secret key |
| `DEMO_MODE` | true | No | Run without database/Redis (in-memory storage) |
| `FIREBASE_AUTH_ENABLED` | false | No | Enable Firebase authentication |
| `DATABASE_URL` | postgresql+asyncpg://postgres:password@localhost:5432/algotrading | If DEMO_MODE=false | PostgreSQL connection string |
| `REDIS_URL` | redis://localhost:6379/0 | No | Redis connection string |
| `DEFAULT_MARKET` | "NSE" | No | Default market (NSE/BSE) |
| `NEWS_API_KEY` | "" | No | News API key for sentiment analysis |
| `USE_SENTIMENT` | false | No | Enable sentiment analysis |
| `MODEL_PATH` | "./models" | No | Directory for ML model files |
| `AUTO_TRAIN_IF_MISSING` | true | No | Auto-train missing models at startup |
| `AUTO_TRAIN_STRICT` | false | No | Fail startup if bootstrap fails |
| `JWT_SECRET` | "jwt-secret-change-in-production" | Yes* | JWT signing secret |
| `JWT_ALGORITHM` | "HS256" | No | JWT signing algorithm |
| `JWT_EXPIRY_HOURS` | 24 | No | JWT token expiry in hours |
| `FIREBASE_PROJECT_ID` | "" | If FIREBASE_AUTH_ENABLED=true | Firebase project ID |
| `FIREBASE_WEB_API_KEY` | "" | If FIREBASE_AUTH_ENABLED=true | Firebase web API key |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | "" | If FIREBASE_AUTH_ENABLED=true | Path to Firebase service account JSON |
| `FIRESTORE_DATABASE_ID` | "(default)" | If FIREBASE_AUTH_ENABLED=true | Firestore database ID |

\* Required in production — must not use default values.

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
- Tables auto-created via SQLAlchemy ORM models

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

1. **BacktestService data_dir path sensitivity** — The `data_dir` default is now resolved from file location (`backend_root / "data" / "raw"`), but if lauched from a different working directory, path resolution may behave unexpectedly.

2. **No tests committed** — Both `backend/tests/` and frontend test infrastructure are empty. The project has no test coverage.

3. **WebSocket is simulated real-time** — Price updates poll yfinance every 30 seconds. This is not a live market data feed and introduces latency.

### Feature Gaps

4. **DeepAR model unused** — `train_deepar.py` trains a model but it is never loaded or used at runtime. The `state_builder.py` has placeholder fields for DeepAR predictions that always default to 0.

5. **Sentiment analysis not implemented** — The `use_sentiment` parameter exists in API routes and config, but the actual sentiment analysis pipeline is not wired up.

6. **State dimension mismatch at architectural level** — `state_builder.py` builds a 30-dimensional vector, but `TradingEnv` uses `state_dim=34`. The gap is filled by zero-padding in `_get_observation()`, but this means the state builder's output is not used directly.

7. **PPO confidence hardcoded** — `get_ppo_signal()` returns confidence=0.8 regardless of actual policy certainty. The `TradingAgent.get_action_with_confidence()` method properly extracts action probabilities, but this is not used by `PredictionService`.

8. **PostgreSQL async database not used** — `async_database.py` defines an async engine, but all routes use the sync `get_db()` from `database.py`.

### Auth & Frontend

9. **Frontend dead code** — `authApi.login()`, `authApi.register()`, and `authApi.getMe()` in `lib/api.ts` are defined but never called. The auth store uses Firebase directly.

10. **Registration bypasses backend** — Firebase user is created, but `POST /auth/register` on the backend is never called, so no backend user record is created during registration in Firebase mode.

11. **Sidebar profile re-fetching** — `Sidebar.tsx` fetches profile on every mount/navigation with no caching.

12. **`.env.example` missing `DEMO_MODE`** — The example env file does not include the `DEMO_MODE` variable, even though it's a primary config toggle.

13. **Frontend Docker Node version** — Frontend Dockerfile uses `node:18-slim` while package.json is designed for Node 20+.

### Model & Training

14. **LSTM inference scaling mismatch** — `train_lstm.py` scales per-symbol with symbol-specific MinMaxScalers, but `PredictionService._get_lstm_prediction()` creates a fresh scaler on the last 30 data points of the query data. These scalers may have different ranges, causing prediction distribution shift.

15. **No model versioning or experiment tracking** — Model files are overwritten without version history. No MLflow, W&B, or other tracking integrated.

16. **Train-inference skew for PPO** — Training uses default behavior array `{capital_per_trade=0.1, tp_sl=0.4, drawdown=0.2, cooldown=0.1}`. Inference uses user-specific values from query parameters. If user behavior differs significantly, the PPO policy may not generalize.

## 6.7 Production Readiness Checklist

- [ ] Replace default `SECRET_KEY` and `JWT_SECRET`
- [ ] Set `DEMO_MODE=false` and `DEBUG=false`
- [ ] Configure `CORS_ORIGINS` for production domains
- [ ] Set up PostgreSQL with proper credentials
- [ ] Configure Redis (optional but recommended)
- [ ] Set up Firebase or alternative auth provider
- [ ] Train models and verify `lstm_final.pt` and `ppo_trading_final.zip` exist
- [ ] Update `.env.example` with all documented variables
- [ ] Fix frontend Dockerfile Node version
- [ ] Add proper error monitoring (Sentry, etc.)
- [ ] Add rate limiting to API endpoints
- [ ] Add HTTPS termination (reverse proxy: Nginx, Caddy, etc.)
- [ ] Add health check monitoring
- [ ] Set up CI/CD pipeline
- [ ] Add automated tests

## 6.8 Troubleshooting

### Backend won't start
```
Error: Unsafe production configuration: ...
```
→ Set `APP_ENV=development` or fix security issues

### ModuleNotFoundError: No module named 'app'
```
python training/download_data.py
```
→ Run from `backend/` directory (scripts now auto-add to sys.path)

### Model not found warnings
```
Warning: Model not found at ./models/lstm_final.pt
```
→ Run training pipeline, or set `AUTO_TRAIN_IF_MISSING=true`

### Database connection error
```
sqlalchemy.exc.OperationalError: connection to server at "localhost"
```
→ Set `DEMO_MODE=true` to bypass PostgreSQL, or start PostgreSQL service

### CORS error in browser
```
Access to fetch at 'http://localhost:8000/...' from origin 'http://localhost:3000'
```
→ Verify `CORS_ORIGINS` includes `http://localhost:3000`

### Firestore credential error
```
google.auth.exceptions.DefaultCredentialsError
```
→ Set correct `FIREBASE_SERVICE_ACCOUNT_PATH` or set `FIREBASE_AUTH_ENABLED=false`

### WebSocket connection fails
```
WebSocket connection to 'ws://localhost:8000/api/v1/ws/prices' failed
```
→ Verify backend is running on port 8000, check `NEXT_PUBLIC_API_URL` value

### Frontend build errors
```
Error: Cannot find module '...'
```
→ Run `rm -rf node_modules .next && npm install`

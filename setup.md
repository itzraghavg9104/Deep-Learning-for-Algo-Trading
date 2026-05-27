# Setup Guide (Firebase + Existing ML Stack)

This guide covers full setup for:
- Backend (FastAPI + ML services)
- Frontend (Next.js)
- Firebase Auth + Firestore integration
- Required environment variables and API usage

---

## 1. Prerequisites

- Python 3.12+
- Node.js 20+
- Firebase project (Auth + Firestore enabled)
- (Optional) Docker + Docker Compose

---

## 2. Firebase Project Setup

## 2.1 Create Project
1. Go to Firebase Console.
2. Create/select project.
3. Note `Project ID`.

## 2.2 Enable Authentication
1. Firebase Console -> Authentication -> Sign-in method.
2. Enable providers you need (Email/Password recommended first).

## 2.3 Enable Firestore
1. Firebase Console -> Firestore Database.
2. Create database in Native mode.
3. Start in test mode for local development (tighten security rules later).

## 2.4 Service Account (Backend)
1. Firebase Console -> Project settings -> Service accounts.
2. Generate new private key JSON.
3. Store file locally (outside git), e.g.:
   - `/absolute/path/firebase-service-account.json`

## 2.5 Web App Config (Frontend)
1. Firebase Console -> Project settings -> Your apps -> Web app.
2. Copy config values:
   - apiKey
   - authDomain
   - projectId
   - appId
   - messagingSenderId

---

## 3. Backend Setup

## 3.1 Install
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3.2 Backend `.env`
Create `backend/.env` (copy from `.env.example`) and set:

```env
# Core
APP_ENV=development
DEBUG=true
DEMO_MODE=true

# Migration switch
FIREBASE_AUTH_ENABLED=false

# Firebase / Firestore
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_WEB_API_KEY=your-firebase-web-api-key
FIREBASE_SERVICE_ACCOUNT_PATH=/absolute/path/to/firebase-service-account.json
FIRESTORE_DATABASE_ID=(default)

# Existing backend settings
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/algotrading
REDIS_URL=redis://localhost:6379/0
MODEL_PATH=./models
```

### Firebase migration mode
- Keep `DEMO_MODE=true` during migration if you want non-Firebase fallbacks.
- Set `FIREBASE_AUTH_ENABLED=true` to enforce Firebase ID token validation in backend.

## 3.3 Run backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

---

## 4. Frontend Setup

## 4.1 Install
```bash
cd frontend
npm install
```

## 4.2 Frontend `.env.local`
Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1

# Firebase client SDK
NEXT_PUBLIC_FIREBASE_API_KEY=your-firebase-api-key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-firebase-project-id
NEXT_PUBLIC_FIREBASE_APP_ID=your-firebase-app-id
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-messaging-sender-id
```

## 4.3 Run frontend
```bash
cd frontend
npm run dev
```

---

## 5. Current Auth/API Behavior (Important)

## 5.1 When `FIREBASE_AUTH_ENABLED=false`
- Existing JWT + demo/local auth flow remains active.
- `/api/v1/auth/login` and `/api/v1/auth/register` work as before.

## 5.2 When `FIREBASE_AUTH_ENABLED=true`
- Backend expects Firebase ID token in `Authorization: Bearer <id_token>`.
- `/api/v1/auth/login` and `/api/v1/auth/register` return `400` intentionally.
- Login/register must happen via Firebase client SDK on frontend.

---

## 6. API Endpoints Added for Behavior Migration

Base: `/api/v1/profile`

1. `POST /risk-assessment`
   - existing scalar risk flow, now also persisted to Firestore when enabled

2. `POST /behavior-assessment`
   - accepts expanded questionnaire answers
   - returns normalized behavior profile payload

3. `POST /trades/evaluate`
   - evaluates planned vs executed trade against behavior constraints
   - returns compliance score + violations

4. `GET /`
   - profile response now includes `behavior_profile` when available

---

## 7. Example Requests

## 7.1 Behavior assessment
```bash
curl -X POST http://localhost:8000/api/v1/profile/behavior-assessment \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "answers": {
      "capital_per_trade_pct": 5,
      "tp_sl_ratio": 2.0,
      "max_profit_close_pct": 12,
      "max_trades_per_day": 6,
      "post_loss_rest_min": 30,
      "max_drawdown_pct": 15,
      "intraday_var_pct": 3,
      "entry_slippage_bps": 10,
      "news_buffer_min": 45,
      "partial_tp_frequency": 3,
      "breakeven_trigger_pct": 1,
      "breakeven_migration_time_min": 60
    }
  }'
```

## 7.2 Trade evaluation
```bash
curl -X POST http://localhost:8000/api/v1/profile/trades/evaluate \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "trade_id": "T-1001",
    "symbol": "RELIANCE.NS",
    "planned": {
      "capital_per_trade_pct": 0.05
    },
    "executed": {
      "capital_per_trade_pct": 0.07,
      "cooldown_respected": false
    },
    "pnl": -1200.5,
    "pnl_pct": -1.2
  }'
```

---

## 8. Firestore Data Layout (Current)

- `users/{uid}`
- `users/{uid}/risk_assessments/latest`
- `users/{uid}/preferences/latest`
- `users/{uid}/behavior_profiles/latest`
- `users/{uid}/trade_events/{docId}` (read path ready)
- `users/{uid}/trade_evaluations/{docId}`

---

## 9. Security Notes

- Never commit Firebase service account JSON.
- Use strict Firestore rules before production.
- Rotate keys if leaked.
- In production, set:
  - `DEBUG=false`
  - `DEMO_MODE=false`
  - `FIREBASE_AUTH_ENABLED=true`

---

## 10. Validation Checklist

1. Backend starts without import/runtime errors.
2. Frontend starts and can hit backend.
3. Firebase token accepted by `/api/v1/auth/me` when enabled.
4. `POST /profile/behavior-assessment` persists profile.
5. `POST /profile/trades/evaluate` stores evaluation entry.
6. `GET /profile/` returns profile + behavior payload.

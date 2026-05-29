# Start Guide (Firebase Mode)

Use this file as the exact command runbook to start and test the project.

## 1) One-Time Setup

### Backend

```bash
cd "/home/raghav-gupta/Major Project/backend"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd "/home/raghav-gupta/Major Project/frontend"
npm install
```

## 2) Required Env Before Start

Backend (`backend/.env`) must have:

```env
APP_ENV=development
DEBUG=true
DEMO_MODE=false
FIREBASE_AUTH_ENABLED=true
FIREBASE_PROJECT_ID=deep-learning-for-algo-trading
FIREBASE_WEB_API_KEY=...
FIREBASE_SERVICE_ACCOUNT_PATH=/home/raghav-gupta/Major Project/secrets/deep-learning-for-algo-trading-firebase-adminsdk-fbsvc-a6986ff702.json
SECRET_KEY=<set-strong-random-value>
JWT_SECRET=<set-strong-random-value>
```

Frontend (`frontend/.env.local`) must have:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_FIREBASE_API_KEY=...
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=deep-learning-for-algo-trading.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=deep-learning-for-algo-trading
NEXT_PUBLIC_FIREBASE_APP_ID=...
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=...
```

## 3) Start Commands (Two Terminals)

### Terminal 1: Backend

```bash
cd "/home/raghav-gupta/Major Project/backend"
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Frontend

```bash
cd "/home/raghav-gupta/Major Project/frontend"
npm run dev
```

## 4) Access URLs

- Frontend: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## 5) Basic Smoke Tests

Run in a third terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/trading/watchlist
curl http://localhost:8000/api/v1/trading/market/RELIANCE.NS
```

## 6) Firebase Auth Test

1. Login from frontend (`/auth/login`) using Firebase-enabled flow.
2. Open browser devtools and verify API calls send `Authorization: Bearer <token>`.
3. Call:

```bash
curl http://localhost:8000/api/v1/auth/me -H "Authorization: Bearer <YOUR_FIREBASE_ID_TOKEN>"
```

Expected: user info response, not `401`.

## 7) Firestore Rule Reminder

Before production testing, set strict Firestore rules (do not keep test mode open).

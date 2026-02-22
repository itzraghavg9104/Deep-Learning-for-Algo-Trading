# 🚀 Project Setup & Final Steps

This document outlines the final steps required to get the **AlgoTrade** platform running on your local machine.

---

## 📋 Prerequisites

Ensure you have the following installed:
- **Python 3.10+**
- **Node.js 18+**
- **PostgreSQL** (Running or via Docker)
- **Redis** (Running or via Docker)

---

## 🛠️ Step 1: Environment Configuration

You need to create `.env` files for both the backend and frontend to store sensitive configuration.

### Backend (`backend/.env`)
Create a file at `backend/.env` and add the following:
```env
# Application
APP_ENV=development
DEBUG=True
SECRET_KEY=generate_a_random_string_here

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/algotrading

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT Authentication
JWT_SECRET=generate_another_random_string_here
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# Market Data
DEFAULT_MARKET=NSE
```

### Frontend (`frontend/.env.local`)
Create a file at `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

---

## 🐍 Step 2: Backend Setup

1. **Create Virtual Environment:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Unix
   .\venv\Scripts\activate   # Windows
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database:**
   Ensure PostgreSQL is running and the `algotrading` database exists. Then run migrations (if using Alembic) or allow SQLAlchemy to create tables on first run.

4. **Prepare Models & Data:**
   ```bash
   # Download historical data for NIFTY 50
   python training/download_data.py
   
   # Train models (if not already in backend/models/)
   python training/train_lstm.py
   python training/train_ppo.py
   ```

---

## ⚛️ Step 3: Frontend Setup

1. **Install Dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Run Development Server:**
   ```bash
   npm run dev
   ```

---

## 🐳 Step 4: Running with Docker (Alternative)

If you prefer using Docker, you can start the entire stack (Database, Redis, Backend, Frontend) with one command:

```bash
docker-compose up --build
```

*Note: You will still need to run the training scripts inside the backend container if models are missing.*

---

## ✅ Step 5: Verification

1. **API Check:** Visit `http://localhost:8000/docs` to see the interactive Swagger UI.
2. **Frontend Check:** Visit `http://localhost:3000` to access the dashboard.
3. **Login:** Register a new user via the `/auth/register` endpoint or the frontend UI (once completed).
4. **Signal Test:** Navigate to the signals page to see the AI recommendations for NSE stocks.

---

## 📈 Troubleshooting

- **Model Not Found:** Ensure `lstm_final.pt` and `ppo_trading_final.zip` are present in `backend/models/`.
- **Database Connection:** Verify that your `DATABASE_URL` in `.env` matches your local PostgreSQL credentials.
- **CORS Errors:** Ensure `CORS_ORIGINS` in `backend/app/config.py` includes `http://localhost:3000`.

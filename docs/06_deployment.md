# Deployment Guide

## Overview

This guide covers deploying the Algo Trading System in development and production environments.

---

## Prerequisites

- **Python**: 3.12+
- **Node.js**: 20+
- **Docker**: 24+ (optional but recommended)
- **PostgreSQL**: 15+ (or use Docker)
- **Redis**: 7+ (or use Docker)

---

## Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd algo-trading-system
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
copy .env.example .env
# Edit .env with your settings

# Start development server
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Environment Variables

### Backend (.env)

```env
# Application
APP_ENV=development
DEBUG=true
SECRET_KEY=your-secret-key-here
DEMO_MODE=True            # No Postgres/Redis needed when True

# Database (not required in DEMO_MODE)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/algotrading

# Redis (not required in DEMO_MODE)
REDIS_URL=redis://localhost:6379/0

# ML Models
MODEL_PATH=./models

# JWT
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

---

## Docker Deployment

### docker-compose.yml

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=algotrading
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/algotrading
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
    depends_on:
      - backend

volumes:
  postgres_data:
```

### Build and Run

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Backend Dockerfile

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Frontend Dockerfile

```dockerfile
FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

---

## Model Training

### Train LSTM Model

```bash
cd backend && source venv/bin/activate
python training/train_lstm.py
```

Trains on NIFTY 50 data from `backend/data/training_data.csv`. Saves to `backend/models/lstm_final.pt`.

### Train PPO Agent

```bash
cd backend && source venv/bin/activate
python training/train_ppo.py
```

Trains on RELIANCE.NS CSV from `backend/data/raw/`. Saves to `backend/models/ppo_trading_final.zip`.

### (Optional) Train DeepAR

```bash
cd backend && source venv/bin/activate
python training/train_deepar.py
```

Requires `pytorch-forecasting` (not in requirements.txt — install separately).

---

## Production Considerations

### Security

- [ ] Use strong SECRET_KEY and JWT_SECRET
- [ ] Enable HTTPS (use nginx reverse proxy)
- [ ] Set CORS origins properly
- [ ] Use rate limiting
- [ ] Sanitize all inputs

### Performance

- [ ] Use Redis for caching
- [ ] Enable database connection pooling
- [ ] Use CDN for static assets
- [ ] Configure proper logging

### Monitoring

- [ ] Set up health check endpoints
- [ ] Configure logging (structured JSON)
- [ ] Use APM tool (Datadog, New Relic)
- [ ] Set up alerts for errors

### Backup

- [ ] Regular database backups
- [ ] Model versioning and backup
- [ ] Configuration backup

---

## Cloud Deployment Options

### Option 1: Vercel + Railway

- **Frontend**: Deploy to Vercel (free tier)
- **Backend**: Deploy to Railway or Render
- **Database**: Railway PostgreSQL
- **Redis**: Upstash Redis

### Option 2: AWS

- **Frontend**: S3 + CloudFront
- **Backend**: ECS Fargate or EC2
- **Database**: RDS PostgreSQL
- **Redis**: ElastiCache

### Option 3: DigitalOcean

- **App Platform**: Full stack deployment
- **Managed Database**: PostgreSQL
- **Managed Redis**: Redis cluster

---

## Troubleshooting

### Backend Issues

```bash
# Check logs
docker-compose logs backend

# Access container shell
docker-compose exec backend bash

# Test database connection
python -c "from app.database import engine; engine.connect(); print('OK')"
```

### Frontend Issues

```bash
# Check build errors
npm run build

# Clear cache
rm -rf .next
npm run dev
```

### Model Loading Issues

```bash
# Verify model files exist
ls -la backend/models/

# Test prediction service
python -c "from app.services.prediction_service import PredictionService; s=PredictionService(); print(s.predict('RELIANCE.NS'))"
```

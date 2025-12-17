# Deployment Guide

## Overview

This guide covers deploying the Algo Trading System in development and production environments.

---

## Prerequisites

- **Python**: 3.11+
- **Node.js**: 18+
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

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Set environment variables
copy .env.example .env.local
# Edit .env.local with your settings

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

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/algotrading

# Redis
REDIS_URL=redis://localhost:6379/0

# Market Data
YFINANCE_ENABLED=true
NEWS_API_KEY=your-newsapi-key

# ML Models
MODEL_PATH=./models
DEEPAR_MODEL=deepar_v1.pt
PPO_MODEL=ppo_agent_v1.zip

# JWT
JWT_SECRET=your-jwt-secret
JWT_EXPIRY_HOURS=24
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

---

## Docker Deployment

### docker-compose.yml

```yaml
version: '3.8'

services:
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
    volumes:
      - ./models:/app/models

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
    depends_on:
      - backend

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=algotrading
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
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
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download ML models (if not mounted)
# RUN python scripts/download_models.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Frontend Dockerfile

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:18-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["node", "server.js"]
```

---

## Model Training

### Train DeepAR Model

```bash
cd backend

# Activate environment
venv\Scripts\activate

# Train on historical data
python training/train_deepar.py \
    --symbols AAPL,GOOGL,MSFT \
    --start-date 2018-01-01 \
    --end-date 2023-12-31 \
    --epochs 100 \
    --batch-size 32 \
    --output models/deepar_v1.pt
```

### Train PPO Agent

```bash
python training/train_ppo.py \
    --symbol AAPL \
    --start-date 2018-01-01 \
    --end-date 2023-12-31 \
    --total-timesteps 1000000 \
    --eval-freq 10000 \
    --output models/ppo_agent_v1.zip
```

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
ls -la models/

# Test model loading
python -c "from app.models.ml import load_models; load_models()"
```

# 🤖 Algo Trading System

**Architecting an Optimization-Based Algorithmic Trading System**

A Framework Integrating Probabilistic Forecasting and Deep Reinforcement Learning with Trader Behavior Modeling

**🇮🇳 Target Market: Indian Stock Market (NSE/BSE)**

---

## 📋 Overview

This project implements a sophisticated algorithmic trading platform for the **Indian stock market** that mimics the cognitive process of professional traders by combining:

1. **Probabilistic Forecasting** — DeepAR-Attention model for price prediction with uncertainty quantification
2. **Deep Reinforcement Learning** — PPO agent for optimized Buy/Sell/Hold decisions
3. **Trader Behavior Integration** — Risk tolerance, trading timeframe, and break-even analysis
4. **Optional Sentiment Analysis** — FinBERT for news sentiment (user-configurable)

### Key Features

- 🎯 **Two-Stage Architecture**: Prediction → Optimization (mimics human trader cognition)
- 📊 **30+ Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- 📰 **Optional Sentiment Analysis**: FinBERT for news sentiment (can be toggled)
- ⚖️ **Risk-Adjusted Returns**: Optimizes for Sharpe Ratio
- 🧠 **Trader Behavior Modeling**: Adapts to personal risk tolerance
- 🇮🇳 **Indian Market Focus**: NSE/BSE stocks via NSEpy/yfinance

---

## 📁 Project Structure

```
algo-trading-system/
├── README.md                   # This file
├── docs/                       # 📚 Documentation
│   ├── 01_architecture.md      # System architecture details
│   ├── 02_data_processing.md   # Layer 1: Data processing pipeline
│   ├── 03_decision_engine.md   # Layer 2: DRL decision engine
│   ├── 04_trader_behavior.md   # Trader behavior modeling
│   ├── 05_api_reference.md     # API documentation
│   └── 06_deployment.md        # Deployment guide
│
├── references/                 # 📄 Research papers & presentations
│   ├── 1-s2.0-S095741742303083X-main.pdf   # Huang et al. - BiLSTM-Attention DRL
│   ├── 1-s2.0-S2590005625000177-main.pdf   # Bhuiyan et al. - DL systematic review
│   ├── s00521-024-09916-3.pdf              # Li et al. - DeepAR-Attention
│   └── Major Project Presentation 1.pptx    # Project proposal
│
├── backend/                    # 🐍 Python FastAPI Backend
│   ├── app/
│   │   ├── main.py
│   │   ├── layer1_data_processing/
│   │   ├── layer2_decision/
│   │   ├── trader_behavior/
│   │   └── api/
│   ├── training/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   # ⚛️ Next.js Frontend
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── package.json
│
└── docker-compose.yml          # 🐳 Container orchestration
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: DATA PROCESSING                         │
├─────────────────┬─────────────────┬─────────────────┬───────────────────┤
│  DeepAR Model   │   Technical     │    FinBERT      │  Trader Behavior  │
│  (Probabilistic │   Indicators    │   (Sentiment)   │  (Risk/Timeframe) │
│   Forecasting)  │   (30+ Signals) │                 │                   │
└────────┬────────┴────────┬────────┴────────┬────────┴─────────┬─────────┘
         │                 │                 │                  │
         └─────────────────┴────────┬────────┴──────────────────┘
                                    │
                              STATE VECTOR
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: DECISION ENGINE                          │
│                                                                          │
│                     PPO Agent (Proximal Policy Optimization)             │
│                     Reward: Sharpe Ratio (Risk-Adjusted Returns)         │
│                                                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  BUY │ SELL│HOLD│
                        └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional)

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker (Full Stack)

```bash
docker-compose up -d
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/01_architecture.md) | System design and components |
| [Data Processing](docs/02_data_processing.md) | DeepAR, indicators, sentiment |
| [Decision Engine](docs/03_decision_engine.md) | PPO agent and training |
| [Trader Behavior](docs/04_trader_behavior.md) | Risk profiling and break-even |
| [API Reference](docs/05_api_reference.md) | Backend API endpoints |
| [Deployment](docs/06_deployment.md) | Deployment instructions |

---

## 📄 Research References

1. **Bhuiyan et al. (2025)** — "Deep learning for algorithmic trading: A systematic review of predictive models and optimization strategies." *Array, 26.*

2. **Huang et al. (2024)** — "A novel deep reinforcement learning framework with BiLSTM-Attention networks for algorithmic trading." *Expert Systems With Applications, 240.*

3. **Li et al. (2024)** — "DeepAR-Attention probabilistic prediction for stock price series." *Neural Computing and Applications, 36.*

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Next.js 14, TypeScript, TailwindCSS |
| Backend | Python 3.11, FastAPI |
| ML/DL | PyTorch, Stable-Baselines3, Transformers |
| Data | PostgreSQL, Redis, yfinance |
| Deployment | Docker, Docker Compose |

---

## 📈 Performance Targets

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 20% |
| Win Rate | > 50% |

---

## 📝 License

This project is for educational purposes as part of a college major project.

---

## 👥 Contributors

- Raghav Gupta

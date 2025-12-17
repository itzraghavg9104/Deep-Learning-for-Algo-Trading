# System Architecture

> AI-Powered Algorithmic Trading Platform for Indian Markets (NSE/BSE)

## Architecture Overview

![System Architecture](images/system_architecture.png)

Our platform implements a **two-stage architecture** that mimics the cognitive process of professional traders:

1. **Layer 1 (Data Processing)**: Analyzes market data and generates predictions
2. **Layer 2 (Decision Engine)**: Optimizes trading decisions using reinforcement learning
3. **Trader Behavior Module**: Personalizes strategies based on individual risk tolerance

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[NSE/BSE Market Data] --> B[yfinance API]
        C[Historical NIFTY 50] --> B
    end
    
    subgraph "Layer 1: Data Processing"
        B --> D[Technical Indicators<br/>30+ Indicators]
        B --> E[LSTM Predictor<br/>Price Forecasting]
        D --> F[State Vector Builder]
        E --> F
    end
    
    subgraph "Trader Behavior"
        G[Risk Profiler] --> H[Position Sizer]
        H --> I[Break-Even Tracker]
    end
    
    subgraph "Layer 2: Decision Engine"
        F --> J[PPO Agent]
        I --> J
        J --> K{Trading Signal}
    end
    
    K -->|BUY| L[Execute Trade]
    K -->|SELL| L
    K -->|HOLD| M[Wait]
    
    subgraph "API Layer"
        L --> N[FastAPI Backend]
        M --> N
        N --> O[REST Endpoints]
    end
    
    subgraph "Frontend"
        O --> P[Next.js Dashboard]
        P --> Q[Real-time Signals]
        P --> R[Charts & Analytics]
    end

    style A fill:#3b82f6
    style B fill:#3b82f6
    style D fill:#8b5cf6
    style E fill:#8b5cf6
    style J fill:#a855f7
    style N fill:#f97316
    style P fill:#06b6d4
```

---

## Component Details

### Layer 1: Data Processing

| Component | Technology | Purpose |
|-----------|------------|---------|
| Market Data | yfinance | Fetch OHLCV data from NSE/BSE |
| Technical Indicators | pandas-ta | Compute 30+ indicators (RSI, MACD, BB, etc.) |
| LSTM Predictor | PyTorch | Probabilistic price forecasting |
| State Builder | NumPy | Normalize and combine all features |

**Files:**
- [`market_data.py`](file:///d:/Major%20Project/backend/app/layer1_data_processing/market_data.py) - NSE/BSE data fetching
- [`technical_indicators.py`](file:///d:/Major%20Project/backend/app/layer1_data_processing/technical_indicators.py) - 30+ indicators
- [`state_builder.py`](file:///d:/Major%20Project/backend/app/layer1_data_processing/state_builder.py) - Feature normalization

---

### Layer 2: Decision Engine

| Component | Technology | Purpose |
|-----------|------------|---------|
| Trading Environment | Gymnasium | Custom trading simulation |
| PPO Agent | Stable-Baselines3 | Policy optimization |
| Reward Function | Custom | Sharpe Ratio optimization |

**Training Results:**
- **Average Return**: 132.28%
- **Sharpe Ratio**: 0.66
- **Timesteps**: 30,000

**Files:**
- [`trading_env.py`](file:///d:/Major%20Project/backend/app/layer2_decision/trading_env.py) - Gymnasium environment
- [`ppo_agent.py`](file:///d:/Major%20Project/backend/app/layer2_decision/ppo_agent.py) - PPO wrapper
- [`reward_function.py`](file:///d:/Major%20Project/backend/app/layer2_decision/reward_function.py) - Sharpe optimization

---

### Trader Behavior Module

```mermaid
graph LR
    A[Risk Questionnaire] --> B[Risk Score<br/>0.0 - 1.0]
    B --> C{Risk Category}
    C -->|< 0.3| D[Conservative]
    C -->|0.3-0.5| E[Moderate]
    C -->|0.5-0.7| F[Growth]
    C -->|> 0.7| G[Aggressive]
    
    D --> H[Position Sizer]
    E --> H
    F --> H
    G --> H
    
    H --> I[Kelly Criterion]
    H --> J[Fixed %]
    H --> K[Volatility-Adjusted]
    
    I --> L[Trade Size]
    J --> L
    K --> L
```

**Components:**
- **Risk Profiler**: Questionnaire-based risk assessment
- **Position Sizer**: Kelly Criterion, volatility-adjusted sizing
- **Break-Even Tracker**: P&L and position management

---

## API Architecture

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant LSTM
    participant PPO
    participant Market

    User->>Frontend: Request Signal
    Frontend->>API: GET /trading/signals/RELIANCE.NS
    API->>Market: Fetch OHLCV Data
    Market-->>API: Price Data
    API->>LSTM: Predict Price
    LSTM-->>API: Predicted Price + Confidence
    API->>PPO: Get Action
    PPO-->>API: BUY/SELL/HOLD
    API-->>Frontend: Signal Response
    Frontend-->>User: Display Signal Card
```

---

## Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Backend | FastAPI | 0.100+ |
| ML Training | PyTorch | 2.0+ |
| RL Agent | Stable-Baselines3 | 2.0+ |
| Frontend | Next.js | 14+ |
| Styling | TailwindCSS | 3.0+ |
| State Management | Zustand | 4.0+ |

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Production"
        A[Nginx Reverse Proxy] --> B[FastAPI Backend]
        A --> C[Next.js Frontend]
        B --> D[(PostgreSQL)]
        B --> E[(Redis Cache)]
        B --> F[Model Files<br/>LSTM + PPO]
    end
    
    subgraph "Development"
        G[uvicorn --reload] --> H[Hot Reload]
        I[npm run dev] --> J[Turbopack]
    end
```

---

## Directory Structure

```
Deep-Learning-for-Algo-Trading/
├── backend/
│   ├── app/
│   │   ├── api/routes/         # FastAPI endpoints
│   │   ├── layer1_data_processing/
│   │   │   ├── market_data.py
│   │   │   ├── technical_indicators.py
│   │   │   └── state_builder.py
│   │   ├── layer2_decision/
│   │   │   ├── trading_env.py
│   │   │   ├── ppo_agent.py
│   │   │   └── reward_function.py
│   │   ├── trader_behavior/
│   │   │   ├── risk_profiler.py
│   │   │   ├── position_sizer.py
│   │   │   └── breakeven_tracker.py
│   │   └── services/
│   │       └── prediction_service.py
│   ├── training/
│   │   ├── train_lstm.py
│   │   └── train_ppo.py
│   ├── models/                 # Trained models
│   │   ├── lstm_final.pt
│   │   └── ppo_trading_final.zip
│   └── data/                   # Training data
├── frontend/
│   └── src/
│       ├── app/                # Next.js pages
│       ├── components/         # React components
│       └── lib/                # Utilities
├── docs/                       # Documentation
└── references/                 # Research papers
```

---

## Next Steps

1. **Database Integration**: PostgreSQL for user profiles and trade history
2. **Authentication**: JWT-based user authentication
3. **Docker Deployment**: Containerized production deployment
4. **Real-time WebSocket**: Live price updates

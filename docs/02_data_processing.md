# Layer 1: Data Processing Pipeline

## Overview

Layer 1 is the "Analysis" component that processes raw market data into a comprehensive state representation, mimicking how a professional trader gathers and synthesizes information.

---

## Components

### 1. DeepAR-Attention Model

**Purpose**: Probabilistic price forecasting with uncertainty quantification

**Why DeepAR?**
- Outputs a **distribution** (mean μ, std σ) instead of a single point
- Allows risk-adjusted decision making
- Attention mechanism highlights important time steps

**Architecture**:
```
Input: [price_t-n, ..., price_t] (lookback window)
    ↓
LSTM Encoder (hidden_size=128)
    ↓
Attention Layer
    ↓
Output: μ(t+1), σ(t+1), attention_weights
```

**Key Outputs**:
| Output | Description |
|--------|-------------|
| `price_mean` | Expected price at t+1 |
| `price_std` | Uncertainty in prediction |
| `price_change_pct` | Relative change (for robustness) |
| `attention_weights` | Which past timesteps matter most |

---

### 2. Technical Indicators (30+)

**Purpose**: Capture market patterns and momentum

**Indicator Categories**:

#### Trend Indicators
| Indicator | Window | Description |
|-----------|--------|-------------|
| SMA | 20, 50, 200 | Simple Moving Average |
| EMA | 12, 26 | Exponential Moving Average |
| MACD | 12, 26, 9 | Moving Average Convergence Divergence |
| ADX | 14 | Average Directional Index |
| Parabolic SAR | - | Stop and Reverse |

#### Momentum Indicators
| Indicator | Window | Range |
|-----------|--------|-------|
| RSI | 14 | 0-100 |
| Stochastic %K | 14 | 0-100 |
| Stochastic %D | 3 | 0-100 |
| CCI | 20 | Unbounded |
| Williams %R | 14 | -100 to 0 |
| ROC | 10 | Percentage |

#### Volatility Indicators
| Indicator | Window | Description |
|-----------|--------|-------------|
| Bollinger Upper | 20, 2σ | Upper band |
| Bollinger Lower | 20, 2σ | Lower band |
| Bollinger %B | 20 | Position within bands |
| ATR | 14 | Average True Range |
| Keltner Upper | 20 | Upper channel |
| Keltner Lower | 20 | Lower channel |

#### Volume Indicators
| Indicator | Description |
|-----------|-------------|
| OBV | On-Balance Volume |
| VWAP | Volume Weighted Average Price |
| MFI | Money Flow Index |
| CMF | Chaikin Money Flow |
| Volume SMA | Volume moving average |

**Normalization**: All indicators are normalized to [-1, 1] or [0, 1] for neural network input.

---

### 3. FinBERT Sentiment Analysis

**Purpose**: Assess market sentiment from financial news

**Model**: `ProsusAI/finbert` (pre-trained on financial text)

**Pipeline**:
```
News Headlines → Tokenizer → FinBERT → Softmax → Sentiment Score
```

**Output**:
| Output | Range | Description |
|--------|-------|-------------|
| `sentiment_score` | [-1, 1] | Negative to Positive |
| `sentiment_confidence` | [0, 1] | Model confidence |

**Aggregation**: Multiple headlines are aggregated using confidence-weighted average.

---

### 4. State Builder

**Purpose**: Combine all features into a unified state vector

**State Vector Structure**:
```python
state = {
    # DeepAR Outputs (4 features)
    "price_mean": float,
    "price_std": float,
    "price_change_pct": float,
    "prediction_confidence": float,
    
    # Technical Indicators (30+ features)
    "rsi_14": float,
    "macd_line": float,
    "macd_signal": float,
    "macd_histogram": float,
    "bollinger_pct_b": float,
    "atr_normalized": float,
    # ... 25+ more
    
    # Sentiment (2 features)
    "news_sentiment": float,
    "sentiment_confidence": float,
    
    # Trader Behavior (5 features)
    "risk_tolerance": float,
    "preferred_timeframe": int,
    "current_position": float,
    "breakeven_price": float,
    "unrealized_pnl_pct": float,
    
    # Portfolio State (3 features)
    "cash_ratio": float,
    "position_value": float,
    "total_portfolio_value": float,
}
```

**Total Dimensions**: ~50 features

---

## Data Sources

| Source | Data Type | Update Frequency | Notes |
|--------|-----------|------------------|-------|
| **NSEpy** | NSE OHLCV | Daily | Primary for NSE stocks |
| **yfinance** | NSE/BSE OHLCV | Daily / 15-min | Use `.NS` suffix (e.g., `RELIANCE.NS`) |
| **NewsAPI** | Headlines | On-demand | **Optional** - User configurable |
| User Profile | Behavior | On-change | Risk preferences |

### Indian Market Stock Symbols

```python
# NSE stocks use .NS suffix
nse_symbols = [
    "RELIANCE.NS",   # Reliance Industries
    "TCS.NS",        # TCS
    "INFY.NS",       # Infosys
    "HDFCBANK.NS",   # HDFC Bank
    "ICICIBANK.NS",  # ICICI Bank
    "SBIN.NS",       # State Bank of India
    "BHARTIARTL.NS", # Bharti Airtel
    "ITC.NS",        # ITC
    "KOTAKBANK.NS",  # Kotak Mahindra Bank
    "LT.NS",         # Larsen & Toubro
]

# BSE stocks use .BO suffix
bse_symbols = ["RELIANCE.BO", "TCS.BO"]
```

### Optional Sentiment Configuration

```python
# User can enable/disable sentiment in their profile
user_config = {
    "use_sentiment": True,  # Toggle sentiment analysis
    "news_api_key": "...",  # Required only if use_sentiment=True
}

# State builder respects this setting
if user_config["use_sentiment"]:
    state["news_sentiment"] = get_finbert_score(headlines)
else:
    state["news_sentiment"] = 0.0  # Neutral default
```

---

## Implementation Files

| File | Purpose |
|------|---------|
| `deepar_attention.py` | DeepAR model definition and inference |
| `bilstm_encoder.py` | BiLSTM feature encoder |
| `technical_indicators.py` | pandas-ta indicator computation |
| `finbert_sentiment.py` | FinBERT wrapper and aggregation |
| `state_builder.py` | Combines all into state vector |

---

## Next Steps

- See [Decision Engine](03_decision_engine.md) for how the state is used
- See [Trader Behavior](04_trader_behavior.md) for behavior modeling details

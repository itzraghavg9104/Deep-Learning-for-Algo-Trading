# API Reference

> Complete REST API documentation for the Algo Trading Platform

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://api.yourdomain.com/api/v1
```

---

## Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trading/signals/{symbol}` | GET | Get AI-powered trading signal |
| `/trading/market/{symbol}` | GET | Get market data with indicators |
| `/trading/watchlist` | GET | Get signals for top NSE stocks |
| `/backtest/run` | POST | Run backtest simulation |
| `/backtest/{id}` | GET | Get backtest results |
| `/profile/risk-assessment` | POST | Submit risk questionnaire |
| `/profile/` | GET | Get user profile |
| `/profile/preferences` | PUT | Update preferences |

---

## Trading Endpoints

### Get Trading Signal

Returns AI-powered trading signal with LSTM prediction and PPO recommendation.

```http
GET /api/v1/trading/signals/{symbol}
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | Yes | Stock symbol (e.g., `RELIANCE.NS`) |
| `use_sentiment` | boolean | No | Include sentiment analysis (default: false) |
| `use_model` | boolean | No | Use trained LSTM model (default: true) |

**Response:**

```json
{
    "symbol": "RELIANCE.NS",
    "timestamp": "2024-12-18T00:30:00",
    "action": "BUY",
    "confidence": 0.72,
    "prediction": {
        "current_price": 1544.40,
        "predicted_price": 1560.25,
        "price_change": 15.85,
        "change_pct": 1.03,
        "model": "LSTM"
    },
    "indicators": {
        "rsi_14": 58.5,
        "macd": 12.3,
        "bb_position": 0.65
    }
}
```

**Action Values:**
- `BUY`: Predicted price increase > 1%
- `SELL`: Predicted price decrease > 1%
- `HOLD`: Predicted change within ±1%

---

### Get Market Data

Returns current market data with technical indicators.

```http
GET /api/v1/trading/market/{symbol}
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | Yes | Stock symbol |
| `period` | string | No | Data period: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y` |

**Response:**

```json
{
    "symbol": "RELIANCE.NS",
    "current_price": 1544.40,
    "change_pct": 0.14,
    "volume": 12500000,
    "indicators": {
        "sma_20": 1520.50,
        "sma_50": 1490.25,
        "ema_12": 1535.80,
        "ema_26": 1522.40,
        "rsi_14": 58.5,
        "macd": 12.3,
        "macd_signal": 10.5,
        "bb_upper": 1580.00,
        "bb_lower": 1480.00,
        "atr": 25.50
    }
}
```

---

### Get Watchlist Signals

Returns signals for top 5 NSE stocks using trained models.

```http
GET /api/v1/trading/watchlist
```

**Response:**

```json
{
    "signals": [
        {
            "symbol": "RELIANCE.NS",
            "price": 1544.40,
            "predicted_price": 1560.25,
            "change_pct": 1.03,
            "action": "BUY",
            "confidence": 0.72,
            "model": "LSTM"
        },
        {
            "symbol": "TCS.NS",
            "price": 3217.80,
            "predicted_price": 3210.00,
            "change_pct": -0.24,
            "action": "HOLD",
            "confidence": 0.55,
            "model": "LSTM"
        }
    ],
    "model_available": true
}
```

---

## Backtest Endpoints

### Run Backtest

Execute a backtest simulation on historical data.

```http
POST /api/v1/backtest/run
```

**Request Body:**

```json
{
    "symbol": "RELIANCE.NS",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 100000,
    "risk_tolerance": 0.5
}
```

**Response:**

```json
{
    "backtest_id": "bt_abc123",
    "status": "running",
    "message": "Backtest started"
}
```

---

### Get Backtest Results

Retrieve results of a completed backtest.

```http
GET /api/v1/backtest/{backtest_id}
```

**Response:**

```json
{
    "backtest_id": "bt_abc123",
    "status": "completed",
    "results": {
        "total_return": 0.1528,
        "sharpe_ratio": 1.45,
        "max_drawdown": -0.085,
        "win_rate": 0.62,
        "total_trades": 45,
        "winning_trades": 28,
        "losing_trades": 17
    },
    "equity_curve": [100000, 100500, 101200, ...],
    "trades": [
        {
            "date": "2023-01-15",
            "action": "BUY",
            "price": 1450.00,
            "quantity": 10
        }
    ]
}
```

---

## Profile Endpoints

### Submit Risk Assessment

Submit questionnaire answers for risk profiling.

```http
POST /api/v1/profile/risk-assessment
```

**Request Body:**

```json
{
    "answers": [3, 4, 2, 5, 3, 4, 2, 3, 4, 3]
}
```

**Response:**

```json
{
    "risk_tolerance": 0.65,
    "category": "Growth",
    "description": "You have a growth-oriented investment profile with moderate-to-high risk tolerance.",
    "recommendations": {
        "max_position_size": 0.15,
        "suggested_stop_loss": 0.05,
        "suggested_take_profit": 0.10,
        "rebalancing_frequency": "monthly"
    }
}
```

---

### Get User Profile

Retrieve current user profile and preferences.

```http
GET /api/v1/profile/
```

**Response:**

```json
{
    "user_id": "user_123",
    "risk_tolerance": 0.65,
    "category": "Growth",
    "preferences": {
        "use_sentiment": false,
        "preferred_timeframe": "1d",
        "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    },
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-12-18T00:00:00Z"
}
```

---

### Update Preferences

Update user trading preferences.

```http
PUT /api/v1/profile/preferences
```

**Request Body:**

```json
{
    "use_sentiment": false,
    "preferred_timeframe": "1d",
    "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
}
```

**Response:**

```json
{
    "message": "Preferences updated successfully",
    "preferences": {
        "use_sentiment": false,
        "preferred_timeframe": "1d",
        "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    }
}
```

---

## System Endpoints

### Health Check

```http
GET /health
```

**Response:**

```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2024-12-18T00:30:00Z"
}
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
    "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Symbol or resource not found |
| 500 | Internal Server Error |

---

## Rate Limiting

| Tier | Requests/minute |
|------|-----------------|
| Free | 60 |
| Pro | 600 |
| Enterprise | Unlimited |

---

## Interactive Documentation

Swagger UI is available at:
```
http://localhost:8000/docs
```

ReDoc is available at:
```
http://localhost:8000/redoc
```

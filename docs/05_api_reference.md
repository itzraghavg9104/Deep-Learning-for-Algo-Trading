# API Reference

## Overview

The backend exposes a RESTful API built with FastAPI. All endpoints return JSON responses.

**Base URL**: `http://localhost:8000/api/v1`

---

## Authentication

### POST `/auth/register`
Register a new user.

**Request Body**:
```json
{
  "email": "trader@example.com",
  "password": "securepassword",
  "name": "John Trader"
}
```

**Response** `201 Created`:
```json
{
  "id": "uuid",
  "email": "trader@example.com",
  "name": "John Trader",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### POST `/auth/login`
Authenticate and receive JWT token.

**Request Body**:
```json
{
  "email": "trader@example.com",
  "password": "securepassword"
}
```

**Response** `200 OK`:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## Trading Signals

### GET `/signals/{symbol}`
Get current trading signal for a symbol.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | Yes | Stock symbol (e.g., AAPL) |

**Headers**:
```
Authorization: Bearer <token>
```

**Response** `200 OK`:
```json
{
  "symbol": "AAPL",
  "timestamp": "2024-01-15T10:30:00Z",
  "signal": {
    "action": "BUY",
    "confidence": 0.78,
    "position_size": 50,
    "position_value": 9500.00
  },
  "prediction": {
    "price_mean": 195.50,
    "price_std": 3.25,
    "change_pct": 2.1
  },
  "indicators": {
    "rsi": 45.2,
    "macd_signal": "bullish",
    "trend": "uptrend"
  },
  "sentiment": {
    "score": 0.35,
    "source_count": 12
  }
}
```

### GET `/signals/portfolio`
Get signals for all watched symbols.

**Response** `200 OK`:
```json
{
  "signals": [
    { "symbol": "AAPL", "action": "BUY", "confidence": 0.78 },
    { "symbol": "GOOGL", "action": "HOLD", "confidence": 0.65 },
    { "symbol": "MSFT", "action": "SELL", "confidence": 0.72 }
  ],
  "generated_at": "2024-01-15T10:30:00Z"
}
```

---

## Market Data

### GET `/market/{symbol}`
Get current market data and indicators.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | Yes | Stock symbol |
| `period` | string | No | Data period (1d, 5d, 1mo, 3mo) |

**Response** `200 OK`:
```json
{
  "symbol": "AAPL",
  "current_price": 191.25,
  "change_pct": 1.5,
  "volume": 45000000,
  "indicators": {
    "rsi_14": 55.3,
    "macd_line": 2.15,
    "macd_signal": 1.89,
    "bollinger_upper": 198.50,
    "bollinger_lower": 185.20,
    "atr_14": 4.25
  },
  "ohlcv": [
    {"date": "2024-01-15", "open": 189.5, "high": 192.0, "low": 188.8, "close": 191.25, "volume": 45000000}
  ]
}
```

### GET `/market/{symbol}/history`
Get historical OHLCV data.

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | Yes | Stock symbol |
| `start` | date | No | Start date (YYYY-MM-DD) |
| `end` | date | No | End date (YYYY-MM-DD) |
| `interval` | string | No | Candle interval (1d, 1h, 15m) |

**Response** `200 OK`:
```json
{
  "symbol": "AAPL",
  "interval": "1d",
  "data": [
    {"date": "2024-01-10", "open": 185.0, "high": 187.5, "low": 184.2, "close": 186.8, "volume": 42000000},
    {"date": "2024-01-11", "open": 186.8, "high": 189.0, "low": 186.0, "close": 188.5, "volume": 38000000}
  ]
}
```

---

## Backtesting

### POST `/backtest/run`
Run a backtest on historical data.

**Request Body**:
```json
{
  "symbol": "AAPL",
  "start_date": "2022-01-01",
  "end_date": "2024-01-01",
  "initial_capital": 100000,
  "risk_tolerance": 0.6
}
```

**Response** `200 OK`:
```json
{
  "backtest_id": "bt_12345",
  "status": "completed",
  "results": {
    "total_return": 45.2,
    "sharpe_ratio": 1.35,
    "max_drawdown": -12.5,
    "win_rate": 58.3,
    "profit_factor": 1.72,
    "total_trades": 156,
    "final_value": 145200.00
  },
  "equity_curve": [
    {"date": "2022-01-01", "value": 100000},
    {"date": "2022-02-01", "value": 102500},
    {"date": "2022-03-01", "value": 98500}
  ],
  "trades": [
    {"date": "2022-01-15", "action": "BUY", "price": 175.50, "quantity": 50, "pnl": null},
    {"date": "2022-01-25", "action": "SELL", "price": 182.30, "quantity": 50, "pnl": 340.00}
  ]
}
```

### GET `/backtest/{backtest_id}`
Get backtest results by ID.

---

## User Profile

### GET `/profile`
Get current user's profile.

**Response** `200 OK`:
```json
{
  "id": "uuid",
  "email": "trader@example.com",
  "name": "John Trader",
  "risk_profile": {
    "tolerance": 0.65,
    "category": "Growth",
    "last_assessed": "2024-01-01T00:00:00Z"
  },
  "preferences": {
    "timeframe": "swing",
    "symbols": ["AAPL", "GOOGL", "MSFT"]
  }
}
```

### POST `/profile/risk-assessment`
Submit risk assessment questionnaire.

**Request Body**:
```json
{
  "answers": [3, 4, 2, 3, 4, 2]
}
```

**Response** `200 OK`:
```json
{
  "risk_tolerance": 0.65,
  "category": "Growth",
  "description": "You have a growth-oriented risk profile. You accept moderate volatility for higher returns.",
  "recommendations": {
    "max_position_size": 0.15,
    "suggested_stop_loss": 0.08,
    "suggested_take_profit": 0.15
  }
}
```

### GET `/profile/portfolio`
Get current portfolio positions.

**Response** `200 OK`:
```json
{
  "total_value": 125000.00,
  "cash": 25000.00,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "avg_entry": 175.50,
      "current_price": 191.25,
      "market_value": 19125.00,
      "unrealized_pnl": 1575.00,
      "pnl_pct": 8.97
    }
  ]
}
```

---

## Health Check

### GET `/health`
Check API health status.

**Response** `200 OK`:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "ml_model": "loaded"
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid symbol format",
    "details": {
      "field": "symbol",
      "value": "invalid!"
    }
  }
}
```

**Error Codes**:
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `UNAUTHORIZED` | 401 | Missing or invalid token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limiting

- **Authenticated users**: 100 requests/minute
- **Signals endpoint**: 10 requests/minute per symbol
- **Backtest endpoint**: 5 requests/minute

---

## Next Steps

- See [Deployment](06_deployment.md) for setup instructions

"""
Trading signals API routes.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.layer1_data_processing.market_data import get_market_data
from app.layer1_data_processing.technical_indicators import compute_indicators

router = APIRouter()

# Try to import prediction service
try:
    from app.services.prediction_service import get_prediction_service
    PREDICTION_AVAILABLE = True
except Exception as e:
    print(f"Prediction service not available: {e}")
    PREDICTION_AVAILABLE = False


class SignalResponse(BaseModel):
    """Trading signal response model."""
    symbol: str
    timestamp: datetime
    action: str  # BUY, SELL, HOLD
    confidence: float
    prediction: dict
    indicators: dict


class MarketDataResponse(BaseModel):
    """Market data response model."""
    symbol: str
    current_price: float
    change_pct: float
    volume: int
    indicators: dict


@router.get("/signals/{symbol}")
async def get_trading_signal(
    symbol: str,
    use_sentiment: bool = Query(False, description="Include sentiment analysis"),
    use_model: bool = Query(True, description="Use trained LSTM model")
):
    """
    Get trading signal for a symbol using trained LSTM model.
    
    Args:
        symbol: Stock symbol (e.g., RELIANCE.NS for NSE)
        use_sentiment: Whether to include sentiment analysis
        use_model: Whether to use trained LSTM model for prediction
    
    Returns:
        Trading signal with action, confidence, and supporting data
    """
    try:
        # Get market data
        data = await get_market_data(symbol, period="3mo")
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Compute indicators
        indicators = compute_indicators(data)
        
        # Get prediction from trained model
        if use_model and PREDICTION_AVAILABLE:
            pred_service = get_prediction_service()
            model_pred = pred_service.predict(symbol)
            
            if model_pred.get("error") is None:
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "action": model_pred["action"],
                    "confidence": round(model_pred["confidence"], 2),
                    "prediction": {
                        "current_price": model_pred["current_price"],
                        "predicted_price": round(model_pred["predicted_price"], 2),
                        "price_change": round(model_pred["price_change"], 2),
                        "change_pct": round(model_pred["change_pct"], 2),
                        "model": model_pred["model"]
                    },
                    "indicators": indicators
                }
        
        # Fallback: Simple rule-based signal
        prediction = {
            "current_price": float(data['Close'].iloc[-1]),
            "predicted_price": float(data['Close'].iloc[-1]),
            "change_pct": 0.0,
            "model": "fallback"
        }
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "action": "HOLD",
            "confidence": 0.5,
            "prediction": prediction,
            "indicators": indicators
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/{symbol}", response_model=MarketDataResponse)
async def get_market_info(
    symbol: str,
    period: str = Query("1mo", description="Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y")
):
    """
    Get market data and technical indicators for a symbol.
    """
    try:
        data = await get_market_data(symbol, period=period)
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        indicators = compute_indicators(data)
        
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
        change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
        
        return MarketDataResponse(
            symbol=symbol,
            current_price=float(current['Close']),
            change_pct=float(change_pct),
            volume=int(current['Volume']),
            indicators=indicators
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist")
async def get_watchlist_signals():
    """
    Get signals for popular NSE stocks using trained model.
    """
    symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
    ]
    
    signals = []
    
    # Try to use prediction service
    if PREDICTION_AVAILABLE:
        pred_service = get_prediction_service()
        for symbol in symbols:
            try:
                pred = pred_service.predict(symbol)
                if pred.get("error") is None:
                    signals.append({
                        "symbol": symbol,
                        "price": round(pred["current_price"], 2),
                        "predicted_price": round(pred["predicted_price"], 2),
                        "change_pct": round(pred["change_pct"], 2),
                        "action": pred["action"],
                        "confidence": round(pred["confidence"], 2),
                        "model": pred["model"]
                    })
                else:
                    # Fallback for this symbol
                    data = await get_market_data(symbol, period="1mo")
                    if data is not None and not data.empty:
                        current = data.iloc[-1]
                        prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
                        change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
                        signals.append({
                            "symbol": symbol,
                            "price": round(float(current['Close']), 2),
                            "predicted_price": round(float(current['Close']), 2),
                            "change_pct": round(float(change_pct), 2),
                            "action": "HOLD",
                            "confidence": 0.5,
                            "model": "fallback"
                        })
            except Exception:
                continue
    else:
        # Fallback mode without model
        for symbol in symbols:
            try:
                data = await get_market_data(symbol, period="1mo")
                if data is not None and not data.empty:
                    current = data.iloc[-1]
                    prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
                    change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
                    
                    signals.append({
                        "symbol": symbol,
                        "price": round(float(current['Close']), 2),
                        "change_pct": round(float(change_pct), 2),
                        "action": "HOLD",
                        "confidence": 0.5
                    })
            except Exception:
                continue
    
    return {"signals": signals, "model_available": PREDICTION_AVAILABLE}

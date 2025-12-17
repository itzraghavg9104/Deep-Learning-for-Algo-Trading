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


@router.get("/signals/{symbol}", response_model=SignalResponse)
async def get_trading_signal(
    symbol: str,
    use_sentiment: bool = Query(False, description="Include sentiment analysis")
):
    """
    Get trading signal for a symbol.
    
    Args:
        symbol: Stock symbol (e.g., RELIANCE.NS for NSE, RELIANCE.BO for BSE)
        use_sentiment: Whether to include sentiment analysis (optional)
    
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
        
        # TODO: Get prediction from DeepAR model
        prediction = {
            "price_mean": float(data['Close'].iloc[-1]),
            "price_std": float(data['Close'].std()),
            "change_pct": 0.0
        }
        
        # TODO: Get action from PPO agent
        action = "HOLD"
        confidence = 0.5
        
        return SignalResponse(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            prediction=prediction,
            indicators=indicators
        )
        
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
    
    Args:
        symbol: Stock symbol (e.g., RELIANCE.NS)
        period: Data period
    
    Returns:
        Current price, change, volume, and indicators
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
    Get signals for popular NSE stocks.
    """
    symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
    ]
    
    signals = []
    for symbol in symbols:
        try:
            data = await get_market_data(symbol, period="1mo")
            if data is not None and not data.empty:
                current = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
                change_pct = ((current['Close'] - prev['Close']) / prev['Close']) * 100
                
                signals.append({
                    "symbol": symbol,
                    "price": float(current['Close']),
                    "change_pct": round(float(change_pct), 2),
                    "action": "HOLD",  # TODO: From PPO agent
                    "confidence": 0.5
                })
        except Exception:
            continue
    
    return {"signals": signals}

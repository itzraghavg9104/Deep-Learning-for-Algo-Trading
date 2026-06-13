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


def _fallback_action_from_change(change_pct: float) -> tuple[str, float]:
    if change_pct >= 0.30:
        return "BUY", min(0.75, 0.52 + min(change_pct, 3.0) / 10.0)
    if change_pct <= -0.30:
        return "SELL", min(0.75, 0.52 + min(abs(change_pct), 3.0) / 10.0)
    return "IDLE", 0.50


def _absolute_trade_plan(trade_plan: Optional[dict], price: float) -> Optional[dict]:
    if not trade_plan:
        return None
    if "profit_target_exit_price" in trade_plan and "capital_amount_inr" in trade_plan:
        return trade_plan
    capital_pct = float(trade_plan.get("capital_per_trade_pct", 0.0))
    tp_sl_ratio = float(trade_plan.get("tp_sl_ratio_target", 0.0))
    max_profit_pct = float(trade_plan.get("max_profit_pct", trade_plan.get("profit_target_pct", 0.0)))
    notional_capital = 100000.0
    capital_amount = max(0.0, notional_capital * (capital_pct / 100.0))
    qty = int(capital_amount / price) if price > 0 else 0
    stop_loss_pct = max(0.5, min(5.0, max_profit_pct / max(tp_sl_ratio, 1.0)))
    take_profit_pct = max(0.5, min(25.0, stop_loss_pct * max(tp_sl_ratio, 1.0)))
    return {
        "capital_per_trade_pct": round(capital_pct, 2),
        "tp_sl_ratio_target": round(tp_sl_ratio, 2),
        "max_profit_pct": round(max_profit_pct, 2),
        "capital_amount_inr": round(capital_amount, 2),
        "position_qty_est": qty,
        "stop_loss_price": round(price * (1 - stop_loss_pct / 100.0), 2),
        "take_profit_price": round(price * (1 + take_profit_pct / 100.0), 2),
        "max_profit_take_price": round(price * (1 + max_profit_pct / 100.0), 2),
        "profit_target_exit_price": round(price * (1 + max_profit_pct / 100.0), 2),
    }


class SignalResponse(BaseModel):
    """Trading signal response model."""
    symbol: str
    timestamp: datetime
    action: str  # HOLD BUY, HOLD SELL, BUY, SELL, IDLE
    confidence: float
    prediction: dict
    indicators: dict
    trade_plan: Optional[dict] = None


class MarketHistoryPoint(BaseModel):
    """OHLCV history point for charting."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketDataResponse(BaseModel):
    """Market data response model."""
    symbol: str
    current_price: float
    change_pct: float
    volume: int
    indicators: dict
    history: List[MarketHistoryPoint]


@router.get("/signals/{symbol}")
async def get_trading_signal(
    symbol: str,
    use_sentiment: bool = Query(False, description="Include sentiment analysis"),
    use_model: bool = Query(True, description="Use trained LSTM model"),
    user_id: Optional[str] = Query(None, description="User id for user-specific PPO model"),
    risk_tolerance: float = Query(0.5, ge=0.0, le=1.0, description="Risk tolerance profile value"),
    capital_per_trade_pref: float = Query(0.1, ge=0.0, le=1.0),
    tp_sl_pref: float = Query(0.4, ge=0.0, le=1.0),
    max_drawdown_pref: float = Query(0.2, ge=0.0, le=1.0),
    cooldown_pref: float = Query(0.1, ge=0.0, le=1.0),
    max_profit_pref: float = Query(0.2, ge=0.0, le=1.0),
    trade_frequency_pref: float = Query(0.15, ge=0.0, le=1.0),
    holding_time_pref: float = Query(0.2, ge=0.0, le=1.0),
    streak_adjustment_pref: float = Query(0.25, ge=0.0, le=1.0),
    intraday_var_pref: float = Query(0.1, ge=0.0, le=1.0),
    slippage_pref: float = Query(0.1, ge=0.0, le=1.0),
    session_bias_pref: float = Query(0.5, ge=0.0, le=1.0),
    news_buffer_pref: float = Query(0.1, ge=0.0, le=1.0),
    partial_tp_pref: float = Query(0.5, ge=0.0, le=1.0),
    breakeven_trigger_pref: float = Query(0.1, ge=0.0, le=1.0),
    breakeven_time_pref: float = Query(0.2, ge=0.0, le=1.0),
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
            model_pred = pred_service.predict(
                symbol,
                risk_tolerance=risk_tolerance,
                behavior_array={
                    "capital_per_trade_pct": capital_per_trade_pref,
                    "tp_sl_ratio_preference": tp_sl_pref,
                    "max_profit_close_pct": max_profit_pref,
                    "trade_frequency_window_score": trade_frequency_pref,
                    "avg_holding_time_score": holding_time_pref,
                    "drawdown_sensitivity": max_drawdown_pref,
                    "streak_risk_adjustment": streak_adjustment_pref,
                    "intraday_var_limit": intraday_var_pref,
                    "entry_slippage_tolerance_bps": slippage_pref,
                    "time_of_day_performance_bias": session_bias_pref,
                    "news_proximity_buffer_min": news_buffer_pref,
                    "partial_tp_preference": partial_tp_pref,
                    "breakeven_migration_trigger_pct": breakeven_trigger_pref,
                    "breakeven_migration_time_min": breakeven_time_pref,
                    "post_loss_rest_min": cooldown_pref,
                },
                user_id=user_id,
            )
            
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
                    "indicators": indicators,
                    "trade_plan": _absolute_trade_plan(model_pred.get("trade_plan"), float(model_pred["current_price"])),
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
            "action": "IDLE",
            "confidence": 0.5,
            "prediction": prediction,
            "indicators": indicators,
            "trade_plan": _absolute_trade_plan({
                "capital_per_trade_pct": round(capital_per_trade_pref * 100, 2),
                "tp_sl_ratio_target": round(max(0.5, tp_sl_pref * 8), 2),
                "max_profit_pct": round(max_profit_pref * 100, 2),
            }, float(data["Close"].iloc[-1])),
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
        
        history = []
        for timestamp, row in data.tail(180).iterrows():
            history.append({
                "timestamp": timestamp.to_pydatetime(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })

        return MarketDataResponse(
            symbol=symbol,
            current_price=float(current["Close"]),
            change_pct=float(change_pct),
            volume=int(current["Volume"]),
            indicators=indicators,
            history=history,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist")
async def get_watchlist_signals(
    user_id: Optional[str] = Query(None, description="User id for user-specific PPO model"),
    risk_tolerance: float = Query(0.5, ge=0.0, le=1.0, description="Risk tolerance profile value"),
    capital_per_trade_pref: float = Query(0.1, ge=0.0, le=1.0),
    tp_sl_pref: float = Query(0.25, ge=0.0, le=1.0),
    max_drawdown_pref: float = Query(0.2, ge=0.0, le=1.0),
    cooldown_pref: float = Query(0.1, ge=0.0, le=1.0),
    max_profit_pref: float = Query(0.2, ge=0.0, le=1.0),
    trade_frequency_pref: float = Query(0.15, ge=0.0, le=1.0),
    holding_time_pref: float = Query(0.2, ge=0.0, le=1.0),
    streak_adjustment_pref: float = Query(0.25, ge=0.0, le=1.0),
    intraday_var_pref: float = Query(0.1, ge=0.0, le=1.0),
    slippage_pref: float = Query(0.1, ge=0.0, le=1.0),
    session_bias_pref: float = Query(0.5, ge=0.0, le=1.0),
    news_buffer_pref: float = Query(0.1, ge=0.0, le=1.0),
    partial_tp_pref: float = Query(0.5, ge=0.0, le=1.0),
    breakeven_trigger_pref: float = Query(0.1, ge=0.0, le=1.0),
    breakeven_time_pref: float = Query(0.2, ge=0.0, le=1.0),
):
    """
    Get signals for popular NSE stocks using trained model.
    """
    symbols = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "ITC.NS",
        "KOTAKBANK.NS",
        "LT.NS",
        "HINDUNILVR.NS",
        "AXISBANK.NS",
        "BAJFINANCE.NS",
        "MARUTI.NS",
        "ASIANPAINT.NS",
        "WIPRO.NS",
        "HCLTECH.NS",
        "SUNPHARMA.NS",
        "TITAN.NS",
        "TATAMOTORS.NS",
    ]
    index_symbols = [
        ("NIFTY 50", "^NSEI"),
        ("NIFTY MIDCAP 150", "NIFTYMIDCAP150.NS"),
        ("NIFTY SMALLCAP 250", "NIFTYSMLCAP250.NS"),
    ]

    signals = []
    index_signals = []
    default_trade_plan = {
        "capital_per_trade_pct": round(capital_per_trade_pref * 100, 2),
        "tp_sl_ratio_target": round(max(0.5, tp_sl_pref * 8), 2),
        "max_profit_pct": round(max_profit_pref * 100, 2),
    }

    async def _day_change(symbol: str) -> Optional[dict]:
        data = await get_market_data(symbol, period="1mo")
        if data is None or data.empty:
            return None
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
        change_pct = ((current["Close"] - prev["Close"]) / prev["Close"]) * 100 if prev["Close"] else 0.0
        return {
            "price": float(current["Close"]),
            "change_pct": float(change_pct),
        }
    
    # Try to use prediction service
    if PREDICTION_AVAILABLE:
        pred_service = get_prediction_service()
        behavior_array = {
            "capital_per_trade_pct": capital_per_trade_pref,
            "tp_sl_ratio_preference": tp_sl_pref,
            "max_profit_close_pct": max_profit_pref,
            "trade_frequency_window_score": trade_frequency_pref,
            "avg_holding_time_score": holding_time_pref,
            "drawdown_sensitivity": max_drawdown_pref,
            "streak_risk_adjustment": streak_adjustment_pref,
            "intraday_var_limit": intraday_var_pref,
            "entry_slippage_tolerance_bps": slippage_pref,
            "time_of_day_performance_bias": session_bias_pref,
            "news_proximity_buffer_min": news_buffer_pref,
            "partial_tp_preference": partial_tp_pref,
            "breakeven_migration_trigger_pct": breakeven_trigger_pref,
            "breakeven_migration_time_min": breakeven_time_pref,
            "post_loss_rest_min": cooldown_pref,
        }
        for symbol in symbols:
            try:
                pred = pred_service.predict(
                    symbol,
                    risk_tolerance=risk_tolerance,
                    behavior_array=behavior_array,
                    user_id=user_id,
                )
                day = await _day_change(symbol)
                if pred.get("error") is None and day is not None:
                    action = pred["action"]
                    target_price = round(pred["predicted_price"], 2) if action in ("BUY", "SELL", "HOLD BUY", "HOLD SELL") else None
                    signals.append({
                        "symbol": symbol,
                        "price": round(day["price"], 2),
                        "predicted_price": round(pred["predicted_price"], 2),
                        "target_price": target_price,
                        "change_pct": round(day["change_pct"], 2),
                        "action": action,
                        "confidence": round(pred["confidence"], 2),
                        "model": pred["model"],
                        "trade_plan": _absolute_trade_plan(pred.get("trade_plan") or default_trade_plan, float(day["price"])),
                    })
                else:
                    # Fallback for this symbol
                    day = await _day_change(symbol)
                    if day is not None:
                        fb_action, fb_conf = _fallback_action_from_change(float(day["change_pct"]))
                        signals.append({
                            "symbol": symbol,
                            "price": round(day["price"], 2),
                            "predicted_price": round(day["price"], 2),
                            "target_price": None,
                            "change_pct": round(day["change_pct"], 2),
                            "action": fb_action,
                            "confidence": round(fb_conf, 2),
                            "model": "fallback",
                            "trade_plan": _absolute_trade_plan(default_trade_plan, float(day["price"])),
                        })
            except Exception:
                continue
    else:
        # Fallback mode without model
        for symbol in symbols:
            try:
                day = await _day_change(symbol)
                if day is not None:
                    fb_action, fb_conf = _fallback_action_from_change(float(day["change_pct"]))
                    signals.append({
                        "symbol": symbol,
                        "price": round(day["price"], 2),
                        "predicted_price": round(day["price"], 2),
                        "target_price": None,
                        "change_pct": round(day["change_pct"], 2),
                        "action": fb_action,
                        "confidence": round(fb_conf, 2),
                        "trade_plan": _absolute_trade_plan(default_trade_plan, float(day["price"])),
                    })
            except Exception:
                continue

    for label, symbol in index_symbols:
        try:
            day = await _day_change(symbol)
            if day is not None:
                index_signals.append(
                    {
                        "label": label,
                        "symbol": symbol,
                        "price": round(day["price"], 2),
                        "change_pct": round(day["change_pct"], 2),
                    }
                )
        except Exception:
            continue

    return {
        "signals": signals,
        "top20": signals,
        "indices": index_signals,
        "model_available": PREDICTION_AVAILABLE,
    }

"""
User profile and risk assessment API routes.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List

from app.trader_behavior.risk_profiler import calculate_risk_score, get_risk_category
from app.api.routes.auth import get_current_user
from app.config import settings
from app.services.demo_store import demo_store

router = APIRouter()


class RiskAssessmentRequest(BaseModel):
    """Risk assessment questionnaire answers."""
    answers: List[int]  # List of answer scores (1-4)


class RiskProfile(BaseModel):
    """User risk profile."""
    risk_tolerance: float
    category: str
    description: str
    recommendations: dict


class UserPreferences(BaseModel):
    """User trading preferences."""
    use_sentiment: bool = False
    preferred_timeframe: str = "swing"  # intraday, swing, position, longterm
    symbols: List[str] = []


@router.post("/risk-assessment", response_model=RiskProfile)
async def submit_risk_assessment(request: RiskAssessmentRequest):
    """
    Submit risk assessment questionnaire and get risk profile.
    
    Args:
        request: List of questionnaire answers (1-4 scale)
    
    Returns:
        Risk profile with tolerance score, category, and recommendations
    """
    if len(request.answers) < 4:
        raise HTTPException(
            status_code=400, 
            detail="At least 4 questionnaire answers required"
        )
    
    # Calculate risk score
    risk_tolerance = calculate_risk_score(request.answers)
    category, description = get_risk_category(risk_tolerance)
    
    # Generate recommendations based on risk profile
    recommendations = {
        "max_position_size": round(0.05 + (risk_tolerance * 0.15), 2),  # 5-20%
        "suggested_stop_loss": round(0.05 + (risk_tolerance * 0.10), 2),  # 5-15%
        "suggested_take_profit": round(0.10 + (risk_tolerance * 0.20), 2),  # 10-30%
    }
    
    return RiskProfile(
        risk_tolerance=risk_tolerance,
        category=category,
        description=description,
        recommendations=recommendations
    )


@router.get("/")
async def get_profile(current_user=Depends(get_current_user)):
    """
    Get current user profile.
    """
    if settings.DEMO_MODE:
        profile = demo_store.get_or_create_profile(current_user.id)
        return {
            "id": current_user.id,
            "email": current_user.email,
            "risk_profile": {
                "tolerance": profile.risk_tolerance,
                "category": profile.risk_category,
            },
            "preferences": {
                "use_sentiment": profile.use_sentiment,
                "preferred_timeframe": profile.preferred_timeframe,
                "symbols": list(profile.symbols),
            },
        }

    return {
        "id": current_user.id,
        "email": current_user.email,
    }


@router.put("/preferences")
async def update_preferences(preferences: UserPreferences, current_user=Depends(get_current_user)):
    """
    Update user trading preferences.
    """
    if settings.DEMO_MODE:
        demo_store.update_profile(
            current_user.id,
            use_sentiment=preferences.use_sentiment,
            preferred_timeframe=preferences.preferred_timeframe,
            symbols=tuple(preferences.symbols),
        )
        return {
            "message": "Preferences updated",
            "preferences": preferences.model_dump(),
        }

    return {
        "message": "Preferences updated",
        "preferences": preferences.model_dump(),
    }


@router.get("/trades")
async def get_trade_history(current_user=Depends(get_current_user)):
    """
    Get trade history for the current user.
    """
    if settings.DEMO_MODE:
        trades = demo_store.get_user_trades(current_user.id)
        total_pnl = sum(t.get("pnl", 0.0) for t in trades)
        winning = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
        return {
            "trades": trades,
            "total_pnl": total_pnl,
            "win_rate": round(winning / len(trades), 2) if trades else 0.0,
        }

    return {"trades": [], "total_pnl": 0.0, "win_rate": 0.0}

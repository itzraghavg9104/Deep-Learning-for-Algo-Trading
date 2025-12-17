"""
User profile and risk assessment API routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.trader_behavior.risk_profiler import calculate_risk_score, get_risk_category

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
async def get_profile():
    """
    Get current user profile.
    """
    # TODO: Get from database based on authenticated user
    return {
        "id": "demo_user",
        "name": "Demo Trader",
        "risk_profile": {
            "tolerance": 0.5,
            "category": "Moderate"
        },
        "preferences": {
            "use_sentiment": False,
            "preferred_timeframe": "swing",
            "symbols": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        }
    }


@router.put("/preferences")
async def update_preferences(preferences: UserPreferences):
    """
    Update user trading preferences.
    """
    # TODO: Save to database
    return {
        "message": "Preferences updated",
        "preferences": preferences.model_dump()
    }

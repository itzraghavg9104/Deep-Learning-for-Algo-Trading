"""
User profile and risk assessment API routes.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.trader_behavior.risk_profiler import calculate_risk_score, get_risk_category
from app.api.routes.auth import get_current_user
from app.config import settings
from app.services.demo_store import demo_store
from app.services.firestore_store import firestore_store

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

class BehaviorAssessmentRequest(BaseModel):
    """Expanded behavior questionnaire response payload."""
    answers: Dict[str, Any]


class TradeEvaluationRequest(BaseModel):
    """Evaluate whether a trade follows user behavior constraints."""
    trade_id: Optional[str] = None
    symbol: str
    planned: Dict[str, Any]
    executed: Dict[str, Any]
    pnl: float = 0.0
    pnl_pct: float = 0.0


def _build_behavior_array(raw_answers: Dict[str, Any]) -> Dict[str, float]:
    """Map questionnaire answers to a normalized behavior vector."""
    def pct(name: str, default: float) -> float:
        value = raw_answers.get(name, default)
        try:
            return max(0.0, min(float(value) / 100.0, 1.0))
        except Exception:
            return default

    def number(name: str, default: float, max_value: float) -> float:
        value = raw_answers.get(name, default)
        try:
            return max(0.0, min(float(value) / max_value, 1.0))
        except Exception:
            return max(0.0, min(default / max_value, 1.0))

    return {
        "capital_per_trade_pct": pct("capital_per_trade_pct", 0.1),
        "tp_sl_ratio_preference": number("tp_sl_ratio", 2.0, 5.0),
        "max_profit_close_pct": pct("max_profit_close_pct", 0.2),
        "trade_frequency_window_score": number("max_trades_per_day", 5.0, 30.0),
        "post_loss_rest_min": number("post_loss_rest_min", 30.0, 720.0),
        "drawdown_sensitivity": pct("max_drawdown_pct", 0.2),
        "streak_risk_adjustment": number("loss_streak_reduce_pct", 20.0, 100.0),
        "intraday_var_limit": pct("intraday_var_pct", 0.03),
        "entry_slippage_tolerance_bps": number("entry_slippage_bps", 15.0, 200.0),
        "news_proximity_buffer_min": number("news_buffer_min", 30.0, 240.0),
        "partial_tp_preference": number("partial_tp_frequency", 2.0, 4.0),
        "breakeven_migration_trigger_pct": pct("breakeven_trigger_pct", 0.01),
        "breakeven_migration_time_min": number("breakeven_migration_time_min", 60.0, 720.0),
    }


@router.post("/risk-assessment", response_model=RiskProfile)
async def submit_risk_assessment(request: RiskAssessmentRequest, current_user=Depends(get_current_user)):
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
    
    response = RiskProfile(
        risk_tolerance=risk_tolerance,
        category=category,
        description=description,
        recommendations=recommendations
    )
    if settings.FIREBASE_AUTH_ENABLED:
        firestore_store.save_risk_assessment(
            str(current_user.id),
            {
                "answers": request.answers,
                "risk_tolerance": risk_tolerance,
                "category": category,
                "description": description,
                "recommendations": recommendations,
            },
        )
    return response


@router.post("/behavior-assessment")
async def submit_behavior_assessment(request: BehaviorAssessmentRequest, current_user=Depends(get_current_user)):
    """
    Submit expanded behavior assessment and persist normalized behavior array.
    """
    behavior_array = _build_behavior_array(request.answers)
    payload = {
        "behavior_array": behavior_array,
        "raw_answers": request.answers,
        "updated_at": datetime.utcnow().isoformat(),
    }

    if settings.FIREBASE_AUTH_ENABLED:
        firestore_store.save_behavior_profile(str(current_user.id), payload)
        return {"message": "Behavior profile saved", "behavior_profile": payload}

    return {"message": "Behavior profile computed (demo)", "behavior_profile": payload}


@router.get("/")
async def get_profile(current_user=Depends(get_current_user)):
    """
    Get current user profile.
    """
    if settings.FIREBASE_AUTH_ENABLED:
        return firestore_store.get_profile(str(current_user.id), current_user.email)

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
    if settings.FIREBASE_AUTH_ENABLED:
        saved = firestore_store.save_preferences(
            str(current_user.id),
            current_user.email,
            preferences.model_dump(),
        )
        return {
            "message": "Preferences updated",
            "preferences": saved,
        }

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
    if settings.FIREBASE_AUTH_ENABLED:
        trades = firestore_store.get_user_trades(str(current_user.id), current_user.email)
        total_pnl = sum(float(t.get("pnl", 0.0)) for t in trades)
        winning = sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0)
        return {
            "trades": trades,
            "total_pnl": total_pnl,
            "win_rate": round(winning / len(trades), 2) if trades else 0.0,
        }

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


@router.post("/trades/evaluate")
async def evaluate_trade(request: TradeEvaluationRequest, current_user=Depends(get_current_user)):
    """
    Evaluate a trade against planned behavior constraints and store report.
    """
    planned_size = float(request.planned.get("capital_per_trade_pct", 0.0))
    executed_size = float(request.executed.get("capital_per_trade_pct", 0.0))
    cooldown_ok = bool(request.executed.get("cooldown_respected", True))

    violations: List[str] = []
    if executed_size > planned_size > 0:
        violations.append("capital_limit_exceeded")
    if not cooldown_ok:
        violations.append("cooldown_violation")

    compliance_score = max(0.0, 100.0 - (30.0 * len(violations)))

    report = {
        "trade_id": request.trade_id,
        "symbol": request.symbol,
        "compliance_score": compliance_score,
        "violations": violations,
        "planned": request.planned,
        "executed": request.executed,
        "pnl": request.pnl,
        "pnl_pct": request.pnl_pct,
        "status": "pass" if not violations else "warn",
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    if settings.FIREBASE_AUTH_ENABLED:
        stored = firestore_store.save_trade_evaluation(str(current_user.id), report)
        return {"message": "Trade evaluated", "evaluation": stored}

    return {"message": "Trade evaluated (demo)", "evaluation": report}

"""
User profile and risk assessment API routes.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.routes.auth import get_current_user
from app.config import settings
from app.services.demo_store import demo_store
from app.services.firestore_store import firestore_store
from app.services.user_model_training_service import trigger_user_retraining, get_user_training_status
from app.trader_behavior.risk_profiler import calculate_risk_score, get_risk_category

router = APIRouter()

MIN_BEHAVIOR_QUESTIONS = 24


class RiskAssessmentRequest(BaseModel):
    """Legacy risk assessment questionnaire answers."""
    answers: List[int]


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pct(name: str, raw_answers: Dict[str, Any], default: float) -> float:
    value = raw_answers.get(name, default)
    return max(0.0, min(_to_float(value, default) / 100.0, 1.0))


def _number(name: str, raw_answers: Dict[str, Any], default: float, max_value: float) -> float:
    value = raw_answers.get(name, default)
    return max(0.0, min(_to_float(value, default) / max_value, 1.0))


def _extract_question_scores(raw_answers: Dict[str, Any]) -> List[int]:
    scores: List[int] = []

    question_scores = raw_answers.get("question_scores", [])
    if isinstance(question_scores, list):
        for item in question_scores:
            if isinstance(item, dict):
                scores.append(int(max(1, min(5, _to_float(item.get("score"), 3.0)))))
            else:
                scores.append(int(max(1, min(5, _to_float(item, 3.0)))))

    if not scores:
        for key, value in raw_answers.items():
            if key.startswith("q_"):
                scores.append(int(max(1, min(5, _to_float(value, 3.0)))))

    return scores


def _build_behavior_array(raw_answers: Dict[str, Any]) -> Dict[str, float]:
    """Map questionnaire answers to a normalized behavior vector."""
    return {
        # User requested core feedback metrics
        "capital_per_trade_pct": _pct("capital_per_trade_pct", raw_answers, 10.0),
        "tp_sl_ratio_preference": _number("tp_sl_ratio", raw_answers, 2.0, 8.0),
        "max_profit_close_pct": _pct("max_profit_close_pct", raw_answers, 20.0),

        # Overtrading & impulse
        "trade_frequency_window_score": _number("max_trades_per_day", raw_answers, 6.0, 40.0),
        "avg_holding_time_score": _number("avg_holding_time_min", raw_answers, 240.0, 10080.0),
        "post_loss_rest_min": _number("post_loss_rest_min", raw_answers, 45.0, 1440.0),

        # Risk & account management
        "drawdown_sensitivity": _pct("max_drawdown_pct", raw_answers, 15.0),
        "streak_risk_adjustment": _number("loss_streak_reduce_pct", raw_answers, 25.0, 100.0),
        "intraday_var_limit": _pct("intraday_var_pct", raw_answers, 3.0),

        # Market context execution
        "entry_slippage_tolerance_bps": _number("entry_slippage_bps", raw_answers, 12.0, 300.0),
        "time_of_day_performance_bias": _number("session_consistency_score", raw_answers, 50.0, 100.0),
        "news_proximity_buffer_min": _number("news_buffer_min", raw_answers, 30.0, 360.0),

        # Advanced trade management
        "partial_tp_preference": _number("partial_tp_frequency", raw_answers, 2.0, 4.0),
        "breakeven_migration_trigger_pct": _pct("breakeven_trigger_pct", raw_answers, 1.0),
        "breakeven_migration_time_min": _number("breakeven_migration_time_min", raw_answers, 60.0, 1440.0),
    }


@router.post("/risk-assessment", response_model=RiskProfile)
async def submit_risk_assessment(request: RiskAssessmentRequest, current_user=Depends(get_current_user)):
    """
    Legacy endpoint for simple risk scoring.
    """
    if len(request.answers) < 4:
        raise HTTPException(status_code=400, detail="At least 4 questionnaire answers required")

    risk_tolerance = calculate_risk_score(request.answers)
    category, description = get_risk_category(risk_tolerance)

    recommendations = {
        "max_position_size": round(0.05 + (risk_tolerance * 0.15), 2),
        "suggested_stop_loss": round(0.05 + (risk_tolerance * 0.10), 2),
        "suggested_take_profit": round(0.10 + (risk_tolerance * 0.20), 2),
    }

    response = RiskProfile(
        risk_tolerance=risk_tolerance,
        category=category,
        description=description,
        recommendations=recommendations,
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
    Submit expanded (~30 question) behavior assessment and persist normalized behavior array.
    """
    question_scores = _extract_question_scores(request.answers)
    if len(question_scores) < MIN_BEHAVIOR_QUESTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_BEHAVIOR_QUESTIONS} behavior-question answers are required",
        )

    # Convert 1-5 answers to 1-4 for legacy scoring compatibility.
    normalized_for_risk = [max(1, min(4, int(round(((s - 1) / 4) * 3 + 1)))) for s in question_scores]
    risk_tolerance = calculate_risk_score(normalized_for_risk)
    category, description = get_risk_category(risk_tolerance)

    recommendations = {
        "max_position_size": round(0.05 + (risk_tolerance * 0.15), 2),
        "suggested_stop_loss": round(0.05 + (risk_tolerance * 0.10), 2),
        "suggested_take_profit": round(0.10 + (risk_tolerance * 0.20), 2),
    }

    behavior_array = _build_behavior_array(request.answers)
    profile_payload = {
        "risk_tolerance": risk_tolerance,
        "category": category,
        "description": description,
        "recommendations": recommendations,
    }
    payload = {
        "behavior_array": behavior_array,
        "raw_answers": request.answers,
        "question_count": len(question_scores),
        "risk_profile": profile_payload,
        "updated_at": datetime.utcnow().isoformat(),
    }

    if settings.FIREBASE_AUTH_ENABLED:
        firestore_store.save_risk_assessment(str(current_user.id), profile_payload)
        firestore_store.save_behavior_profile(str(current_user.id), payload)
        trigger_user_retraining(str(current_user.id), behavior_array)
        return {
            "message": "Behavior profile saved. Per-user model retraining started.",
            "behavior_profile": payload,
            "model_training": {
                "started": True,
                "scope": "user-specific-ppo",
                "user_id": str(current_user.id),
            },
        }

    if settings.DEMO_MODE:
        demo_store.update_profile(
            current_user.id,
            risk_tolerance=risk_tolerance,
            risk_category=category,
            behavior_profile=payload,
        )
        trigger_user_retraining(str(current_user.id), behavior_array)

    return {
        "message": "Behavior profile computed (demo). Per-user model retraining started.",
        "behavior_profile": payload,
        "model_training": {
            "started": True,
            "scope": "user-specific-ppo",
            "user_id": str(current_user.id),
        },
    }


@router.get("/")
async def get_profile(current_user=Depends(get_current_user)):
    """Get current user profile."""
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
            "behavior_profile": profile.behavior_profile,
        }

    return {
        "id": current_user.id,
        "email": current_user.email,
    }


@router.put("/preferences")
async def update_preferences(preferences: UserPreferences, current_user=Depends(get_current_user)):
    """Update user trading preferences."""
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
    """Get trade history for the current user."""
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


@router.get("/model-training-status")
async def model_training_status(current_user=Depends(get_current_user)):
    """Get per-user model training status."""
    return get_user_training_status(str(current_user.id))


@router.post("/trades/evaluate")
async def evaluate_trade(request: TradeEvaluationRequest, current_user=Depends(get_current_user)):
    """Evaluate a trade against planned behavior constraints and store report."""
    planned_size = _to_float(request.planned.get("capital_per_trade_pct"), 0.0)
    executed_size = _to_float(request.executed.get("capital_per_trade_pct"), 0.0)

    planned_tp_sl = _to_float(request.planned.get("tp_sl_ratio"), 0.0)
    executed_tp_sl = _to_float(request.executed.get("tp_sl_ratio"), 0.0)

    planned_max_profit_close = _to_float(request.planned.get("max_profit_close_pct"), 0.0)
    executed_max_profit_close = _to_float(request.executed.get("max_profit_close_pct"), 0.0)

    cooldown_ok = bool(request.executed.get("cooldown_respected", True))

    violations: List[str] = []
    if executed_size > planned_size > 0:
        violations.append("capital_limit_exceeded")
    if planned_tp_sl > 0 and executed_tp_sl < planned_tp_sl:
        violations.append("tp_sl_ratio_below_plan")
    if planned_max_profit_close > 0 and executed_max_profit_close < planned_max_profit_close:
        violations.append("premature_profit_close")
    if not cooldown_ok:
        violations.append("cooldown_violation")

    compliance_score = max(0.0, 100.0 - (20.0 * len(violations)))

    report = {
        "trade_id": request.trade_id,
        "symbol": request.symbol,
        "compliance_score": compliance_score,
        "violations": violations,
        "planned": request.planned,
        "executed": request.executed,
        "feedback_loop": {
            "capital_pct_delta": round(executed_size - planned_size, 4),
            "tp_sl_ratio_delta": round(executed_tp_sl - planned_tp_sl, 4),
            "max_profit_close_pct_delta": round(executed_max_profit_close - planned_max_profit_close, 4),
        },
        "pnl": request.pnl,
        "pnl_pct": request.pnl_pct,
        "status": "pass" if not violations else "warn",
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    if settings.FIREBASE_AUTH_ENABLED:
        stored = firestore_store.save_trade_evaluation(str(current_user.id), report)
        return {"message": "Trade evaluated", "evaluation": stored}

    return {"message": "Trade evaluated (demo)", "evaluation": report}

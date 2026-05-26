"""
Backtesting API routes.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import date
from sqlalchemy.orm import Session

from app.models.db.database import get_db
from app.models.db.db_service import DBService
from app.services.backtest_service import BacktestService
from app.config import settings
from app.services.demo_store import demo_store

router = APIRouter()

# Initialize services
backtest_service = BacktestService()

class BacktestRequest(BaseModel):
    """Backtest request model."""
    symbol: str
    start_date: date
    end_date: date
    initial_capital: float = 100000.0
    risk_tolerance: float = 0.5


class BacktestResultResponse(BaseModel):
    """Backtest result model."""
    backtest_id: Optional[str] = None
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    final_value: float
    trades: List[Dict] = []
    equity_curve: List[float] = []


@router.post("/run", response_model=BacktestResultResponse)
async def run_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    """
    Run a backtest on historical data.
    """
    try:
        # 1. Run the simulation
        result = backtest_service.run(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            risk_tolerance=request.risk_tolerance
        )
        
        # 2. Persist to database (if user is authenticated, for now we assume a default user or just test)
        # TODO: Add authentication and use real user_id
        # Mock user_id for now if no auth
        user_id = 1

        if settings.DEMO_MODE:
            db_backtest = demo_store.create_backtest_result(
                user_id=user_id,
                symbol=request.symbol,
                result=result,
            )
        else:
            db_service = DBService(db)
            db_backtest = db_service.create_backtest_result(
                user_id=user_id,
                symbol=request.symbol,
                metrics=result,
                config=request.dict()
            )
        
        # 3. Return results
        return BacktestResultResponse(
            backtest_id=str(db_backtest.id),
            **result
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/{backtest_id}")
async def get_backtest_result(backtest_id: str):
    """
    Get backtest results by ID.
    """
    try:
        bid = int(backtest_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid backtest ID")

    if settings.DEMO_MODE:
        bt = demo_store.get_backtest(bid)
        if bt is None:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return {
            "backtest_id": str(bt.id),
            "symbol": bt.symbol,
            "created_at": bt.created_at.isoformat(),
            **bt.result,
        }

    raise HTTPException(status_code=404, detail="Backtest not found")

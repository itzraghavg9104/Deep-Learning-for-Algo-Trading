from sqlalchemy.orm import Session
from .models import User, Trade, BacktestResult, RiskProfile
from typing import List, Optional, Dict
from datetime import datetime

class DBService:
    """Service for handling all database CRUD operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    # User Operations
    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()
    
    def create_user(self, email: str, hashed_password: str) -> User:
        db_user = User(email=email, hashed_password=hashed_password)
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user
    
    # Trade Operations
    def create_trade(self, user_id: int, symbol: str, action: str, quantity: int, price: float, pnl: float = 0.0) -> Trade:
        db_trade = Trade(
            user_id=user_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            pnl=pnl
        )
        self.db.add(db_trade)
        self.db.commit()
        self.db.refresh(db_trade)
        return db_trade
    
    def get_user_trades(self, user_id: int) -> List[Trade]:
        return self.db.query(Trade).filter(Trade.user_id == user_id).order_by(Trade.timestamp.desc()).all()
    
    # Backtest Operations
    def create_backtest_result(self, user_id: int, symbol: str, metrics: Dict, config: Dict) -> BacktestResult:
        db_backtest = BacktestResult(
            user_id=user_id,
            symbol=symbol,
            total_return=metrics.get('total_return'),
            sharpe_ratio=metrics.get('sharpe_ratio'),
            max_drawdown=metrics.get('max_drawdown'),
            win_rate=metrics.get('win_rate'),
            profit_factor=metrics.get('profit_factor'),
            total_trades=metrics.get('total_trades'),
            final_value=metrics.get('final_value'),
            config=config
        )
        self.db.add(db_backtest)
        self.db.commit()
        self.db.refresh(db_backtest)
        return db_backtest
    
    def get_user_backtests(self, user_id: int) -> List[BacktestResult]:
        return self.db.query(BacktestResult).filter(BacktestResult.user_id == user_id).order_by(BacktestResult.timestamp.desc()).all()
    
    # Risk Profile Operations
    def update_risk_profile(self, user_id: int, score: float, answers: Dict) -> RiskProfile:
        db_profile = self.db.query(RiskProfile).filter(RiskProfile.user_id == user_id).first()
        if db_profile:
            db_profile.score = score
            db_profile.answers = answers
        else:
            db_profile = RiskProfile(user_id=user_id, score=score, answers=answers)
            self.db.add(db_profile)
        
        self.db.commit()
        self.db.refresh(db_profile)
        return db_profile
    
    def get_risk_profile(self, user_id: int) -> Optional[RiskProfile]:
        return self.db.query(RiskProfile).filter(RiskProfile.user_id == user_id).first()

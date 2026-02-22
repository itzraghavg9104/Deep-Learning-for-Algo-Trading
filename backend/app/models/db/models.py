from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    trades = relationship("Trade", back_populates="user")
    backtests = relationship("BacktestResult", back_populates="user")
    risk_profile = relationship("RiskProfile", back_populates="user", uselist=False)

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False)  # BUY, SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    pnl = Column(Float, default=0.0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="trades")

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, nullable=False)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
    final_value = Column(Float)
    config = Column(JSON)  # Store parameters used (start_date, end_date, etc.)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="backtests")

class RiskProfile(Base):
    __tablename__ = "risk_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    score = Column(Float)  # 0.0 to 1.0
    answers = Column(JSON)  # Store questionnaire answers
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    user = relationship("User", back_populates="risk_profile")

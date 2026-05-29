"""
In-memory demo store for running without Postgres/Redis.

This store is process-local and resets on restart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class DemoUser:
    id: int
    email: str
    hashed_password: str
    is_active: bool = True


@dataclass
class DemoBacktest:
    id: int
    user_id: int
    symbol: str
    created_at: datetime
    result: dict


@dataclass
class DemoProfile:
    user_id: int
    risk_tolerance: float = 0.5
    risk_category: str = "Moderate"
    use_sentiment: bool = False
    preferred_timeframe: str = "swing"
    symbols: tuple = ("RELIANCE.NS", "TCS.NS", "INFY.NS")
    behavior_profile: Dict[str, Any] = field(default_factory=dict)


class DemoStore:
    def __init__(self) -> None:
        self._users_by_email: Dict[str, DemoUser] = {}
        self._profiles_by_user_id: Dict[int, DemoProfile] = {}
        self._backtests_by_id: Dict[int, DemoBacktest] = {}
        self._user_id_seq = 1
        self._backtest_id_seq = 1

    def get_user_by_email(self, email: str) -> Optional[DemoUser]:
        return self._users_by_email.get(email.lower())

    def get_or_create_profile(self, user_id: int) -> DemoProfile:
        if user_id not in self._profiles_by_user_id:
            self._profiles_by_user_id[user_id] = DemoProfile(user_id=user_id)
        return self._profiles_by_user_id[user_id]

    def update_profile(self, user_id: int, **kwargs) -> DemoProfile:
        profile = self.get_or_create_profile(user_id)
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        self._profiles_by_user_id[user_id] = profile
        return profile

    def create_user(self, email: str, hashed_password: str) -> DemoUser:
        user = DemoUser(id=self._user_id_seq, email=email.lower(), hashed_password=hashed_password)
        self._users_by_email[user.email] = user
        self._user_id_seq += 1
        return user

    def get_backtest(self, backtest_id: int) -> Optional[DemoBacktest]:
        return self._backtests_by_id.get(backtest_id)

    def get_user_trades(self, user_id: int) -> list:
        trades = []
        for bt in self._backtests_by_id.values():
            if bt.user_id != user_id:
                continue
            for t in bt.result.get("trades", []):
                trades.append({
                    "id": f"BT{bt.id}-{t.get('step', 0)}",
                    "date": str(t.get("date", bt.created_at.date())),
                    "symbol": bt.symbol,
                    "action": t.get("action", "BUY"),
                    "quantity": t.get("quantity", 0),
                    "price": t.get("price", 0.0),
                    "pnl": t.get("pnl", 0.0),
                })
        return trades

    def create_backtest_result(self, user_id: int, symbol: str, result: dict) -> DemoBacktest:
        backtest = DemoBacktest(
            id=self._backtest_id_seq,
            user_id=user_id,
            symbol=symbol,
            created_at=datetime.utcnow(),
            result=result,
        )
        self._backtests_by_id[backtest.id] = backtest
        self._backtest_id_seq += 1
        return backtest


demo_store = DemoStore()

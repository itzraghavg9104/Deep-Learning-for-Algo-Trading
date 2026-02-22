"""
In-memory demo store for running without Postgres/Redis.

This store is process-local and resets on restart.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


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


class DemoStore:
    def __init__(self) -> None:
        self._users_by_email: Dict[str, DemoUser] = {}
        self._backtests_by_id: Dict[int, DemoBacktest] = {}
        self._user_id_seq = 1
        self._backtest_id_seq = 1

    def get_user_by_email(self, email: str) -> Optional[DemoUser]:
        return self._users_by_email.get(email.lower())

    def create_user(self, email: str, hashed_password: str) -> DemoUser:
        user = DemoUser(id=self._user_id_seq, email=email.lower(), hashed_password=hashed_password)
        self._users_by_email[user.email] = user
        self._user_id_seq += 1
        return user

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

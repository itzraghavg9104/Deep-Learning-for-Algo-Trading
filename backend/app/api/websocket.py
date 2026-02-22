"""
WebSocket endpoints for live price updates.
"""
import asyncio
from datetime import datetime
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.layer1_data_processing.market_data import get_market_data, normalize_symbol

router = APIRouter()


async def build_price_update(symbol: str) -> dict | None:
    data = await get_market_data(symbol, period="1d", interval="5m")
    if data is None or data.empty:
        return None

    current = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
    change_pct = ((current["Close"] - prev["Close"]) / prev["Close"]) * 100 if prev["Close"] else 0

    return {
        "symbol": symbol,
        "price": round(float(current["Close"]), 2),
        "change_pct": round(float(change_pct), 2),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.websocket("/ws/prices")
async def prices_ws(websocket: WebSocket):
    await websocket.accept()
    subscriptions: Set[str] = set()
    sender_task: asyncio.Task | None = None

    async def sender_loop():
        while True:
            if not subscriptions:
                await asyncio.sleep(2)
                continue

            updates = []
            for symbol in list(subscriptions):
                update = await build_price_update(symbol)
                if update:
                    updates.append(update)

            if updates:
                await websocket.send_json({"type": "prices", "data": updates})

            await asyncio.sleep(30)

    sender_task = asyncio.create_task(sender_loop())

    try:
        while True:
            message = await websocket.receive_json()
            action = message.get("action")
            symbols = message.get("symbols") or []

            if action == "subscribe":
                for symbol in symbols:
                    subscriptions.add(normalize_symbol(symbol))
            elif action == "unsubscribe":
                for symbol in symbols:
                    subscriptions.discard(normalize_symbol(symbol))
            elif action == "set":
                normalized = {normalize_symbol(symbol) for symbol in symbols}
                subscriptions = normalized
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if sender_task:
            sender_task.cancel()
    except Exception:
        if sender_task:
            sender_task.cancel()
        await websocket.close()

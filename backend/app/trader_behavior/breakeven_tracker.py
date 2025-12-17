"""
Break-even price tracker for position management.

Tracks entry prices, calculates break-even, and monitors P&L.
"""
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    total_cost: float = 0.0
    first_entry_date: Optional[datetime] = None
    last_update: Optional[datetime] = None


class BreakevenTracker:
    """
    Tracks break-even prices and P&L for trading positions.
    """
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
    
    def add_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: float,
        timestamp: Optional[datetime] = None
    ) -> dict:
        """
        Record a trade and update position.
        
        Args:
            symbol: Stock symbol
            action: "BUY" or "SELL"
            price: Trade price
            quantity: Number of shares
            timestamp: Trade timestamp
        
        Returns:
            Updated position info
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        symbol = symbol.upper()
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        pos = self.positions[symbol]
        
        if action.upper() == "BUY":
            # Calculate new weighted average entry price
            new_cost = price * quantity
            pos.total_cost += new_cost
            pos.quantity += quantity
            
            if pos.quantity > 0:
                pos.avg_entry_price = pos.total_cost / pos.quantity
            
            if pos.first_entry_date is None:
                pos.first_entry_date = timestamp
        
        elif action.upper() == "SELL":
            # Reduce position
            pos.quantity -= quantity
            
            if pos.quantity <= 0:
                # Position closed
                pos.quantity = 0
                pos.avg_entry_price = 0
                pos.total_cost = 0
                pos.first_entry_date = None
            else:
                # Update total cost proportionally
                pos.total_cost = pos.avg_entry_price * pos.quantity
        
        pos.last_update = timestamp
        
        return self.get_position_info(symbol)
    
    def get_position_info(self, symbol: str) -> dict:
        """
        Get current position information.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Position details including break-even price
        """
        symbol = symbol.upper()
        pos = self.positions.get(symbol)
        
        if pos is None or pos.quantity == 0:
            return {
                "symbol": symbol,
                "has_position": False,
                "quantity": 0,
                "breakeven_price": 0,
            }
        
        return {
            "symbol": symbol,
            "has_position": True,
            "quantity": pos.quantity,
            "breakeven_price": round(pos.avg_entry_price, 2),
            "total_cost": round(pos.total_cost, 2),
            "first_entry_date": pos.first_entry_date.isoformat() if pos.first_entry_date else None,
            "last_update": pos.last_update.isoformat() if pos.last_update else None,
        }
    
    def calculate_pnl(self, symbol: str, current_price: float) -> dict:
        """
        Calculate current P&L for a position.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
        
        Returns:
            P&L information
        """
        symbol = symbol.upper()
        pos = self.positions.get(symbol)
        
        if pos is None or pos.quantity == 0:
            return {
                "symbol": symbol,
                "has_position": False,
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0,
                "distance_to_breakeven_pct": 0,
            }
        
        current_value = pos.quantity * current_price
        unrealized_pnl = current_value - pos.total_cost
        unrealized_pnl_pct = (unrealized_pnl / pos.total_cost) * 100 if pos.total_cost > 0 else 0
        
        # Distance from current price to break-even
        distance_pct = ((current_price - pos.avg_entry_price) / pos.avg_entry_price) * 100
        
        return {
            "symbol": symbol,
            "has_position": True,
            "quantity": pos.quantity,
            "breakeven_price": round(pos.avg_entry_price, 2),
            "current_price": round(current_price, 2),
            "current_value": round(current_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
            "distance_to_breakeven_pct": round(distance_pct, 2),
            "in_profit": unrealized_pnl > 0,
        }
    
    def get_all_positions(self) -> list:
        """Get all active positions."""
        return [
            self.get_position_info(symbol)
            for symbol, pos in self.positions.items()
            if pos.quantity > 0
        ]
    
    def clear_position(self, symbol: str):
        """Clear a position."""
        symbol = symbol.upper()
        if symbol in self.positions:
            del self.positions[symbol]


# Global tracker instance
_tracker = BreakevenTracker()


def get_tracker() -> BreakevenTracker:
    """Get the global tracker instance."""
    return _tracker

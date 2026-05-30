"""
Canonical trading action space used across training and inference.
"""

ACTION_LABELS = [
    "HOLD BUY",
    "HOLD SELL",
    "BUY",
    "SELL",
    "IDLE",
]

ACTION_HOLD_BUY = 0
ACTION_HOLD_SELL = 1
ACTION_BUY = 2
ACTION_SELL = 3
ACTION_IDLE = 4

# Trader Behavior Module

## Overview

This is the **unique selling proposition (USP)** of our system. Unlike traditional algo trading systems that ignore the human element, our platform adapts to individual trader psychology and risk preferences.

---

## The Human Element in Trading

Professional traders exhibit specific behaviors:

| Behavior | Description | Our Solution |
|----------|-------------|--------------|
| **Risk Tolerance** | How much loss can they stomach? | Personalized position sizing |
| **Loss Aversion** | Prefer avoiding losses over gains | Adaptive stop-loss levels |
| **Overconfidence** | Overestimate abilities | Conservative guardrails |
| **Time Horizon** | Short-term vs long-term focus | Timeframe-adjusted signals |
| **Break-Even Fixation** | Psychological price levels | Break-even tracking |

---

## Components

### 1. Risk Profiler

**Purpose**: Assess trader's risk tolerance through an interactive questionnaire

**Questionnaire Categories**:

#### Investment Experience
```
Q1: How many years of trading experience do you have?
    a) 0-1 years (1 point)
    b) 1-3 years (2 points)
    c) 3-5 years (3 points)
    d) 5+ years (4 points)
```

#### Loss Tolerance
```
Q2: If your portfolio dropped 20% in a week, you would:
    a) Panic sell everything (1 point)
    b) Sell some positions (2 points)
    c) Hold and wait (3 points)
    d) Buy more at lower prices (4 points)
```

#### Time Horizon
```
Q3: What is your typical holding period?
    a) Intraday (1 point)
    b) Days to weeks (2 points)
    c) Weeks to months (3 points)
    d) Months to years (4 points)
```

#### Risk Preferences
```
Q4: Choose your preferred scenario:
    a) Guaranteed 5% annual return (1 point)
    b) 50% chance of 15% or 0% return (2 points)
    c) 50% chance of 25% or -10% return (3 points)
    d) 50% chance of 50% or -30% return (4 points)
```

**Scoring**:
```python
def calculate_risk_score(answers: List[int]) -> float:
    """
    Calculate normalized risk tolerance score.
    
    Returns:
        float: Risk tolerance [0.0 (conservative) to 1.0 (aggressive)]
    """
    total_points = sum(answers)
    max_points = 4 * len(answers)
    risk_score = (total_points - len(answers)) / (max_points - len(answers))
    return round(risk_score, 2)
```

**Risk Categories**:
| Score | Category | Description |
|-------|----------|-------------|
| 0.0 - 0.25 | Conservative | Capital preservation priority |
| 0.25 - 0.50 | Moderate | Balanced risk/reward |
| 0.50 - 0.75 | Growth | Accepts volatility for growth |
| 0.75 - 1.0 | Aggressive | High risk tolerance |

---

### 2. Position Sizer

**Purpose**: Calculate optimal position size based on risk tolerance

**Methods**:

#### Fixed Percentage
```python
def fixed_percentage_size(portfolio_value: float, risk_pct: float) -> float:
    """
    Simple fixed percentage of portfolio.
    
    Example: 5% per trade for conservative, 15% for aggressive
    """
    return portfolio_value * risk_pct
```

#### Kelly Criterion
```python
def kelly_criterion_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    portfolio_value: float,
    risk_tolerance: float
) -> float:
    """
    Kelly Criterion for optimal position sizing.
    
    f* = (bp - q) / b
    where:
        b = odds received on the bet (avg_win / avg_loss)
        p = probability of winning
        q = probability of losing (1 - p)
    
    Adjusted by risk_tolerance factor (0.5 Kelly is common for safety)
    """
    if avg_loss == 0:
        return 0
    
    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp to [0, 1]
    
    # Apply risk tolerance modifier (conservative = 0.25x, aggressive = 1.0x)
    adjusted_fraction = kelly_fraction * (0.25 + 0.75 * risk_tolerance)
    
    return portfolio_value * adjusted_fraction
```

#### Volatility-Adjusted
```python
def volatility_adjusted_size(
    portfolio_value: float,
    atr: float,
    current_price: float,
    risk_tolerance: float
) -> float:
    """
    Adjust position size based on market volatility.
    
    Higher volatility = smaller position
    """
    atr_pct = atr / current_price
    base_size = portfolio_value * 0.1  # 10% base
    
    # Inverse relationship with volatility
    vol_multiplier = 1 / (1 + atr_pct * 10)
    
    # Apply risk tolerance
    size = base_size * vol_multiplier * (0.5 + risk_tolerance * 0.5)
    
    return min(size, portfolio_value * 0.25)  # Cap at 25%
```

---

### 3. Break-Even Tracker

**Purpose**: Track psychological price levels and P&L status

**Features**:
- Entry price tracking (with averaging for multiple entries)
- Real-time unrealized P&L
- Break-even distance alerts
- Psychological level notifications

```python
class BreakevenTracker:
    def __init__(self):
        self.positions = {}  # symbol -> PositionInfo
    
    def update_position(self, symbol: str, action: str, price: float, quantity: float):
        if symbol not in self.positions:
            self.positions[symbol] = PositionInfo(symbol)
        
        pos = self.positions[symbol]
        
        if action == "BUY":
            # Weighted average entry price
            total_cost = pos.avg_entry * pos.quantity + price * quantity
            pos.quantity += quantity
            pos.avg_entry = total_cost / pos.quantity
        
        elif action == "SELL":
            pos.quantity -= quantity
            if pos.quantity <= 0:
                pos.reset()
    
    def get_pnl(self, symbol: str, current_price: float) -> dict:
        pos = self.positions.get(symbol)
        if not pos or pos.quantity == 0:
            return {"pnl_pct": 0, "pnl_abs": 0, "distance_to_breakeven": 0}
        
        pnl_abs = (current_price - pos.avg_entry) * pos.quantity
        pnl_pct = (current_price - pos.avg_entry) / pos.avg_entry * 100
        distance = (current_price - pos.avg_entry) / pos.avg_entry * 100
        
        return {
            "pnl_pct": round(pnl_pct, 2),
            "pnl_abs": round(pnl_abs, 2),
            "breakeven_price": pos.avg_entry,
            "distance_to_breakeven": round(distance, 2)
        }
```

---

### 4. Timeframe Analyzer

**Purpose**: Adjust signals based on preferred trading timeframe

| Timeframe | Signal Adjustment |
|-----------|-------------------|
| Intraday | Higher weight on momentum indicators |
| Swing (days) | Balance of trend + momentum |
| Position (weeks) | Higher weight on trend indicators |
| Long-term (months) | Fundamentals + macro sentiment |

```python
def adjust_state_for_timeframe(state: dict, timeframe: str) -> dict:
    """
    Re-weight state features based on trader's preferred timeframe.
    """
    weights = {
        "intraday": {"momentum": 1.5, "trend": 0.7, "sentiment": 1.0},
        "swing": {"momentum": 1.0, "trend": 1.0, "sentiment": 1.0},
        "position": {"momentum": 0.7, "trend": 1.5, "sentiment": 0.8},
        "longterm": {"momentum": 0.5, "trend": 1.2, "sentiment": 1.3},
    }
    
    w = weights.get(timeframe, weights["swing"])
    
    # Apply weights to indicator groups
    adjusted = state.copy()
    for key in ["rsi", "stochastic", "cci"]:
        if key in adjusted:
            adjusted[key] *= w["momentum"]
    for key in ["macd", "adx", "sma_trend"]:
        if key in adjusted:
            adjusted[key] *= w["trend"]
    
    return adjusted
```

---

## Integration with DRL Agent

The Trader Behavior module contributes to the state vector:

```python
trader_state = {
    "risk_tolerance": 0.65,           # From questionnaire
    "preferred_timeframe": 2,          # Encoded (0=intra, 1=swing, 2=position)
    "current_position": 100,           # Shares held
    "breakeven_price": 150.25,         # Average entry
    "unrealized_pnl_pct": 3.5,         # Current P&L %
}
```

This allows the PPO agent to learn policies that respect individual trader preferences.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `risk_profiler.py` | Questionnaire and scoring |
| `position_sizer.py` | Position sizing algorithms |
| `breakeven_tracker.py` | P&L and break-even tracking |
| `timeframe_analyzer.py` | Timeframe-based adjustments |

---

## Next Steps

- See [API Reference](05_api_reference.md) for endpoint details
- See [Deployment](06_deployment.md) for setup instructions

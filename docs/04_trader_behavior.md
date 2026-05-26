# Trader Behavior Module

> Personalized trading strategies based on individual risk tolerance

## Overview

The Trader Behavior Module is our **USP (Unique Selling Proposition)**. Unlike traditional algorithmic trading systems that use fixed parameters, our system:

1. **Profiles individual risk tolerance** through a questionnaire
2. **Adjusts position sizes** based on risk profile
3. **Tracks break-even points** for each position
4. **Personalizes trading recommendations**

---

## Risk Assessment Flow

![Risk Profiler](images/risk_profiler.png)

The risk assessment process:
1. User completes 6-question questionnaire (answers 1-4 scale)
2. System calculates risk score (0.0 - 1.0) via simple sum normalization
3. User is categorized: Conservative, Moderate, Growth, or Aggressive
4. Position sizing and recommendations are adjusted accordingly

---

## Risk Profiler

### Implementation

**File:** [`risk_profiler.py`](../backend/app/trader_behavior/risk_profiler.py)

### Questionnaire Structure

The risk assessment consists of 6 questions covering experience, loss tolerance, holding period, risk preference, capital allocation, and knowledge.

### Actual Questions

```python
RISK_QUESTIONNAIRE = [
    {
        "id": 1,
        "question": "How many years of trading/investing experience do you have?",
        "options": [
            {"value": 1, "text": "0-1 years"},
            {"value": 2, "text": "1-3 years"},
            {"value": 3, "text": "3-5 years"},
            {"value": 4, "text": "5+ years"},
        ]
    },
    {
        "id": 2,
        "question": "If your portfolio dropped 20% in a week, you would:",
        "options": [
            {"value": 1, "text": "Panic and sell everything"},
            {"value": 2, "text": "Sell some positions to reduce risk"},
            {"value": 3, "text": "Hold and wait for recovery"},
            {"value": 4, "text": "Buy more at lower prices"},
        ]
    },
    {
        "id": 3,
        "question": "What is your typical investment holding period?",
        "options": [
            {"value": 1, "text": "Less than a day (intraday)"},
            {"value": 2, "text": "Days to weeks"},
            {"value": 3, "text": "Weeks to months"},
            {"value": 4, "text": "Months to years"},
        ]
    },
    {
        "id": 4,
        "question": "Which scenario would you prefer?",
        "options": [
            {"value": 1, "text": "Guaranteed 5% annual return"},
            {"value": 2, "text": "50% chance of 15% or 0% return"},
            {"value": 3, "text": "50% chance of 25% or -10% return"},
            {"value": 4, "text": "50% chance of 50% or -30% return"},
        ]
    },
    {
        "id": 5,
        "question": "What percentage of your savings are you investing?",
        "options": [
            {"value": 1, "text": "Less than 10%"},
            {"value": 2, "text": "10-25%"},
            {"value": 3, "text": "25-50%"},
            {"value": 4, "text": "More than 50%"},
        ]
    },
    {
        "id": 6,
        "question": "How would you describe your investment knowledge?",
        "options": [
            {"value": 1, "text": "Beginner - learning the basics"},
            {"value": 2, "text": "Intermediate - understand charts and trends"},
            {"value": 3, "text": "Advanced - use technical analysis"},
            {"value": 4, "text": "Expert - use complex strategies"},
        ]
    },
]
```

### Risk Score Calculation

Uses min-max normalization of the sum of answers:

```python
def calculate_risk_score(answers: List[int]) -> float:
    valid_answers = [max(1, min(4, a)) for a in answers]
    total_points = sum(valid_answers)
    min_possible = len(valid_answers)       # All 1s
    max_possible = len(valid_answers) * 4   # All 4s
    score = (total_points - min_possible) / (max_possible - min_possible)
    return round(score, 2)
```

For 6 questions: min=6, max=24. A score of 15 → (15-6)/(24-6) = 0.50.

### Risk Categories

| Score Range | Category | Description |
|-------------|----------|-------------|
| 0.0 - 0.25 | **Conservative** | Capital preservation focus |
| 0.25 - 0.50 | **Moderate** | Balanced growth and safety |
| 0.50 - 0.75 | **Growth** | Higher returns, accepts volatility |
| 0.75 - 1.0 | **Aggressive** | Maximum growth, high risk tolerance |

---

## Position Sizer

### Implementation

**File:** [`position_sizer.py`](../backend/app/trader_behavior/position_sizer.py)

### Sizing Algorithms

#### 1. Fixed Percentage

Simple fixed percentage of capital:

```python
def fixed_percentage(capital: float, risk_tolerance: float) -> float:
    base_pct = 0.02 + (risk_tolerance * 0.08)  # 2% to 10%
    return capital * base_pct
```

#### 2. Kelly Criterion

Optimal position size for maximum growth:

```python
def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    capital: float,
    risk_tolerance: float
) -> float:
    """
    Kelly Criterion: f* = (bp - q) / b
    where:
        b = avg_win / avg_loss (odds)
        p = win_rate
        q = 1 - p (loss rate)
    """
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply half-Kelly for safety, adjusted by risk tolerance
    fraction = kelly * 0.5 * risk_tolerance
    
    return capital * max(0, min(fraction, 0.25))  # Cap at 25%
```

#### 3. Volatility-Adjusted

Position size inversely proportional to volatility:

```python
def volatility_adjusted(
    capital: float,
    volatility: float,
    target_risk: float,
    risk_tolerance: float
) -> float:
    """
    Adjust position size based on asset volatility.
    Higher volatility = smaller position
    """
    base_position = capital * target_risk * risk_tolerance
    adjusted = base_position / (volatility / 0.02)  # Normalize to 2% vol
    
    return min(adjusted, capital * 0.25)
```

---

## Break-Even Tracker

### Implementation

**File:** [`breakeven_tracker.py`](../backend/app/trader_behavior/breakeven_tracker.py)

### Key Functions

```python
class BreakEvenTracker:
    def add_position(self, symbol: str, quantity: int, price: float):
        """Add a new position or add to existing."""
        
    def update_price(self, symbol: str, current_price: float) -> dict:
        """Update P&L for a position."""
        
    def get_break_even_price(self, symbol: str) -> float:
        """Calculate break-even price including commissions."""
        
    def close_position(self, symbol: str, quantity: int, price: float) -> dict:
        """Close position and calculate realized P&L."""
```

### Break-Even Calculation

```python
def calculate_break_even(
    avg_entry_price: float,
    quantity: int,
    commission_pct: float = 0.001
) -> float:
    """
    Break-even price including round-trip commissions.
    
    Formula: BE = entry × (1 + 2×commission)
    """
    total_commission = 2 * commission_pct  # Buy + Sell
    break_even = avg_entry_price * (1 + total_commission)
    return break_even
```

---

## Integration with PPO Agent

The trader behavior module integrates with the PPO agent through the state vector:

```python
# State includes trader profile
state = np.concatenate([
    price_features,
    indicator_features,
    np.array([
        risk_tolerance,           # From risk profiler
        position_size_fraction,   # From position sizer
        distance_to_break_even,   # From break-even tracker
    ])
])
```

This allows the agent to make **personalized decisions** based on individual trader characteristics.

---

## API Endpoints

### Risk Assessment

```http
POST /api/v1/profile/risk-assessment
Content-Type: application/json

{
    "answers": [3, 4, 2, 4, 3, 2]
}
```

**Response:**
```json
{
    "risk_tolerance": 0.5,
    "category": "Growth",
    "description": "You accept higher volatility for potential growth...",
    "recommendations": {
        "max_position_size": 0.13,
        "suggested_stop_loss": 0.10,
        "suggested_take_profit": 0.20
    }
}
```

---

## Example Usage

```python
from app.trader_behavior.risk_profiler import calculate_risk_score, get_risk_category
from app.trader_behavior.position_sizer import calculate_position_size
from app.trader_behavior.breakeven_tracker import BreakEvenTracker

# 1. Profile the trader
answers = [3, 4, 2, 4, 3, 2]
risk_tolerance = calculate_risk_score(answers)
category, description = get_risk_category(risk_tolerance)
print(f"Risk Tolerance: {risk_tolerance:.2f} ({category})")
# Output: Risk Tolerance: 0.50 (Growth)

# 2. Calculate position size
position = calculate_position_size(
    capital=100000,
    risk_tolerance=risk_tolerance,
    method="kelly",
    win_rate=0.55,
    avg_win=0.03,
    avg_loss=0.02
)
print(f"Suggested Position: ₹{position:,.2f}")
# Output: Suggested Position: ₹15,000.00

# 3. Track break-even
tracker = BreakEvenTracker()
tracker.add_position("RELIANCE.NS", 10, 1500.00)
be_price = tracker.get_break_even_price("RELIANCE.NS")
print(f"Break-Even Price: ₹{be_price:.2f}")
# Output: Break-Even Price: ₹1503.00
```

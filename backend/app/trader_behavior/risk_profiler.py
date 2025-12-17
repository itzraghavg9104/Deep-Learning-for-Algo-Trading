"""
Risk profiler for trader behavior modeling.

Assesses trader's risk tolerance through questionnaire scoring.
"""
from typing import List, Tuple


def calculate_risk_score(answers: List[int]) -> float:
    """
    Calculate normalized risk tolerance score from questionnaire answers.
    
    Args:
        answers: List of answer scores (each 1-4, where 4 is most aggressive)
    
    Returns:
        Risk tolerance score (0.0 = conservative, 1.0 = aggressive)
    """
    if not answers:
        return 0.5  # Default moderate
    
    # Validate answers
    valid_answers = [max(1, min(4, a)) for a in answers]
    
    total_points = sum(valid_answers)
    min_possible = len(valid_answers)  # All 1s
    max_possible = len(valid_answers) * 4  # All 4s
    
    # Normalize to 0-1 range
    score = (total_points - min_possible) / (max_possible - min_possible)
    
    return round(score, 2)


def get_risk_category(risk_tolerance: float) -> Tuple[str, str]:
    """
    Get risk category and description from tolerance score.
    
    Args:
        risk_tolerance: Score from 0.0 to 1.0
    
    Returns:
        Tuple of (category_name, description)
    """
    if risk_tolerance < 0.25:
        return (
            "Conservative",
            "You prioritize capital preservation over growth. Suitable for "
            "low-volatility investments with steady returns."
        )
    elif risk_tolerance < 0.50:
        return (
            "Moderate",
            "You seek a balance between risk and reward. Comfortable with "
            "some volatility for better returns."
        )
    elif risk_tolerance < 0.75:
        return (
            "Growth",
            "You accept higher volatility for potential growth. Willing to "
            "take calculated risks for better long-term returns."
        )
    else:
        return (
            "Aggressive",
            "You have high risk tolerance and seek maximum returns. "
            "Comfortable with significant volatility and potential drawdowns."
        )


def get_position_size_multiplier(risk_tolerance: float) -> float:
    """
    Get position size multiplier based on risk tolerance.
    
    Args:
        risk_tolerance: Score from 0.0 to 1.0
    
    Returns:
        Multiplier for position sizing (0.5 to 1.5)
    """
    # Conservative: 0.5x, Aggressive: 1.5x
    return 0.5 + risk_tolerance


def get_stop_loss_percentage(risk_tolerance: float) -> float:
    """
    Get suggested stop-loss percentage based on risk tolerance.
    
    Args:
        risk_tolerance: Score from 0.0 to 1.0
    
    Returns:
        Stop-loss percentage (5% to 15%)
    """
    # Conservative: 5%, Aggressive: 15%
    return 0.05 + (risk_tolerance * 0.10)


def get_take_profit_percentage(risk_tolerance: float) -> float:
    """
    Get suggested take-profit percentage based on risk tolerance.
    
    Args:
        risk_tolerance: Score from 0.0 to 1.0
    
    Returns:
        Take-profit percentage (10% to 30%)
    """
    # Conservative: 10%, Aggressive: 30%
    return 0.10 + (risk_tolerance * 0.20)


# Risk assessment questionnaire
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


def get_questionnaire() -> List[dict]:
    """Get the risk assessment questionnaire."""
    return RISK_QUESTIONNAIRE

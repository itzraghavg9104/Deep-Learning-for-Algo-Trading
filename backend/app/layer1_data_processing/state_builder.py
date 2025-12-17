"""
State builder for DRL agent.

Combines all Layer 1 outputs into a unified state vector for the PPO agent.
"""
import numpy as np
from typing import Dict, Any, Optional
import pandas as pd

from app.layer1_data_processing.technical_indicators import compute_indicators


def build_state(
    market_data: pd.DataFrame,
    trader_profile: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
    sentiment: Optional[float] = None,
    use_sentiment: bool = False
) -> np.ndarray:
    """
    Build state vector for DRL agent.
    
    Combines:
    - DeepAR predictions (price_mean, price_std, change_pct)
    - Technical indicators (30+)
    - Trader behavior (risk_tolerance, timeframe, position, breakeven, pnl)
    - Optional sentiment
    
    Args:
        market_data: OHLCV DataFrame
        trader_profile: User's risk profile and preferences
        portfolio_state: Current holdings and P&L
        prediction: DeepAR model predictions (optional)
        sentiment: FinBERT sentiment score (optional)
        use_sentiment: Whether to include sentiment
    
    Returns:
        Normalized state vector as numpy array
    """
    state_dict = {}
    
    # === PRICE DATA ===
    if market_data is not None and not market_data.empty:
        current_price = market_data['Close'].iloc[-1]
        prev_price = market_data['Close'].iloc[-2] if len(market_data) > 1 else current_price
        
        state_dict['current_price'] = current_price
        state_dict['price_change_pct'] = (current_price - prev_price) / prev_price * 100
    
    # === DEEPAR PREDICTIONS ===
    if prediction:
        state_dict['pred_price_mean'] = prediction.get('price_mean', 0)
        state_dict['pred_price_std'] = prediction.get('price_std', 0)
        state_dict['pred_change_pct'] = prediction.get('change_pct', 0)
        state_dict['pred_confidence'] = prediction.get('confidence', 0.5)
    else:
        # Placeholder when model not yet trained
        state_dict['pred_price_mean'] = state_dict.get('current_price', 0)
        state_dict['pred_price_std'] = 0
        state_dict['pred_change_pct'] = 0
        state_dict['pred_confidence'] = 0.5
    
    # === TECHNICAL INDICATORS ===
    if market_data is not None and not market_data.empty:
        indicators = compute_indicators(market_data)
        
        # Extract key indicators for state
        indicator_keys = [
            'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
            'bb_pct_b', 'atr_pct', 'adx', 'stoch_k', 'stoch_d',
            'cci_20', 'williams_r', 'roc_10', 'mfi_14', 'volume_ratio'
        ]
        
        for key in indicator_keys:
            state_dict[key] = indicators.get(key, 0)
        
        # Trend signals (binary)
        state_dict['above_sma_20'] = 1.0 if indicators.get('above_sma_20', False) else 0.0
        state_dict['above_sma_50'] = 1.0 if indicators.get('above_sma_50', False) else 0.0
        state_dict['trend_bullish'] = 1.0 if indicators.get('trend') == 'bullish' else 0.0
    
    # === TRADER BEHAVIOR ===
    state_dict['risk_tolerance'] = trader_profile.get('risk_tolerance', 0.5)
    
    # Encode timeframe (0=intraday, 1=swing, 2=position, 3=longterm)
    timeframe_map = {'intraday': 0, 'swing': 1, 'position': 2, 'longterm': 3}
    state_dict['timeframe'] = timeframe_map.get(
        trader_profile.get('preferred_timeframe', 'swing'), 1
    ) / 3.0  # Normalize to 0-1
    
    # === PORTFOLIO STATE ===
    state_dict['current_position'] = portfolio_state.get('quantity', 0)
    state_dict['position_pct'] = portfolio_state.get('position_pct', 0)
    state_dict['unrealized_pnl_pct'] = portfolio_state.get('unrealized_pnl_pct', 0)
    state_dict['cash_ratio'] = portfolio_state.get('cash_ratio', 1.0)
    
    # Break-even distance
    if portfolio_state.get('has_position', False):
        breakeven = portfolio_state.get('breakeven_price', 0)
        current = state_dict.get('current_price', breakeven)
        if breakeven > 0:
            state_dict['breakeven_distance_pct'] = (current - breakeven) / breakeven * 100
        else:
            state_dict['breakeven_distance_pct'] = 0
    else:
        state_dict['breakeven_distance_pct'] = 0
    
    # === SENTIMENT (OPTIONAL) ===
    if use_sentiment and sentiment is not None:
        state_dict['sentiment'] = sentiment
    else:
        state_dict['sentiment'] = 0.0  # Neutral
    
    # === CONVERT TO ARRAY ===
    state_vector = _dict_to_normalized_array(state_dict)
    
    return state_vector


def _dict_to_normalized_array(state_dict: Dict[str, Any]) -> np.ndarray:
    """
    Convert state dictionary to normalized numpy array.
    
    Args:
        state_dict: State dictionary
    
    Returns:
        Normalized array
    """
    # Define expected keys and normalization ranges
    normalization_config = {
        # Price (log-normalized)
        'current_price': ('log', 100, 10000),
        'pred_price_mean': ('log', 100, 10000),
        'pred_price_std': ('linear', 0, 100),
        
        # Percentages (-10 to 10 typical)
        'price_change_pct': ('clip', -10, 10),
        'pred_change_pct': ('clip', -10, 10),
        'unrealized_pnl_pct': ('clip', -50, 50),
        'breakeven_distance_pct': ('clip', -20, 20),
        
        # RSI-like (0-100)
        'rsi_14': ('linear', 0, 100),
        'stoch_k': ('linear', 0, 100),
        'stoch_d': ('linear', 0, 100),
        'mfi_14': ('linear', 0, 100),
        
        # CCI (-200 to 200)
        'cci_20': ('clip', -200, 200),
        
        # Williams %R (-100 to 0)
        'williams_r': ('linear', -100, 0),
        
        # MACD (small values)
        'macd_line': ('clip', -5, 5),
        'macd_signal': ('clip', -5, 5),
        'macd_histogram': ('clip', -2, 2),
        
        # Bollinger %B (0-1 typical, can exceed)
        'bb_pct_b': ('clip', -0.5, 1.5),
        
        # ATR % (0-5% typical)
        'atr_pct': ('clip', 0, 10),
        
        # ADX (0-100)
        'adx': ('linear', 0, 100),
        
        # ROC (-20 to 20)
        'roc_10': ('clip', -20, 20),
        
        # Volume ratio (0.5 to 3 typical)
        'volume_ratio': ('clip', 0, 5),
        
        # Binary/normalized (0-1)
        'above_sma_20': ('none', 0, 1),
        'above_sma_50': ('none', 0, 1),
        'trend_bullish': ('none', 0, 1),
        'risk_tolerance': ('none', 0, 1),
        'timeframe': ('none', 0, 1),
        'cash_ratio': ('none', 0, 1),
        'pred_confidence': ('none', 0, 1),
        
        # Position (can be large)
        'current_position': ('none', 0, 1000),
        'position_pct': ('none', 0, 100),
        
        # Sentiment (-1 to 1)
        'sentiment': ('none', -1, 1),
    }
    
    values = []
    for key, (method, min_val, max_val) in normalization_config.items():
        val = state_dict.get(key, 0)
        
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0
        
        if method == 'log' and val > 0:
            val = np.log(val) / np.log(max_val)
        elif method == 'clip':
            val = np.clip(val, min_val, max_val)
            val = (val - min_val) / (max_val - min_val) * 2 - 1  # Normalize to -1 to 1
        elif method == 'linear':
            val = (val - min_val) / (max_val - min_val)  # Normalize to 0-1
        # 'none' keeps value as-is (already normalized)
        
        values.append(float(val))
    
    return np.array(values, dtype=np.float32)


def get_state_dim() -> int:
    """Get the dimension of the state vector."""
    return 30  # Approximate, based on normalization config

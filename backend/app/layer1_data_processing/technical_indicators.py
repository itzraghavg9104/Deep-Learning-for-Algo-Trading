"""
Technical indicators computation (30+ indicators).

Uses pandas-ta for indicator calculations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas-ta not installed. Using basic indicators only.")


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute 30+ technical indicators from OHLCV data.
    
    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns
    
    Returns:
        Dictionary of indicator values (latest values)
    """
    if df is None or df.empty:
        return {}
    
    indicators = {}
    
    try:
        # Ensure column names are correct
        df.columns = [col.capitalize() for col in df.columns]
        
        if PANDAS_TA_AVAILABLE:
            indicators = _compute_pandas_ta_indicators(df)
        else:
            indicators = _compute_basic_indicators(df)
            
    except Exception as e:
        print(f"Error computing indicators: {e}")
        indicators = {"error": str(e)}
    
    return indicators


def _compute_pandas_ta_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute indicators using pandas-ta library.
    """
    indicators = {}
    
    # === TREND INDICATORS ===
    
    # Moving Averages
    sma_20 = ta.sma(df['Close'], length=20)
    sma_50 = ta.sma(df['Close'], length=50)
    sma_200 = ta.sma(df['Close'], length=200)
    ema_12 = ta.ema(df['Close'], length=12)
    ema_26 = ta.ema(df['Close'], length=26)
    
    if sma_20 is not None and len(sma_20) > 0:
        indicators['sma_20'] = round(float(sma_20.iloc[-1]), 2)
    if sma_50 is not None and len(sma_50) > 0:
        indicators['sma_50'] = round(float(sma_50.iloc[-1]), 2)
    if sma_200 is not None and len(sma_200) > 0:
        indicators['sma_200'] = round(float(sma_200.iloc[-1]), 2)
    if ema_12 is not None and len(ema_12) > 0:
        indicators['ema_12'] = round(float(ema_12.iloc[-1]), 2)
    if ema_26 is not None and len(ema_26) > 0:
        indicators['ema_26'] = round(float(ema_26.iloc[-1]), 2)
    
    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None and len(macd) > 0:
        indicators['macd_line'] = round(float(macd.iloc[-1, 0]), 4)
        indicators['macd_signal'] = round(float(macd.iloc[-1, 2]), 4)
        indicators['macd_histogram'] = round(float(macd.iloc[-1, 1]), 4)
    
    # ADX
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None and len(adx) > 0:
        indicators['adx'] = round(float(adx.iloc[-1, 0]), 2)
    
    # === MOMENTUM INDICATORS ===
    
    # RSI
    rsi = ta.rsi(df['Close'], length=14)
    if rsi is not None and len(rsi) > 0:
        indicators['rsi_14'] = round(float(rsi.iloc[-1]), 2)
    
    # Stochastic
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    if stoch is not None and len(stoch) > 0:
        indicators['stoch_k'] = round(float(stoch.iloc[-1, 0]), 2)
        indicators['stoch_d'] = round(float(stoch.iloc[-1, 1]), 2)
    
    # CCI
    cci = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    if cci is not None and len(cci) > 0:
        indicators['cci_20'] = round(float(cci.iloc[-1]), 2)
    
    # Williams %R
    willr = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    if willr is not None and len(willr) > 0:
        indicators['williams_r'] = round(float(willr.iloc[-1]), 2)
    
    # ROC
    roc = ta.roc(df['Close'], length=10)
    if roc is not None and len(roc) > 0:
        indicators['roc_10'] = round(float(roc.iloc[-1]), 2)
    
    # === VOLATILITY INDICATORS ===
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None and len(bbands) > 0:
        indicators['bb_upper'] = round(float(bbands.iloc[-1, 0]), 2)
        indicators['bb_middle'] = round(float(bbands.iloc[-1, 1]), 2)
        indicators['bb_lower'] = round(float(bbands.iloc[-1, 2]), 2)
        # %B position
        current_price = df['Close'].iloc[-1]
        bb_range = bbands.iloc[-1, 0] - bbands.iloc[-1, 2]
        if bb_range != 0:
            indicators['bb_pct_b'] = round((current_price - bbands.iloc[-1, 2]) / bb_range, 2)
    
    # ATR
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    if atr is not None and len(atr) > 0:
        indicators['atr_14'] = round(float(atr.iloc[-1]), 2)
        # Normalized ATR (as % of price)
        indicators['atr_pct'] = round(float(atr.iloc[-1]) / df['Close'].iloc[-1] * 100, 2)
    
    # === VOLUME INDICATORS ===
    
    # OBV
    obv = ta.obv(df['Close'], df['Volume'])
    if obv is not None and len(obv) > 0:
        indicators['obv'] = float(obv.iloc[-1])
    
    # MFI
    mfi = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    if mfi is not None and len(mfi) > 0:
        indicators['mfi_14'] = round(float(mfi.iloc[-1]), 2)
    
    # Volume SMA
    vol_sma = ta.sma(df['Volume'], length=20)
    if vol_sma is not None and len(vol_sma) > 0:
        indicators['volume_sma_20'] = float(vol_sma.iloc[-1])
        indicators['volume_ratio'] = round(df['Volume'].iloc[-1] / vol_sma.iloc[-1], 2)
    
    # === TREND SIGNALS ===
    
    # Price vs SMAs
    current_price = df['Close'].iloc[-1]
    if 'sma_20' in indicators:
        indicators['above_sma_20'] = current_price > indicators['sma_20']
    if 'sma_50' in indicators:
        indicators['above_sma_50'] = current_price > indicators['sma_50']
    if 'sma_200' in indicators:
        indicators['above_sma_200'] = current_price > indicators['sma_200']
    
    # Overall trend
    if 'sma_20' in indicators and 'sma_50' in indicators:
        if indicators['sma_20'] > indicators['sma_50']:
            indicators['trend'] = 'bullish'
        elif indicators['sma_20'] < indicators['sma_50']:
            indicators['trend'] = 'bearish'
        else:
            indicators['trend'] = 'neutral'
    
    return indicators


def _compute_basic_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic indicators without pandas-ta (fallback).
    """
    indicators = {}
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Simple Moving Averages
    indicators['sma_20'] = round(float(close.rolling(20).mean().iloc[-1]), 2)
    indicators['sma_50'] = round(float(close.rolling(50).mean().iloc[-1]), 2)
    
    # EMA
    indicators['ema_12'] = round(float(close.ewm(span=12).mean().iloc[-1]), 2)
    indicators['ema_26'] = round(float(close.ewm(span=26).mean().iloc[-1]), 2)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    indicators['rsi_14'] = round(float(rsi.iloc[-1]), 2)
    
    # Bollinger Bands
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    indicators['bb_upper'] = round(float((sma + 2 * std).iloc[-1]), 2)
    indicators['bb_lower'] = round(float((sma - 2 * std).iloc[-1]), 2)
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    indicators['macd_line'] = round(float(macd_line.iloc[-1]), 4)
    indicators['macd_signal'] = round(float(signal_line.iloc[-1]), 4)
    
    # Volume ratio
    vol_sma = volume.rolling(20).mean()
    indicators['volume_ratio'] = round(float(volume.iloc[-1] / vol_sma.iloc[-1]), 2)
    
    return indicators


def get_indicator_summary(indicators: Dict[str, Any]) -> Dict[str, str]:
    """
    Get human-readable summary of indicators.
    
    Returns:
        Dictionary with signal interpretations
    """
    summary = {}
    
    # RSI interpretation
    if 'rsi_14' in indicators:
        rsi = indicators['rsi_14']
        if rsi > 70:
            summary['rsi'] = 'overbought'
        elif rsi < 30:
            summary['rsi'] = 'oversold'
        else:
            summary['rsi'] = 'neutral'
    
    # MACD interpretation
    if 'macd_line' in indicators and 'macd_signal' in indicators:
        if indicators['macd_line'] > indicators['macd_signal']:
            summary['macd'] = 'bullish'
        else:
            summary['macd'] = 'bearish'
    
    # Trend interpretation
    if 'trend' in indicators:
        summary['trend'] = indicators['trend']
    
    # Bollinger interpretation
    if 'bb_pct_b' in indicators:
        pct_b = indicators['bb_pct_b']
        if pct_b > 1:
            summary['bollinger'] = 'above_upper'
        elif pct_b < 0:
            summary['bollinger'] = 'below_lower'
        else:
            summary['bollinger'] = 'within_bands'
    
    return summary

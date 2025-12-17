"""
Market data fetching for Indian markets (NSE/BSE).

Uses yfinance with .NS (NSE) and .BO (BSE) suffixes.
"""
import yfinance as yf
import pandas as pd
from typing import Optional
import asyncio
from functools import lru_cache


# Popular NSE stocks
NSE_STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "AXISBANK": "AXISBANK.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "MARUTI": "MARUTI.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
}


def normalize_symbol(symbol: str) -> str:
    """
    Normalize stock symbol to include exchange suffix.
    
    Args:
        symbol: Stock symbol (with or without suffix)
    
    Returns:
        Symbol with .NS suffix if no suffix present
    """
    symbol = symbol.upper().strip()
    
    # Already has suffix
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    
    # Check if it's a known NSE stock
    if symbol in NSE_STOCKS:
        return NSE_STOCKS[symbol]
    
    # Default to NSE
    return f"{symbol}.NS"


def fetch_market_data_sync(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Fetch market data synchronously using yfinance.
    
    Args:
        symbol: Stock symbol (e.g., RELIANCE.NS)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
    
    Returns:
        DataFrame with OHLCV data or None if error
    """
    try:
        symbol = normalize_symbol(symbol)
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            return None
        
        # Standardize column names
        data.columns = [col.capitalize() for col in data.columns]
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


async def get_market_data(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Async wrapper for fetching market data.
    
    Args:
        symbol: Stock symbol
        period: Data period
        interval: Data interval
    
    Returns:
        DataFrame with OHLCV data
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        fetch_market_data_sync,
        symbol,
        period,
        interval
    )


def get_stock_info(symbol: str) -> dict:
    """
    Get stock information.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dictionary with stock info
    """
    try:
        symbol = normalize_symbol(symbol)
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            "symbol": symbol,
            "name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }
        
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def get_nifty50_symbols() -> list:
    """
    Get list of NIFTY 50 stock symbols.
    
    Returns:
        List of .NS suffixed symbols
    """
    return list(NSE_STOCKS.values())

"""
Data download script for NIFTY 50 stocks.

Downloads historical data using yfinance for model training.
"""
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional


# NIFTY 50 constituent stocks (top 20 for training)
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "HINDUNILVR.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "ASIANPAINT.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "TATAMOTORS.NS",
]


def download_stock_data(
    symbols: List[str],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d",
    output_dir: str = "./data"
) -> pd.DataFrame:
    """
    Download historical data for multiple stocks.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (default: today)
        interval: Data interval (1d, 1h, etc.)
        output_dir: Directory to save data
    
    Returns:
        Combined DataFrame with all stocks
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if not data.empty:
                data = data.reset_index()
                data["Symbol"] = symbol
                data["time_idx"] = range(len(data))
                all_data.append(data)
                
                # Save individual file
                filename = f"{symbol.replace('.NS', '')}.csv"
                data.to_csv(os.path.join(output_dir, filename), index=False)
                print(f"  ✓ Downloaded {len(data)} rows")
            else:
                print(f"  ✗ No data found")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Combine all data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(output_dir, "nifty50_combined.csv"), index=False)
        print(f"\nTotal: {len(combined)} rows saved to {output_dir}/nifty50_combined.csv")
        return combined
    
    return pd.DataFrame()


def prepare_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for DeepAR training.
    
    Args:
        data: Raw OHLCV data
    
    Returns:
        Prepared DataFrame with required columns
    """
    # Standardize column names
    if "Date" in data.columns:
        data["date"] = pd.to_datetime(data["Date"])
    elif "Datetime" in data.columns:
        data["date"] = pd.to_datetime(data["Datetime"])
    
    # Select and rename columns
    prepared = data[["date", "Symbol", "Open", "High", "Low", "Close", "Volume"]].copy()
    prepared.columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
    
    # Add time index for each symbol
    prepared = prepared.sort_values(["symbol", "date"])
    prepared["time_idx"] = prepared.groupby("symbol").cumcount()
    
    # Add price change features
    prepared["returns"] = prepared.groupby("symbol")["close"].pct_change()
    prepared["log_returns"] = prepared.groupby("symbol")["close"].transform(
        lambda x: (x / x.shift(1)).apply(lambda y: 0 if pd.isna(y) else y)
    )
    
    # Drop NA
    prepared = prepared.dropna()
    
    return prepared


if __name__ == "__main__":
    print("=" * 50)
    print("NIFTY 50 Data Downloader")
    print("=" * 50)
    
    # Download 5 years of data
    data = download_stock_data(
        symbols=NIFTY_50_SYMBOLS,
        start_date="2020-01-01",
        output_dir="./data/raw"
    )
    
    if not data.empty:
        # Prepare for training
        prepared = prepare_training_data(data)
        prepared.to_csv("./data/training_data.csv", index=False)
        
        print(f"\nTraining data prepared: {len(prepared)} rows")
        print(f"Symbols: {prepared['symbol'].nunique()}")
        print(f"Date range: {prepared['date'].min()} to {prepared['date'].max()}")

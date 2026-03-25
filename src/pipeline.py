"""
pipeline.py - Data cleaning and feature engineering module
Cleans raw stock data and generates technical indicator features
"""

import pandas as pd
import numpy as np
from fetcher import load_from_db

SPLIT_THRESHOLD = 0.50  # Flag if overnight price change exceeds 50% （not normal for single stock has over 50% change overnight）
def detect_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect potential stock splits or reverse splits.

    A split causes an abnormal overnight price drop (e.g. -50% for 2:1 split).
    A reverse split causes an abnormal overnight price jump (e.g. +100% for 1:2).

    Flags rows where the absolute overnight return exceeds SPLIT_THRESHOLD.
    Since we use auto_adjust=True in yfinance, prices should already be
    split-adjusted -- but this acts as a sanity check for data integrity.
    """
    df = df.copy()

    # Calculate overnight return: close[t] vs close[t-1]
    overnight_return = df["close"].pct_change()

    # Flag suspicious rows
    df["split_flag"] = overnight_return.abs() > SPLIT_THRESHOLD

    flagged = df[df["split_flag"]]

    if flagged.empty:
        print("Split detection: no anomalies found.")
    else:
        print(f"Split detection: {len(flagged)} suspicious date(s) flagged:")
        for date, row in flagged.iterrows():
            change = overnight_return.loc[date]
            direction = "DROP" if change < 0 else "JUMP"
            print(f"  {date.date()}  {direction}  {change:+.1%}  "
                  f"close={row['close']:.2f}")
        print("  Note: yfinance auto_adjust=True should handle splits.")
        print("  These may indicate a data quality issue -- verify manually.")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw stock data:
    1. Remove duplicate rows
    2. Filter invalid prices (zero or negative)
    3. Forward-fill missing values
    """
    original_len = len(df)

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Filter out invalid prices
    df = df[(df["close"] > 0) & (df["open"] > 0)]

    # Forward-fill missing values, then drop any remaining NaNs
    df = df.ffill().dropna()

    print(f"Cleaned: {original_len} -> {len(df)} rows "
          f"(removed {original_len - len(df)} rows)")
    return df


def add_features(
    df: pd.DataFrame,
    ma_fast: int = 5,
    ma_slow: int = 20
) -> pd.DataFrame:
    """
    Feature engineering: add technical indicators with configurable MA windows.

    Args:
        df:      Cleaned stock DataFrame
        ma_fast: Fast moving average window (default 5)
        ma_slow: Slow moving average window (default 20)
    """
    if ma_fast >= ma_slow:
        raise ValueError(
            f"ma_fast ({ma_fast}) must be smaller than ma_slow ({ma_slow})"
        )

    df = df.copy()

    # Daily percentage return
    df["daily_return"] = df["close"].pct_change()

    # Configurable moving averages
    df[f"ma{ma_fast}"] = df["close"].rolling(window=ma_fast).mean()
    df[f"ma{ma_slow}"] = df["close"].rolling(window=ma_slow).mean()

    # Store window sizes as metadata for downstream modules
    df.attrs["ma_fast"] = ma_fast
    df.attrs["ma_slow"] = ma_slow

    # Rolling volatility (20-day or ma_slow window)
    df["volatility"] = df["daily_return"].rolling(window=ma_slow).std()

    # Volume moving average
    df["volume_ma"] = df["volume"].rolling(window=ma_fast).mean()

    # Drop NaNs introduced by rolling windows
    df = df.dropna()

    print(f"Features added: MA{ma_fast}, MA{ma_slow}. Final shape: {df.shape}")
    return df


def run_pipeline(
    ticker: str,
    ma_fast: int = 5,
    ma_slow: int = 20
) -> pd.DataFrame:
    """Full pipeline: load -> clean -> split detection -> feature engineering"""
    print(f"\nRunning pipeline for {ticker} (MA{ma_fast} / MA{ma_slow})...")
    df = load_from_db(ticker)
    df = clean_data(df)
    df = add_features(df, ma_fast=ma_fast, ma_slow=ma_slow)
    return df


if __name__ == "__main__":
    df = run_pipeline("AAPL")
    print("\nSample output:")
    print(df[["close", "daily_return", "ma5", "ma20", "volatility_20"]].tail(10))
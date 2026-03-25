"""
fetcher.py - Data ingestion module
Downloads historical stock data from Yahoo Finance and persists it to a local SQLite database.
"""

import yfinance as yf
import pandas as pd
import sqlite3

DB_PATH = "stock_data.db"


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock symbol, e.g. "AAPL"
        start:  Start date, e.g. "2020-01-01"
        end:    End date,   e.g. "2024-01-01"

    Returns:
        DataFrame with columns: open, high, low, close, volume, ticker

    Raises:
        ValueError: If no data is returned for the given ticker/date range
    """
    print(f"Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten multi-level columns and normalize to lowercase
    df.columns = [col[0].lower() for col in df.columns]
    df.index.name = "date"
    df["ticker"] = ticker

    print(f"Downloaded {len(df)} rows.")
    return df


def save_to_db(df: pd.DataFrame, ticker: str) -> None:
    """
    Persist a DataFrame to the local SQLite database.
    Replaces existing data for the given ticker if present.

    Args:
        df:     DataFrame to store
        ticker: Used as the table name in the database
    """
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(ticker, conn, if_exists="replace", index=True)
    conn.close()
    print(f"Saved {ticker} to database at '{DB_PATH}'.")


def load_from_db(ticker: str) -> pd.DataFrame:
    """
    Load stock data for a given ticker from the local SQLite database.

    Args:
        ticker: Stock symbol to retrieve

    Returns:
        DataFrame indexed by date
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM '{ticker}'",
        conn,
        index_col="date",
        parse_dates=["date"]
    )
    conn.close()
    return df


def fetch_and_save(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Convenience function: fetch from Yahoo Finance and save to database in one call.

    Args:
        ticker: Stock symbol, e.g. "AAPL"
        start:  Start date, e.g. "2020-01-01"
        end:    End date,   e.g. "2024-01-01"

    Returns:
        The downloaded DataFrame
    """
    df = fetch_stock_data(ticker, start, end)
    save_to_db(df, ticker)
    return df


if __name__ == "__main__":
    df = fetch_and_save("AAPL", "2020-01-01", "2024-01-01")
    print("\nSample data:")
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
"""
backtest.py - Vectorized backtesting engine
Tests a moving average crossover strategy on historical stock data.
Uses NumPy vectorized operations instead of row-by-row loops for performance.
"""

import pandas as pd
import numpy as np
from pipeline import run_pipeline


def generate_signals(
    df: pd.DataFrame,
    operator: str = ">"
) -> pd.DataFrame:
    """
    Generate buy/sell signals based on MA crossover.

    The two MA columns are read from df.attrs (set by pipeline).
    Operator defines when to go long:
      ">"  : buy when ma_fast >  ma_slow  (default momentum strategy)
      ">=" : buy when ma_fast >= ma_slow
      "<"  : buy when ma_fast <  ma_slow  (mean-reversion strategy)
      "<=" : buy when ma_fast <= ma_slow

    Args:
        df:       DataFrame from run_pipeline()
        operator: Comparison operator string: ">", ">=", "<", "<="
    """
    VALID_OPERATORS = {">", ">=", "<", "<="}
    if operator not in VALID_OPERATORS:
        raise ValueError(
            f"Invalid operator '{operator}'. Must be one of: {VALID_OPERATORS}"
        )

    df = df.copy()

    ma_fast = df.attrs["ma_fast"]
    ma_slow = df.attrs["ma_slow"]
    fast_col = f"ma{ma_fast}"
    slow_col = f"ma{ma_slow}"

    # Apply operator to determine long/short position
    ops = {
        ">":  df[fast_col] >  df[slow_col],
        ">=": df[fast_col] >= df[slow_col],
        "<":  df[fast_col] <  df[slow_col],
        "<=": df[fast_col] <= df[slow_col],
    }
    condition = ops[operator]

    # +1 = long, -1 = out
    position = np.where(condition, 1, -1)

    # Signal fires only on crossover day
    df["signal"] = np.where(
        np.diff(position, prepend=position[0]) != 0,
        position,
        0
    )
    df["position"] = pd.Series(position, index=df.index)

    print(f"Signals generated: MA{ma_fast} {operator} MA{ma_slow}  "
          f"| Buys: {(df['signal']==1).sum()}  "
          f"Sells: {(df['signal']==-1).sum()}")
    return df


def run_backtest(df: pd.DataFrame, initial_capital: float = 100_000.0) -> tuple:
    """
    Run backtest and compute performance metrics.

    Returns:
        Tuple of (metrics dict, enriched DataFrame with portfolio columns)
    """
    df = df.copy()

    # Strategy daily return
    df["strategy_return"] = df["position"].shift(1) * df["daily_return"]

    # Cumulative returns
    df["cumulative_market"]   = (1 + df["daily_return"]).cumprod()
    df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()

    # Portfolio value
    df["portfolio_value"] = initial_capital * df["cumulative_strategy"]

    # Performance metrics
    total_return  = df["cumulative_strategy"].iloc[-1] - 1
    market_return = df["cumulative_market"].iloc[-1] - 1

    sharpe = (
        df["strategy_return"].mean() /
        df["strategy_return"].std() *
        np.sqrt(252)
    )

    rolling_max  = df["portfolio_value"].cummax()
    drawdown     = (df["portfolio_value"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    num_trades = (df["signal"] != 0).sum()
    win_rate   = (df["strategy_return"] > 0).sum() / (df["strategy_return"] != 0).sum()

    metrics = {
        "total_return":    round(total_return * 100, 2),
        "market_return":   round(market_return * 100, 2),
        "sharpe_ratio":    round(sharpe, 4),
        "max_drawdown":    round(max_drawdown * 100, 2),
        "num_trades":      int(num_trades),
        "win_rate":        round(win_rate * 100, 2),
        "final_value":     round(df["portfolio_value"].iloc[-1], 2),
        "initial_capital": initial_capital,
    }

    return metrics, df


def run_full_backtest(
    ticker: str,
    start: str,
    end: str,
    ma_fast: int = 5,
    ma_slow: int = 20,
    operator: str = ">"
) -> tuple:
    """
    End-to-end backtest with configurable MA windows and signal operator.

    Args:
        ticker:   Stock symbol, e.g. "AAPL"
        start:    Start date, e.g. "2020-01-01"
        end:      End date,   e.g. "2024-01-01"
        ma_fast:  Fast MA window (default 5)
        ma_slow:  Slow MA window (default 20)
        operator: Signal condition: ">", ">=", "<", "<="

    Returns:
        Tuple of (metrics dict, chart file path)
    """
    from visualizer import plot_signals

    df          = run_pipeline(ticker, ma_fast=ma_fast, ma_slow=ma_slow)
    df          = generate_signals(df, operator=operator)
    metrics, df = run_backtest(df)
    chart_path  = plot_signals(df, ticker, ma_fast, ma_slow, operator)

    return metrics, chart_path


if __name__ == "__main__":
    # Test with custom parameters
    results, chart = run_full_backtest(
        "AAPL", "2020-01-01", "2024-01-01",
        ma_fast=10, ma_slow=50, operator=">"
    )

    print("\n===== Backtest Results: AAPL MA10 > MA50 =====")
    print(f"  Initial Capital : ${results['initial_capital']:,.0f}")
    print(f"  Final Value     : ${results['final_value']:,.2f}")
    print(f"  Total Return    : {results['total_return']}%")
    print(f"  Market Return   : {results['market_return']}%")
    print(f"  Sharpe Ratio    : {results['sharpe_ratio']}")
    print(f"  Max Drawdown    : {results['max_drawdown']}%")
    print(f"  Number of Trades: {results['num_trades']}")
    print(f"  Win Rate        : {results['win_rate']}%")
    print(f"\n  Chart saved to  : {chart}")



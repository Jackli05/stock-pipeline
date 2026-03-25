"""
visualizer.py - Chart generation module
Generates a buy/sell signal chart for a given backtest result.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

def plot_signals(
    df: pd.DataFrame,
    ticker: str,
    ma_fast: int,
    ma_slow: int,
    operator: str) -> str:
    """
    Generate a price chart with MA lines and buy/sell markers.

    Args:
        df:     DataFrame with columns: close, ma5, ma20, signal, position
        ticker: Stock symbol, used in title and filename

    Returns:
        File path of the saved chart image
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.patch.set_facecolor("#0f0f0f")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # --- Top panel: price + MAs + signals ---
    ax1.plot(df.index, df["close"], color="#a8dadc", linewidth=1.2,
             label="Close Price", zorder=2)
    ax1.plot(df.index, df[f"ma{ma_fast}"],  color="#e9c46a", linewidth=1.0,
             linestyle="--", label=f"ma{ma_fast}",  zorder=2)
    ax1.plot(df.index, df[f"ma{ma_slow}"], color="#f4a261", linewidth=1.0,
             linestyle="--", label=f"ma{ma_slow}", zorder=2)

    # Buy signals (MA5 crosses above MA20)
    buys  = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]

    ax1.scatter(buys.index,  buys["close"],  marker="^", color="#2ecc71",
                s=80, zorder=5, label="Buy")
    ax1.scatter(sells.index, sells["close"], marker="v", color="#e74c3c",
                s=80, zorder=5, label="Sell")

    # Shade background by position: green when long, red when short/cash
    for i in range(1, len(df)):
        color = "#1a3a2a" if df["position"].iloc[i] == 1 else "#3a1a1a"
        ax1.axvspan(df.index[i - 1], df.index[i],
                    alpha=0.3, color=color, linewidth=0)

    ax1.set_title(
    f"{ticker} — MA{ma_fast} {operator} MA{ma_slow} Strategy",
    fontsize=14, pad=10
)    
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", facecolor="#1a1a2e",
               labelcolor="white", framealpha=0.8)
    ax1.grid(axis="y", color="#333333", linestyle="--", linewidth=0.5)

    # --- Bottom panel: cumulative strategy vs market ---
    ax2.plot(df.index, df["cumulative_strategy"] * 100 - 100,
             color="#a8dadc", linewidth=1.2, label="Strategy Return %")
    ax2.plot(df.index, df["cumulative_market"] * 100 - 100,
             color="#e9c46a", linewidth=1.0,
             linestyle="--", label="Buy & Hold Return %")
    ax2.axhline(0, color="#666666", linewidth=0.8)
    ax2.set_ylabel("Return (%)")
    ax2.legend(loc="upper left", facecolor="#1a1a2e",
               labelcolor="white", framealpha=0.8)
    ax2.grid(axis="y", color="#333333", linestyle="--", linewidth=0.5)

    # X-axis formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, color="white")

    plt.tight_layout()

    # Save to file
    output_path = CHARTS_DIR / f"{ticker}_backtest.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"Chart saved to {output_path}")
    return str(output_path)
"""
api.py - REST API layer
Exposes the backtesting engine via FastAPI.
Users can POST any ticker, date range, MA windows, and signal operator.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from fetcher import fetch_and_save
from backtest import run_full_backtest

app = FastAPI(
    title="Stock Backtesting API",
    description="Vectorized MA crossover backtesting engine with configurable strategy",
    version="1.0.0"
)


class BacktestRequest(BaseModel):
    ticker:          str   = Field(...,   example="AAPL")
    start:           str   = Field(...,   example="2020-01-01")
    end:             str   = Field(...,   example="2024-01-01")
    ma_fast:         int   = Field(5,     example=5,   ge=1)
    ma_slow:         int   = Field(20,    example=20,  ge=2)
    operator:        str   = Field(">",   example=">")
    initial_capital: float = Field(100_000.0, example=100000)

    @validator("operator")
    def validate_operator(cls, v):
        valid = {">", ">=", "<", "<="}
        if v not in valid:
            raise ValueError(f"operator must be one of: {valid}")
        return v

    @validator("ma_slow")
    def validate_ma_windows(cls, ma_slow, values):
        ma_fast = values.get("ma_fast", 5)
        if ma_fast >= ma_slow:
            raise ValueError(
                f"ma_fast ({ma_fast}) must be smaller than ma_slow ({ma_slow})"
            )
        return ma_slow


class BacktestResponse(BaseModel):
    ticker:          str
    strategy:        str
    total_return:    float
    market_return:   float
    sharpe_ratio:    float
    max_drawdown:    float
    num_trades:      int
    win_rate:        float
    final_value:     float
    initial_capital: float
    chart_url:       str


@app.get("/")
def root():
    return {"message": "Stock Backtesting API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest):
    """
    Run a backtest with fully configurable strategy parameters.

    Example request:
    {
        "ticker":   "TSLA",
        "start":    "2020-01-01",
        "end":      "2024-01-01",
        "ma_fast":  10,
        "ma_slow":  50,
        "operator": ">"
    }

    Operators:
    - ">"  : buy when fast MA is above slow MA (momentum)
    - ">=" : buy when fast MA >= slow MA
    - "<"  : buy when fast MA is below slow MA (mean reversion)
    - "<=" : buy when fast MA <= slow MA
    """
    try:
        fetch_and_save(request.ticker, request.start, request.end)

        metrics, chart_path = run_full_backtest(
            ticker=request.ticker,
            start=request.start,
            end=request.end,
            ma_fast=request.ma_fast,
            ma_slow=request.ma_slow,
            operator=request.operator,
        )

        strategy_label = (
            f"MA{request.ma_fast} {request.operator} MA{request.ma_slow}"
        )

        return BacktestResponse(
            ticker=request.ticker,
            strategy=strategy_label,
            chart_url=f"/chart/{request.ticker}",
            **metrics
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chart/{ticker}")
def get_chart(ticker: str):
    """
    Return the most recently generated chart image for a given ticker.
    """
    from pathlib import Path
    chart_path = Path("charts") / f"{ticker}_backtest.png"

    if not chart_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No chart found for {ticker}. Run /backtest first."
        )

    return FileResponse(chart_path, media_type="image/png")
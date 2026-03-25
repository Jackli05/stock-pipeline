# Stock Data Pipeline & Backtesting Engine

A vectorized stock data pipeline and strategy backtesting engine exposed via a REST API.
Supports configurable moving average crossover strategies on any stock ticker.

## Architecture
```
User Request (POST /backtest)
        │
        ▼
    api.py          ← FastAPI REST layer, input validation
        │
        ├── fetcher.py      ← Downloads OHLCV data from Yahoo Finance → SQLite
        │
        ├── pipeline.py     ← Cleans data, detects splits, computes MA features
        │
        ├── backtest.py     ← Vectorized signal generation + performance metrics
        │
        └── visualizer.py  ← Generates buy/sell chart (matplotlib)
```

## Design Decisions

**Why vectorized operations instead of row-by-row loops?**
NumPy vectorized operations run on contiguous memory blocks, avoiding Python
interpreter overhead on every iteration. On 1,000 rows, this is ~100x faster
than a for-loop. This matters when users run multi-ticker batch backtests.

**Why SQLite instead of PostgreSQL?**
For a single-user local backtesting tool, SQLite is sufficient — no server
setup, zero configuration, and pandas reads it natively. PostgreSQL would be
the right choice if this scaled to concurrent users or required role-based
access control. This is a deliberate tradeoff, not an oversight.

**Why FastAPI over Flask?**
FastAPI auto-generates OpenAPI docs, enforces request/response schemas via
Pydantic, and supports async natively — better suited for a data API that may
need to handle concurrent backtest requests.

## Features

- Download historical OHLCV data for any stock ticker via Yahoo Finance
- Automated data cleaning: deduplication, forward-fill, invalid price filtering
- Split/reverse-split detection: flags abnormal overnight price changes (>40%)
- Configurable MA crossover strategy: set any fast/slow window and operator
- Vectorized backtesting engine: computes Sharpe ratio, max drawdown, win rate
- Buy/sell signal chart with cumulative return comparison vs. buy-and-hold
- REST API with auto-generated Swagger docs

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API server
```bash
cd src
uvicorn api:app --reload
```

### 3. Open API docs
```
http://127.0.0.1:8000/docs
```

### 4. Run a backtest
```bash
curl -X POST http://127.0.0.1:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker":   "AAPL",
    "start":    "2020-01-01",
    "end":      "2024-01-01",
    "ma_fast":  5,
    "ma_slow":  20,
    "operator": ">"
  }'
```

### 5. View the chart
```
http://127.0.0.1:8000/chart/AAPL
```

## API Reference

### `POST /backtest`

| Field | Type | Default | Description |
|---|---|---|---|
| `ticker` | string | required | Stock symbol, e.g. `"AAPL"` |
| `start` | string | required | Start date, e.g. `"2020-01-01"` |
| `end` | string | required | End date, e.g. `"2024-01-01"` |
| `ma_fast` | int | `5` | Fast MA window (must be < ma_slow) |
| `ma_slow` | int | `20` | Slow MA window |
| `operator` | string | `">"` | Signal condition: `>`, `>=`, `<`, `<=` |
| `initial_capital` | float | `100000` | Starting portfolio value in USD |

**Strategy logic:**
- `">"` — go long when fast MA is above slow MA (momentum)
- `"<"` — go long when fast MA is below slow MA (mean reversion)

**Response fields:**
`total_return`, `market_return`, `sharpe_ratio`, `max_drawdown`,
`num_trades`, `win_rate`, `final_value`, `chart_url`

### `GET /chart/{ticker}`
Returns the most recently generated strategy chart as a PNG image.

## Example Results

**AAPL — MA5 > MA20 — 2020 to 2024**
| Metric | Value |
|---|---|
| Total Return | 137.11% |
| Market Return (Buy & Hold) | 3165.82% |
| Sharpe Ratio | 0.36 |
| Max Drawdown | -73.06% |
| Number of Trades | 186 |
| Win Rate | 51.69% |

## Project Structure
```
stock_pipepline/
├── src/
│   ├── fetcher.py      # Data ingestion: Yahoo Finance → SQLite
│   ├── pipeline.py     # Data cleaning + feature engineering
│   ├── backtest.py     # Vectorized signal generation + metrics
│   ├── visualizer.py   # Chart generation
│   └── api.py          # FastAPI REST layer
├── tests/
│   └── test_backtest.py
├── requirements.txt
└── README.md
```

## Tech Stack

Python · FastAPI · pandas · NumPy · SQLite · yfinance · matplotlib
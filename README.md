# The Brief — Personal Macro Market Intelligence Dashboard

A personal daily market analysis dashboard built with Python and Plotly.js. 
Pulls live data from FRED and Yahoo Finance, runs a custom analytics engine, 
and renders an interactive multi-tab dashboard in the browser.

![Dashboard](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Data](https://img.shields.io/badge/data-FRED%20%2B%20Yahoo%20Finance-orange)

## What It Does

Every morning, one command fetches fresh market data and regenerates the 
full dashboard:

```bash
python generate.py
python -m http.server 8000
# open http://localhost:8000
```

## Dashboard Tabs

### Equities
- **Market Regime Gauge** — Fear/Greed score (0–100) combining 5 signals: 
  breadth, cyclical/defensive ratio, volume z-score, macro/micro dispersion, 
  and sector composite
- **U.S. Major Indices** — Cumulative return chart (SPY, QQQ, IWM, DIA) 
  with 1M/3M/6M/YTD/1Y toggle
- **Commodities & Dollar** — UUP, GLD, USO, CPER with Copper/Gold ratio
- **Sector Positioning by Composite** — 11 sectors ranked by proprietary 
  composite signal (Return Z + Volume Z), top 3 / bottom 3 highlighted
- **52-Week High/Low Proximity** — Where each sector sits in its annual range
- **Daily Positioning Feed** — Macro vs Micro dispersion, RSP/SPY breadth, 
  Cyclical/Defensive ratio, SPY Volume Z-score
- **Drawdown from Peak** — SPY + all sector ETFs rolling drawdown
- **Rolling 60-Day Correlation Heatmap** — 11×11 sector correlation matrix
- **Factor & Sector Relative Performance** — Factor ETFs and sectors indexed 
  to 1.0 vs SPY with rolling 6M alpha
- **Individual ETF Flow & Price** — Dual-axis chart per ETF: return % + 
  volume z-score (green=accumulation, red=distribution)
- **Sector Return Heatmap** — 1M and 5D returns across all sectors
- **SPY Candlestick** — OHLCV with 20-day moving average

### Fixed Income & Macro
- **Key Rates Snapshot** — 2Y, 10Y, 30Y Treasury, Fed Funds, CPI YoY, 
  10Y-2Y spread with daily change badges
- **Yield Curve** — Current spot rates (1M→30Y) + historical yields over time
- **Curve Spreads** — 10Y-2Y and 10Y-3M with inversion markers
- **Real Yield & Breakeven** — TIPS real yield + 10Y breakeven inflation
- **Credit Spreads (OAS)** — HY OAS, IG OAS, HY-IG gap
- **CPI Component Breakdown** — Shelter, Food, Core Goods, Services YoY
- **Macro Charts** — CPI/PCE with 2% target, Fed Funds & Unemployment 
  dual mandate, Real GDP QoQ, Macro Pulse

### Calendar
- **Key Releases** — Latest vs previous for major macro indicators
- **Economic Release Calendar** — Past 35 days + next 45 days, 
  today highlighted
- **FOMC Meeting Dates** — Full 2025–2026 schedule with outcomes

## Architecture
the-brief/
├── generate.py          # Master script: fetch → analyze → write data.json
├── index.html           # Frontend: reads data.json, renders Plotly charts
├── data/
│   ├── fred_data.py     # FRED API fetching (15+ macro series)
│   ├── equity_data.py   # yfinance fetching (26 tickers, 2yr history)
│   └── cache.py         # Daily pickle cache with auto-purge
├── analytics/
│   ├── signals.py       # Z-scores, composite, breadth, regime score,
│   │                    # correlation matrix, drawdown, 52W range
│   ├── factors.py       # Factor/sector relative performance, rolling alpha,
│   │                    # ETF flow & price signals
│   └── fixed_income.py  # Yield curve, spreads, inflation decomposition,
│                        # credit spreads, macro series
└── utils/
└── formatting.py    # Color coding, number formatting helpers

## Data Sources

| Source | What | Series |
|--------|------|---------|
| FRED | Treasury yields | DGS1MO → DGS30 (9 maturities) |
| FRED | Macro indicators | CPI, PCE, Fed Funds, GDP, Unemployment, Retail Sales, Industrial Production |
| FRED | Spreads | T10Y2Y, T10Y3M, BAMLH0A0HYM2 (HY OAS), BAMLC0A0CM (IG OAS) |
| FRED | Inflation | DFII10 (Real Yield), T10YIE (Breakeven), CPI sub-components |
| Yahoo Finance | Equities | SPY, QQQ, IWM, DIA, RSP + 11 sector ETFs + 6 factor ETFs |
| Yahoo Finance | Commodities | GLD, USO, CPER, UUP |

## Setup

```bash
# 1. Clone
git clone https://github.com/SatvikRepaka19/the-brief.git
cd the-brief

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your FRED API key (free at fred.stlouisfed.org)
echo "FRED_API_KEY=your_key_here" > .env

# 4. Generate data and run
python generate.py
python -m http.server 8000
# Open http://localhost:8000
```

## Key Concepts

**Composite Score** — Per-sector signal combining Return Z-score and Volume 
Z-score (each rolling 252 days, clipped ±3). Positive = accumulation. 
Negative = distribution.

**Market Regime Score** — Weighted combination of 5 market signals normalized 
to 0–100. Scores: 0–25 Extreme Fear, 26–45 Fear, 46–55 Neutral, 56–75 Greed, 
76–100 Extreme Greed.

**Breadth (RSP/SPY)** — Equal-weight vs cap-weight ratio. Rising = broad 
market participation. Falling = narrow large-cap driven rally.

**Macro vs Micro Dispersion** — Rolling percentile rank of between-sector 
vs within-sector volatility. >0.75 = macro/sector-driven market. 
<0.25 = stock-picker's market.

---

*Built by Satvik Repaka | Data: FRED + Yahoo Finance*

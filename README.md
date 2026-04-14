# AlphaNet

Multi-asset investment intelligence platform combining equities, fixed income, and cross-asset analytics with institutional-grade tools.

## What It Does

AlphaNet bridges equity and credit markets in a single platform — the way institutional investors actually think. It answers questions like:

- **Where on the curve should I invest?** (Carry & roll-down analysis)
- **Why are yields moving?** (Term premium decomposition)
- **Is credit or equity mispriced?** (Merton structural credit model)
- **Does my diversification still work?** (Cross-asset correlation regimes)
- **What happens when things break?** (Historical stress testing)
- **Is the financial system stable?** (Funding stress monitoring)
- **Will the next auction go well?** (Treasury auction analytics)

## Tech Stack

- **Backend:** Python 3.11, FastAPI, NumPy, Pandas, scikit-learn, SciPy, QuantLib
- **Frontend:** HTMX, Tailwind CSS, Plotly.js
- **Data:** FRED API, yfinance, Finnhub, Reddit (PRAW), SEC EDGAR, TreasuryDirect
- **ML/Stats:** Gaussian Mixture Models, Hidden Markov Models, PCA, Nelson-Siegel curve fitting

## Architecture

```
src/
├── data/               # Data ingestion layer
│   ├── equities/       # yfinance price data
│   ├── fixed_income/   # FRED yields, ETFs, Treasury auctions
│   └── alternative/    # Reddit, news (Finnhub), SEC EDGAR
├── curve/              # Yield curve engine
│   ├── yield_curve.py  # Nelson-Siegel fitting, PCA, inversion detection
│   ├── duration.py     # Bond pricing, duration, convexity, DV01
│   ├── carry.py        # Carry & roll-down analysis
│   └── term_premium.py # ACM term premium decomposition
├── regime/             # Market regime detection
│   └── detector.py     # GMM/HMM with expanding-window (no look-ahead bias)
├── agents/             # Signal generation
│   └── council.py      # 4 rule-based analysts + risk manager with veto
├── strategies/         # Trading strategies
│   ├── equity/         # Momentum, mean reversion, sector rotation
│   ├── credit/         # Curve flattener, IG/HY rotation, butterfly
│   └── cross_asset/    # Credit-equity divergence, flight-to-quality
├── risk/               # Risk analytics
│   ├── metrics.py      # VaR, CVaR, drawdown, rolling Sharpe
│   ├── merton.py       # Structural credit model (equity as call option)
│   ├── correlation.py  # Cross-asset correlation regime tracker
│   ├── stress_test.py  # Historical & custom scenario stress testing
│   ├── funding_stress.py # Systemic funding stress monitor
│   └── auction_analytics.py # Treasury auction demand analysis
├── backtest/           # Backtesting engine
│   └── engine.py       # Walk-forward, regime-conditional, benchmark comparison
├── execution/          # Paper trading
│   └── paper_trader.py # Alpaca API integration
├── nlp/                # NLP
│   └── sentiment.py    # FinBERT financial sentiment scoring
├── api/routes/         # FastAPI endpoints (36 total)
└── dashboard/templates/# Frontend (10 pages)
```

## Dashboard Pages

| Page | What It Shows |
|------|---------------|
| **Market Intelligence** | Regime, yield curve, credit spreads, signal council (4 analysts debate), news sentiment, Reddit ticker buzz, ticker lookup |
| **Strategy Lab** | 6 backtestable strategies with configurable parameters, multi-strategy comparison |
| **Cross-Asset Signals** | Credit-equity divergence, risk-on/off composite score, equity/credit momentum |
| **Fixed Income Tools** | Bond calculator (price/duration/convexity/DV01), yield curve analysis, curve strategy signals |
| **Carry & Term Premium** | Carry/roll-down by maturity with breakeven, 10Y yield decomposition (expected rate vs term premium) |
| **Correlation Regimes** | Cross-asset correlation heatmap, stock-bond correlation history, PCA risk concentration |
| **Stress Testing** | 4 historical scenarios (GFC, COVID, 2022 Rate Shock, Tariff Turmoil), custom factor shocks |
| **Funding Stress** | 6 systemic stress indicators with z-scores, composite score, threshold alerts |
| **Treasury Auctions** | Recent results, upcoming calendar, bid-to-cover trends, demand forecasting |
| **Execution** | Paper trading via Alpaca (positions, P&L, order execution) |

## Key Features

### Signal Council
Four rule-based analysts score the market independently:
- **Equity Analyst** — momentum, trend, volatility
- **Credit Analyst** — spread z-scores, curve shape, spread momentum
- **Macro Strategist** — regime, VIX, cross-asset divergence
- **Risk Manager** — drawdown limits, VIX threshold, can veto any signal

### Data Leakage Prevention
- All rolling z-scores use `.shift(1)` to exclude current observation
- Expanding-window regime fitting (`fit_gmm_expanding`) for backtest-safe labels
- Backtest engine properly shifts signals by 1 day before multiplying with returns
- Documented warnings on full-sample methods

### Cross-Asset Analysis
- Credit-equity divergence detection (credit leads equity by 2-6 weeks)
- Merton structural model bridges equity and credit markets
- Correlation regime tracking identifies when diversification breaks down
- Composite risk-on/off score from 4 cross-asset factors

## Setup

```bash
# Clone
git clone https://github.com/aryamansingh01/AlphaNet.git
cd AlphaNet

# Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys (FRED is required, others optional)

# Run
uvicorn main:app --reload --port 8000

# Open http://localhost:8000
```

### API Keys Needed

| Key | Required | Free Tier | Get It |
|-----|----------|-----------|--------|
| FRED | Yes | Unlimited | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| Finnhub | Optional | 60 req/min | [finnhub.io](https://finnhub.io/register) |
| Reddit | Optional | 60 req/min | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |
| FMP | Optional | 250 req/day | [financialmodelingprep.com](https://financialmodelingprep.com/developer) |
| Alpaca | Paper trade only | Free | [alpaca.markets](https://app.alpaca.markets/signup) |

## Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
# 113 tests, all passing
```

## Docker

```bash
docker build -t alphanet .
docker run -p 8000:8000 --env-file .env alphanet
```

## Project Stats

- **80** Python files
- **19** HTML templates  
- **8,190** lines of Python
- **4,822** lines of HTML
- **36** API endpoints
- **10** dashboard pages
- **113** tests
- **7** institutional-grade analytics features
- **0** external AI/LLM dependencies

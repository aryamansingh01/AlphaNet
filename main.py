"""AlphaNet -- Multi-Asset Investment Intelligence & Systematic Credit Platform."""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import get_settings
from src.api.routes import signals, regime, curve, agents, backtest, portfolio
from src.api.routes import sentiment, fixed_income, cross_asset, funding, auction
from src.api.routes import merton, stress

settings = get_settings()

app = FastAPI(
    title="AlphaNet",
    description="Multi-Asset Investment Intelligence & Systematic Credit Platform",
    version="0.1.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")
templates = Jinja2Templates(directory="src/dashboard/templates")

# Core routes (always available)
app.include_router(signals.router)
app.include_router(regime.router)
app.include_router(curve.router)
app.include_router(agents.router)
app.include_router(backtest.router)
app.include_router(sentiment.router)
app.include_router(fixed_income.router)
app.include_router(cross_asset.router)
app.include_router(funding.router)
app.include_router(auction.router)
app.include_router(merton.router)
app.include_router(stress.router)

# Paper trading routes (only if mode = paper_trade)
if settings.mode == "paper_trade":
    app.include_router(portfolio.router)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/")
async def page_market(request: Request):
    """Market Intelligence dashboard (default landing page)."""
    return templates.TemplateResponse(
        "market.html",
        {"request": request, "mode": settings.mode, "page": "market"},
    )


@app.get("/market")
async def page_market_explicit(request: Request):
    """Market Intelligence dashboard (explicit path)."""
    return templates.TemplateResponse(
        "market.html",
        {"request": request, "mode": settings.mode, "page": "market"},
    )


@app.get("/strategy")
async def page_strategy(request: Request):
    """Strategy Lab -- backtesting and comparison."""
    return templates.TemplateResponse(
        "strategy.html",
        {"request": request, "mode": settings.mode, "page": "strategy"},
    )


@app.get("/cross-asset")
async def page_cross_asset(request: Request):
    """Cross-Asset Signals dashboard."""
    return templates.TemplateResponse(
        "cross_asset.html",
        {"request": request, "mode": settings.mode, "page": "cross-asset"},
    )


@app.get("/fixed-income")
async def page_fixed_income(request: Request):
    """Fixed Income Tools."""
    return templates.TemplateResponse(
        "fixed_income.html",
        {"request": request, "mode": settings.mode, "page": "fixed-income"},
    )


if settings.mode == "paper_trade":
    @app.get("/execution")
    async def page_execution(request: Request):
        """Execution panel (paper-trade mode only)."""
        return templates.TemplateResponse(
            "execution.html",
            {"request": request, "mode": settings.mode, "page": "execution"},
        )


@app.get("/carry")
async def page_carry(request: Request):
    """Carry & Term Premium dashboard."""
    return templates.TemplateResponse(
        "carry.html",
        {"request": request, "mode": settings.mode, "page": "carry"},
    )


@app.get("/correlation")
async def page_correlation(request: Request):
    """Correlation Regimes dashboard."""
    return templates.TemplateResponse(
        "correlation.html",
        {"request": request, "mode": settings.mode, "page": "correlation"},
    )


@app.get("/stress")
async def page_stress(request: Request):
    """Stress Testing dashboard."""
    return templates.TemplateResponse(
        "stress.html",
        {"request": request, "mode": settings.mode, "page": "stress"},
    )


@app.get("/funding")
async def page_funding(request: Request):
    """Funding Stress dashboard."""
    return templates.TemplateResponse(
        "funding.html",
        {"request": request, "mode": settings.mode, "page": "funding"},
    )


@app.get("/auction")
async def page_auction(request: Request):
    """Treasury Auctions dashboard."""
    return templates.TemplateResponse(
        "auction.html",
        {"request": request, "mode": settings.mode, "page": "auction"},
    )


@app.get("/demo")
async def page_demo(request: Request):
    """UI style comparison demo."""
    return templates.TemplateResponse(
        "demo.html",
        {"request": request, "mode": settings.mode, "page": "demo"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": "alphanet",
        "version": "0.1.0",
        "mode": settings.mode,
    }

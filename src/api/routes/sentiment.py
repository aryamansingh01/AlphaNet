"""Sentiment API endpoints — news and Reddit buzz."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

# ---------------------------------------------------------------------------
# Fallback sample data
# ---------------------------------------------------------------------------

_SAMPLE_NEWS = [
    {
        "title": "Fed signals potential rate pause amid cooling inflation",
        "source": "Reuters",
        "datetime": "2026-04-14T09:30:00",
        "sentiment_label": "positive",
        "sentiment_score": 0.72,
    },
    {
        "title": "Tech earnings disappoint as cloud growth slows",
        "source": "Bloomberg",
        "datetime": "2026-04-14T08:15:00",
        "sentiment_label": "negative",
        "sentiment_score": 0.81,
    },
    {
        "title": "Oil prices steady as OPEC holds production targets",
        "source": "CNBC",
        "datetime": "2026-04-13T16:00:00",
        "sentiment_label": "neutral",
        "sentiment_score": 0.65,
    },
    {
        "title": "Treasury yields fall on weaker-than-expected jobs data",
        "source": "WSJ",
        "datetime": "2026-04-13T14:30:00",
        "sentiment_label": "positive",
        "sentiment_score": 0.68,
    },
    {
        "title": "Credit spreads widen as default fears grow in CCC segment",
        "source": "FT",
        "datetime": "2026-04-13T11:00:00",
        "sentiment_label": "negative",
        "sentiment_score": 0.77,
    },
]

_SAMPLE_REDDIT_BUZZ = [
    {"ticker": "NVDA", "mentions": 142},
    {"ticker": "AAPL", "mentions": 98},
    {"ticker": "TSLA", "mentions": 87},
    {"ticker": "AMD", "mentions": 76},
    {"ticker": "PLTR", "mentions": 63},
    {"ticker": "MSFT", "mentions": 55},
    {"ticker": "AMZN", "mentions": 48},
    {"ticker": "GME", "mentions": 41},
    {"ticker": "META", "mentions": 37},
    {"ticker": "SOFI", "mentions": 29},
]


@router.get("/news")
async def news_sentiment():
    """Fetch recent financial news with sentiment scores.

    Tries Finnhub + FinBERT; falls back to sample data if keys are missing
    or models cannot be loaded.
    """
    try:
        from src.data.alternative.news_client import FinnhubNewsClient
        from src.nlp.sentiment import SentimentAnalyzer

        client = FinnhubNewsClient()
        df = client.get_market_news(limit=20)

        if df.empty:
            raise ValueError("No articles returned from Finnhub")

        analyzer = SentimentAnalyzer()
        df = analyzer.score_dataframe(df, text_col="title")

        articles = []
        for _, row in df.iterrows():
            articles.append({
                "title": row["title"],
                "source": row["source"],
                "datetime": str(row["datetime"]) if row["datetime"] else None,
                "sentiment_label": row["sentiment_label"],
                "sentiment_score": round(float(row["sentiment_score"]), 4),
            })

        return {
            "articles": articles,
            "count": len(articles),
            "source": "finnhub",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.warning("Live news sentiment unavailable, using sample data: %s", exc)
        return {
            "articles": _SAMPLE_NEWS,
            "count": len(_SAMPLE_NEWS),
            "source": "sample",
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/reddit")
async def reddit_buzz(subreddit: str = "wallstreetbets", limit: int = 20):
    """Top ticker mentions from a subreddit.

    Tries Reddit via PRAW; falls back to hardcoded sample buzz data if
    Reddit API keys are missing.
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from src.data.alternative.reddit_client import RedditClient

        client = RedditClient()
        buzz = client.get_ticker_buzz(
            subreddit=subreddit, limit=100, top_n=limit,
        )

        results = [{"ticker": ticker, "mentions": count} for ticker, count in buzz]

        return {
            "buzz": results,
            "subreddit": subreddit,
            "count": len(results),
            "source": "reddit",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.warning("Reddit API unavailable, using sample data: %s", exc)
        return {
            "buzz": _SAMPLE_REDDIT_BUZZ[:limit],
            "subreddit": subreddit,
            "count": min(limit, len(_SAMPLE_REDDIT_BUZZ)),
            "source": "sample",
            "timestamp": datetime.now().isoformat(),
        }

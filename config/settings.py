from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "AlphaNet"
    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./alphanet.db"

    # FRED
    fred_api_key: str = ""

    # Financial Modeling Prep
    fmp_api_key: str = ""

    # Finnhub
    finnhub_api_key: str = ""

    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "AlphaNet/1.0"

    # Alpaca (only needed if mode = "paper_trade")
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # Mode: "backtest" or "paper_trade"
    mode: str = "backtest"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

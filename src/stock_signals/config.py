"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    edgar_user_agent: str
    fmp_key: str | None
    twelvedata_key: str | None
    finnhub_key: str | None
    tiingo_key: str | None
    fred_key: str | None
    anthropic_key: str | None
    reddit_client_id: str | None
    reddit_client_secret: str | None
    bluesky_handle: str | None = None
    bluesky_app_password: str | None = None
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "signals.duckdb"


def _env(name: str) -> str | None:
    val = os.environ.get(name, "").strip().strip('"')
    return val or None


def load_config(env_file: Path | None = None) -> Config:
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    return Config(
        edgar_user_agent=_env("EDGAR_USER_AGENT")
        or "stock-signals/0.1 (ishtiaq.fauzan@gmail.com)",
        fmp_key=_env("FMP_API_KEY"),
        twelvedata_key=_env("TWELVEDATA_API_KEY"),
        finnhub_key=_env("FINNHUB_API_KEY"),
        tiingo_key=_env("TIINGO_API_KEY"),
        fred_key=_env("FRED_API_KEY"),
        anthropic_key=_env("ANTHROPIC_API_KEY"),
        reddit_client_id=_env("REDDIT_CLIENT_ID"),
        reddit_client_secret=_env("REDDIT_CLIENT_SECRET"),
        bluesky_handle=_env("BLUESKY_HANDLE"),
        bluesky_app_password=_env("BLUESKY_APP_PASSWORD"),
    )

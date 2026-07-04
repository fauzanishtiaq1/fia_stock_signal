"""S&P 500 universe: fetch from Wikipedia, cache locally, persist to DuckDB."""

from __future__ import annotations

import datetime
import io
import logging
from pathlib import Path

import duckdb
import pandas as pd
import requests

from stock_signals import db
from stock_signals.config import Config

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_COLUMNS = ["symbol", "name", "sector", "sub_industry", "cik", "added_at"]


def normalize_symbol(sym: str) -> str:
    """Normalize a ticker: strip whitespace, uppercase, '.' -> '-' (BRK.B -> BRK-B)."""
    return sym.strip().upper().replace(".", "-")


def fetch_sp500(user_agent: str) -> pd.DataFrame:
    """Fetch current S&P 500 constituents from Wikipedia.

    Returns a DataFrame with columns symbol, name, sector, sub_industry,
    cik (10-digit zero-padded string), added_at (today's date).
    """
    resp = requests.get(WIKI_URL, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    raw = pd.read_html(io.StringIO(resp.text))[0]
    df = pd.DataFrame(
        {
            "symbol": raw["Symbol"].astype(str).map(normalize_symbol),
            "name": raw["Security"].astype(str),
            "sector": raw["GICS Sector"].astype(str),
            "sub_industry": raw["GICS Sub-Industry"].astype(str),
            "cik": raw["CIK"].map(lambda x: f"{int(x):010d}"),
            "added_at": datetime.date.today(),
        }
    )
    return df[UNIVERSE_COLUMNS]


def _cache_path(config: Config) -> Path:
    return config.data_dir / "cache" / "sp500.parquet"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet via DuckDB (avoids a pyarrow dependency)."""
    con = duckdb.connect()
    try:
        con.register("universe_df", df)
        escaped = str(path).replace("'", "''")
        con.execute(f"COPY universe_df TO '{escaped}' (FORMAT PARQUET)")
    finally:
        con.close()


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame via DuckDB (avoids pyarrow)."""
    con = duckdb.connect()
    try:
        escaped = str(path).replace("'", "''")
        return con.execute(f"SELECT * FROM read_parquet('{escaped}')").df()
    finally:
        con.close()


def load_universe(config: Config) -> pd.DataFrame:
    """Fetch the S&P 500 universe, caching to parquet; fall back to cache on failure."""
    cache = _cache_path(config)
    try:
        df = fetch_sp500(config.edgar_user_agent)
    except Exception:
        if cache.exists():
            logger.warning(
                "fetch_sp500 failed; falling back to cached universe at %s",
                cache,
                exc_info=True,
            )
            return _read_parquet(cache)
        raise
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(df, cache)
    return df


def refresh_universe(con: duckdb.DuckDBPyConnection, config: Config) -> int:
    """Load the universe and upsert it into the universe table; return row count."""
    df = load_universe(config)
    return db.upsert_df(con, "universe", df)

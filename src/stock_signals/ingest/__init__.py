"""Data-source adapters. Each module defines one Source subclass."""

from __future__ import annotations

from stock_signals.config import Config
from stock_signals.ingest.base import Source


def all_sources(config: Config) -> list[Source]:
    from stock_signals.ingest.bluesky import BlueskySource
    from stock_signals.ingest.edgar import EdgarSource
    from stock_signals.ingest.finnhub import FinnhubSource
    from stock_signals.ingest.fmp import FmpSource
    from stock_signals.ingest.fred import FredSource
    from stock_signals.ingest.news_rss import GoogleNewsSource
    from stock_signals.ingest.reddit_src import RedditSource
    from stock_signals.ingest.tiingo import TiingoSource
    from stock_signals.ingest.twelvedata import TwelveDataSource

    return [
        EdgarSource(config),
        FmpSource(config),
        TwelveDataSource(config),
        FinnhubSource(config),
        TiingoSource(config),
        FredSource(config),
        GoogleNewsSource(config),
        RedditSource(config),
        BlueskySource(config),
    ]

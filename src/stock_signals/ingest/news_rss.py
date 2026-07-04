"""Google News RSS adapter (keyless): per-ticker headlines shaped for the news table."""

from __future__ import annotations

import calendar
import hashlib
from urllib.parse import quote_plus

import feedparser
import pandas as pd

from stock_signals.ingest.base import Source

FEED_URL = "https://news.google.com/rss/search"
NEWS_COLUMNS = ["id", "symbol", "published", "title", "source", "url"]
MAX_ROWS = 50


class GoogleNewsSource(Source):
    """Headlines from Google News RSS search (no API key required)."""

    name = "google_news"
    key_attr = None
    min_interval = 2.0

    def _fetch_feed(self, query: str) -> feedparser.FeedParserDict:
        """Fetch and parse the Google News RSS feed for a search query."""
        url = f"{FEED_URL}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = self._get(url)
        return feedparser.parse(resp.text)

    def ticker_news(self, symbol: str, company_name: str | None = None) -> pd.DataFrame:
        """Recent headlines for a ticker, shaped for the news table (max 50 rows)."""
        if company_name:
            query = f'"{company_name}" OR {symbol} stock'
        else:
            query = f"{symbol} stock"
        feed = self._fetch_feed(query)
        rows: list[dict] = []
        for entry in feed.entries[:MAX_ROWS]:
            link = entry.get("link", "")
            parsed = entry.get("published_parsed")
            published = (
                pd.Timestamp(calendar.timegm(parsed), unit="s") if parsed else pd.NaT
            )
            src = entry.get("source")
            source = src.get("title") if src and src.get("title") else "google_news"
            rows.append(
                {
                    "id": hashlib.md5(link.encode()).hexdigest(),
                    "symbol": symbol,
                    "published": published,
                    "title": entry.get("title", ""),
                    "source": source,
                    "url": link,
                }
            )
        if not rows:
            return pd.DataFrame(columns=NEWS_COLUMNS)
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)

    def _healthcheck_call(self) -> str:
        feed = self._fetch_feed("stock market")
        return f"{len(feed.entries)} headlines"

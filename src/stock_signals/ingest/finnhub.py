"""Finnhub adapter: real-time quote and company news (free tier: 60/min)."""

from __future__ import annotations

import pandas as pd

from stock_signals.ingest.base import Source

BASE = "https://finnhub.io/api/v1"

NEWS_COLUMNS = ["id", "symbol", "published", "title", "source", "url"]


class FinnhubSource(Source):
    """Finnhub (finnhub.io)."""

    name = "finnhub"
    key_attr = "finnhub_key"
    min_interval = 1.1  # free tier: 60 requests/minute

    def _get_json(self, path: str, **params) -> dict | list:
        params["token"] = self.key
        return self._get(f"{BASE}{path}", params=params).json()

    def quote(self, symbol: str) -> dict:
        """Raw quote JSON: c, d, dp, h, l, o, pc, t."""
        return self._get_json("/quote", symbol=symbol)

    def company_news(self, symbol: str, date_from: str, date_to: str) -> pd.DataFrame:
        """Company news between two YYYY-MM-DD dates, shaped for the news table."""
        data = self._get_json(
            "/company-news", symbol=symbol, **{"from": date_from, "to": date_to}
        )
        rows = [
            {
                "id": str(item["id"]),
                "symbol": symbol,
                "published": pd.to_datetime(item["datetime"], unit="s"),
                "title": item["headline"],
                "source": item["source"],
                "url": item["url"],
            }
            for item in (data if isinstance(data, list) else [])
        ]
        if not rows:
            return pd.DataFrame(columns=NEWS_COLUMNS)
        return pd.DataFrame(rows)[NEWS_COLUMNS]

    def _healthcheck_call(self) -> str:
        return f"AAPL {self.quote('AAPL')['c']}"

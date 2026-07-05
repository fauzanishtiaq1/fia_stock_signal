"""Tiingo adapter: long daily price history (header token auth)."""

from __future__ import annotations

import pandas as pd

from stock_signals.ingest.base import Source

BASE = "https://api.tiingo.com"

PRICE_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close",
    "adj_close", "volume", "source",
]


class TiingoSource(Source):
    """Tiingo (tiingo.com)."""

    name = "tiingo"
    key_attr = "tiingo_key"
    min_interval = 0.5

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self.key}",
            "Content-Type": "application/json",
        }

    def daily_history(self, symbol: str, start: str = "1990-01-01") -> pd.DataFrame:
        """Full daily OHLCV history since `start`, shaped for prices_daily."""
        resp = self._get(
            f"{BASE}/tiingo/daily/{symbol}/prices",
            params={"startDate": start, "format": "json"},
            headers=self._headers,
        )
        data = resp.json()
        if not data:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        df = pd.DataFrame(data)
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": pd.to_datetime(df["date"]).dt.date,
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "adj_close": df["adjClose"].astype(float),
                "volume": df["volume"].astype("int64"),
                "source": "tiingo",
            }
        )
        return out[PRICE_COLUMNS]

    def _healthcheck_call(self) -> str:
        # /api/test does not validate the token, so prove auth with a real
        # metadata request instead.
        resp = self._get(f"{BASE}/tiingo/daily/AAPL", headers=self._headers)
        meta = resp.json()
        return f"AAPL history {meta.get('startDate', '?')[:10]} → {meta.get('endDate', '?')[:10]}"

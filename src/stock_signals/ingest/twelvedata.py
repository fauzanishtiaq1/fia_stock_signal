"""Twelve Data adapter: daily price series (free tier: 8 credits/min)."""

from __future__ import annotations

import pandas as pd

from stock_signals.ingest.base import Source

BASE = "https://api.twelvedata.com"

PRICE_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close",
    "adj_close", "volume", "source",
]


class TwelveDataSource(Source):
    """Twelve Data (twelvedata.com)."""

    name = "twelvedata"
    key_attr = "twelvedata_key"
    min_interval = 8.0  # free tier: 8 credits/minute

    def _get_json(self, path: str, **params) -> dict:
        params["apikey"] = self.key
        data = self._get(f"{BASE}{path}", params=params).json()
        if isinstance(data, dict) and data.get("code") and data.get("message"):
            raise RuntimeError(f"twelvedata {data['code']}: {data['message']}")
        return data

    def daily_prices(self, symbol: str, outputsize: int = 500) -> pd.DataFrame:
        """Daily OHLCV series shaped for prices_daily (adj_close = close)."""
        data = self._get_json(
            "/time_series", symbol=symbol, interval="1day", outputsize=outputsize
        )
        values = data.get("values", [])
        if not values:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        df = pd.DataFrame(values)
        close = df["close"].astype(float)
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": pd.to_datetime(df["datetime"]).dt.date,
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": close,
                "adj_close": close,  # endpoint returns unadjusted series only
                "volume": df["volume"].astype("int64"),
                "source": "twelvedata",
            }
        )
        return out[PRICE_COLUMNS]

    def _healthcheck_call(self) -> str:
        data = self._get_json("/price", symbol="AAPL")
        return str(data["price"])

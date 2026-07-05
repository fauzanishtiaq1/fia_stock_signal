"""Financial Modeling Prep adapter: prices, fundamentals, analyst estimates.

Uses the current "stable" API — the legacy /api/v3 endpoints return 403 for
accounts created after FMP's migration.
"""

from __future__ import annotations

import pandas as pd

from stock_signals.ingest.base import Source

BASE = "https://financialmodelingprep.com/stable"

PRICE_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close",
    "adj_close", "volume", "source",
]
ESTIMATE_COLUMNS = ["symbol", "date", "metric", "value"]


class FmpSource(Source):
    """Financial Modeling Prep (financialmodelingprep.com)."""

    name = "fmp"
    key_attr = "fmp_key"
    min_interval = 0.25

    def _get_json(self, path: str, **params) -> dict | list:
        params["apikey"] = self.key
        return self._get(f"{BASE}{path}", params=params).json()

    def daily_prices(self, symbol: str, start: str | None = None) -> pd.DataFrame:
        """Daily OHLCV plus dividend-adjusted close, shaped for prices_daily.

        The stable API splits these across two endpoints, so this costs TWO
        API calls per symbol: /historical-price-eod/full (unadjusted OHLCV)
        and /historical-price-eod/dividend-adjusted (adjClose), merged on date.
        """
        params: dict = {"symbol": symbol}
        if start:
            params["from"] = start
        full = self._get_json("/historical-price-eod/full", **params)
        if not isinstance(full, list) or not full:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        df = pd.DataFrame(full)

        adj = self._get_json("/historical-price-eod/dividend-adjusted", **params)
        if isinstance(adj, list) and adj:
            adj_df = pd.DataFrame(adj)[["date", "adjClose"]]
            df = df.merge(adj_df, on="date", how="left")
        else:
            df["adjClose"] = df["close"]

        out = pd.DataFrame(
            {
                "symbol": symbol,
                "date": pd.to_datetime(df["date"]).dt.date,
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "adj_close": df["adjClose"].fillna(df["close"]).astype(float),
                "volume": df["volume"].astype("int64"),
                "source": "fmp",
            }
        )
        return out[PRICE_COLUMNS]

    def income_statements(
        self, symbol: str, period: str = "quarter", limit: int = 20
    ) -> pd.DataFrame:
        """Raw income statements as returned by the API (one row per period)."""
        data = self._get_json(
            "/income-statement", symbol=symbol, period=period, limit=limit
        )
        return pd.DataFrame(data if isinstance(data, list) else [])

    def analyst_estimates(
        self, symbol: str, period: str = "annual", limit: int = 8
    ) -> pd.DataFrame:
        """Analyst estimates in long form, shaped for the estimates table.

        One row per (symbol, date, metric, value); metric names are the raw
        API field names. period="quarter" requires a paid FMP plan — the
        free tier only allows annual.
        """
        data = self._get_json(
            "/analyst-estimates", symbol=symbol, period=period, limit=limit
        )
        rows: list[dict] = []
        for item in data if isinstance(data, list) else []:
            date = pd.to_datetime(item.get("date")).date()
            for field, value in item.items():
                if field in ("symbol", "date") or not isinstance(value, (int, float)):
                    continue
                rows.append(
                    {"symbol": symbol, "date": date, "metric": field,
                     "value": float(value)}
                )
        if not rows:
            return pd.DataFrame(columns=ESTIMATE_COLUMNS)
        return pd.DataFrame(rows)[ESTIMATE_COLUMNS]

    def _healthcheck_call(self) -> str:
        data = self._get_json("/profile", symbol="AAPL")
        return data[0]["companyName"]

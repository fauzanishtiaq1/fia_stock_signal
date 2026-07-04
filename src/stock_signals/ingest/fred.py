"""FRED (St. Louis Fed) adapter: macro series observations shaped for the macro table."""

from __future__ import annotations

import pandas as pd

from stock_signals.ingest.base import Source

BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_SERIES = ["DGS10", "DGS2", "CPIAUCSL", "UNRATE", "VIXCLS"]
MACRO_COLUMNS = ["series_id", "date", "value"]


class FredSource(Source):
    """Federal Reserve Economic Data API (requires FRED_API_KEY)."""

    name = "fred"
    key_attr = "fred_key"
    min_interval = 0.6

    def series_observations(
        self, series_id: str, start: str | None = None, **params: object
    ) -> pd.DataFrame:
        """Observations for a FRED series as a macro-shaped DataFrame.

        Skips FRED's missing-value marker ("."). Extra keyword args are passed
        through as query params (e.g. sort_order="desc", limit=1).
        """
        query: dict[str, object] = {
            "series_id": series_id,
            "api_key": self.key,
            "file_type": "json",
            **params,
        }
        if start:
            query["observation_start"] = start
        resp = self._get(f"{BASE_URL}/series/observations", params=query)
        observations = resp.json().get("observations", [])
        rows = [
            {
                "series_id": series_id,
                "date": obs["date"],
                "value": float(obs["value"]),
            }
            for obs in observations
            if obs.get("value") not in (".", "", None)
        ]
        if not rows:
            return pd.DataFrame(columns=MACRO_COLUMNS)
        df = pd.DataFrame(rows, columns=MACRO_COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _healthcheck_call(self) -> str:
        df = self.series_observations("DGS10", sort_order="desc", limit=1)
        value = df["value"].iloc[0] if not df.empty else "n/a"
        return f"DGS10={value}"

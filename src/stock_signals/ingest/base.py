"""Adapter contract every data source implements."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from stock_signals.config import Config


@dataclass
class SourceStatus:
    name: str
    available: bool  # credentials present (always True for keyless sources)
    ok: bool | None  # healthcheck result; None if skipped because unavailable
    detail: str


class Source:
    """Base class for data-source adapters.

    Subclasses set `name`, and `key_attr` (Config attribute holding the API key,
    or None for keyless sources), and implement `healthcheck()` plus their own
    fetch methods returning pandas DataFrames shaped for db.py tables.
    """

    name: str = ""
    key_attr: str | None = None
    min_interval: float = 0.0  # seconds between requests (rate limiting)

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.edgar_user_agent
        self._last_request = 0.0

    @property
    def key(self) -> str | None:
        return getattr(self.config, self.key_attr) if self.key_attr else None

    @property
    def available(self) -> bool:
        return self.key_attr is None or self.key is not None

    def _get(self, url: str, **kwargs) -> requests.Response:
        """GET with source-level rate limiting; raises for HTTP errors."""
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=30, **kwargs)
        self._last_request = time.monotonic()
        resp.raise_for_status()
        return resp

    def healthcheck(self) -> SourceStatus:
        """Cheapest possible live call proving the source works."""
        if not self.available:
            return SourceStatus(self.name, False, None, f"no key ({self.key_attr})")
        try:
            detail = self._healthcheck_call()
            return SourceStatus(self.name, True, True, detail)
        except Exception as exc:  # noqa: BLE001 - smoke report, not control flow
            return SourceStatus(self.name, True, False, f"{type(exc).__name__}: {exc}")

    def _healthcheck_call(self) -> str:
        raise NotImplementedError

"""Bluesky adapter: post search over AT Protocol with an app-password session.

Live probe (2026-07): GET public.api.bsky.app/xrpc/app.bsky.feed.searchPosts
returns 403 without auth (other appview endpoints like getProfile stay open),
so search authenticates against bsky.social via com.atproto.server.createSession.
"""

from __future__ import annotations

import pandas as pd
import requests

from stock_signals.config import Config
from stock_signals.ingest.base import Source

PDS_BASE = "https://bsky.social"

POST_COLUMNS = ["id", "platform", "created", "author", "text"]


class BlueskySource(Source):
    """Bluesky post search (bsky.social, handle + app password required)."""

    name = "bluesky"
    key_attr = "bluesky_app_password"
    min_interval = 1.0

    def __init__(self, config: Config):
        super().__init__(config)
        self._jwt: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.config.bluesky_handle and self.config.bluesky_app_password)

    def _create_session(self) -> str:
        resp = self.session.post(
            f"{PDS_BASE}/xrpc/com.atproto.server.createSession",
            json={
                "identifier": self.config.bluesky_handle,
                "password": self.config.bluesky_app_password,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["accessJwt"]

    def _get_json(self, path: str, **params) -> dict:
        if self._jwt is None:
            self._jwt = self._create_session()
        url = f"{PDS_BASE}/xrpc/{path}"
        try:
            headers = {"Authorization": f"Bearer {self._jwt}"}
            return self._get(url, params=params, headers=headers).json()
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 401:
                raise
            self._jwt = self._create_session()  # expired access token: refresh once
            headers = {"Authorization": f"Bearer {self._jwt}"}
            return self._get(url, params=params, headers=headers).json()

    def search_posts(self, query: str, limit: int = 50) -> pd.DataFrame:
        """Posts matching a search query, shaped like the reddit posts frame."""
        data = self._get_json("app.bsky.feed.searchPosts", q=query, limit=limit)
        rows = [
            {
                "id": post["uri"],
                "platform": "bluesky",
                "created": _naive_utc(post.get("record", {}).get("createdAt")),
                "author": post.get("author", {}).get("handle", ""),
                "text": post.get("record", {}).get("text", ""),
            }
            for post in data.get("posts", [])
        ]
        if not rows:
            return pd.DataFrame(columns=POST_COLUMNS)
        return pd.DataFrame(rows)[POST_COLUMNS]

    def _healthcheck_call(self) -> str:
        return f"{len(self.search_posts('stock market', limit=5))} posts"


def _naive_utc(value: str | None) -> pd.Timestamp:
    """Parse an ISO timestamp to a tz-naive UTC Timestamp (NaT if missing)."""
    ts = pd.to_datetime(value, utc=True)
    return ts.tz_localize(None) if ts is not pd.NaT else ts

"""Reddit adapter: subreddit listings, unauthenticated or OAuth app-only."""

from __future__ import annotations

import pandas as pd
import requests

from stock_signals.config import Config
from stock_signals.ingest.base import Source

PUBLIC_BASE = "https://www.reddit.com"
OAUTH_BASE = "https://oauth.reddit.com"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

POST_COLUMNS = ["id", "platform", "created", "author", "text"]

UNAUTH_INTERVAL = 6.5  # official unauthenticated limit: 10 requests/minute
OAUTH_INTERVAL = 0.7  # app-only OAuth: 100 requests/minute


class RedditSource(Source):
    """Subreddit posts via public JSON listings (keyless) or OAuth app-only.

    With no credentials, hits www.reddit.com/*.json at 10 QPM. When
    reddit_client_id + reddit_client_secret are set, fetches an app-only
    token (client_credentials) and uses oauth.reddit.com at 100 QPM,
    refreshing the token once on 401.
    """

    name = "reddit"
    key_attr = "reddit_client_id"  # unauth JSON is 403-blocked in practice (2026)
    min_interval = UNAUTH_INTERVAL

    SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]

    def __init__(self, config: Config):
        super().__init__(config)
        self._token: str | None = None
        if self._has_creds:
            self.min_interval = OAUTH_INTERVAL

    @property
    def _has_creds(self) -> bool:
        return bool(self.config.reddit_client_id and self.config.reddit_client_secret)

    @property
    def available(self) -> bool:
        # Reddit hard-blocks anonymous JSON access from most IPs now, so
        # require OAuth creds; recent_posts still tries unauth if called
        # directly without them.
        return self._has_creds

    def _fetch_token(self) -> str:
        resp = self.session.post(
            TOKEN_URL,
            auth=(self.config.reddit_client_id, self.config.reddit_client_secret),
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def _get_listing(self, subreddit: str, listing: str, limit: int) -> dict:
        """One listing page as parsed JSON, via OAuth when creds exist."""
        params = {"limit": limit, "raw_json": 1}
        if not self._has_creds:
            return self._get(f"{PUBLIC_BASE}/r/{subreddit}/{listing}.json", params=params).json()
        if self._token is None:
            self._token = self._fetch_token()
        url = f"{OAUTH_BASE}/r/{subreddit}/{listing}"
        try:
            headers = {"Authorization": f"bearer {self._token}"}
            return self._get(url, params=params, headers=headers).json()
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 401:
                raise
            self._token = self._fetch_token()  # expired token: refresh once
            headers = {"Authorization": f"bearer {self._token}"}
            return self._get(url, params=params, headers=headers).json()

    def recent_posts(
        self,
        subreddits: list[str] | None = None,
        listings: tuple[str, ...] = ("hot", "new"),
        limit: int = 100,
    ) -> pd.DataFrame:
        """Recent non-stickied posts across subreddits, deduped by fullname."""
        rows: list[dict] = []
        seen: set[str] = set()
        for subreddit in subreddits if subreddits is not None else self.SUBREDDITS:
            for listing in listings:
                data = self._get_listing(subreddit, listing, limit)
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    post_id = post.get("name")
                    if not post_id or post_id in seen or post.get("stickied"):
                        continue
                    seen.add(post_id)
                    rows.append(
                        {
                            "id": post_id,
                            "platform": "reddit",
                            "created": pd.to_datetime(post.get("created_utc"), unit="s"),
                            "author": post.get("author", ""),
                            "text": f"{post.get('title', '')}\n{post.get('selftext', '')}".strip(),
                        }
                    )
        if not rows:
            return pd.DataFrame(columns=POST_COLUMNS)
        return pd.DataFrame(rows)[POST_COLUMNS]

    def _healthcheck_call(self) -> str:
        mode = "oauth" if self._has_creds else "unauth"
        data = self._get_listing("wallstreetbets", "hot", 5)
        return f"{len(data.get('data', {}).get('children', []))} posts via {mode}"

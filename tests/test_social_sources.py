"""Offline tests for the Reddit and Bluesky adapters.

No network: fixture JSON (embedded below) is fed through the DataFrame-shaping
code paths by monkeypatching the HTTP layer (_get / _get_json).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from stock_signals.ingest.bluesky import BlueskySource
from stock_signals.ingest.bluesky import POST_COLUMNS as BSKY_COLUMNS
from stock_signals.ingest.reddit_src import POST_COLUMNS as REDDIT_COLUMNS
from stock_signals.ingest.reddit_src import RedditSource

REDDIT_HOT_JSON = """
{
  "kind": "Listing",
  "data": {
    "after": "t3_bbb222",
    "children": [
      {
        "kind": "t3",
        "data": {
          "name": "t3_sticky1",
          "title": "Daily Discussion Thread",
          "selftext": "Talk here.",
          "author": "AutoModerator",
          "created_utc": 1751673600,
          "stickied": true
        }
      },
      {
        "kind": "t3",
        "data": {
          "name": "t3_aaa111",
          "title": "NVDA to the moon",
          "selftext": "Bought calls this morning.",
          "author": "diamondhands42",
          "created_utc": 1751670000,
          "stickied": false
        }
      },
      {
        "kind": "t3",
        "data": {
          "name": "t3_bbb222",
          "title": "AAPL earnings play",
          "selftext": "",
          "author": "thetagang99",
          "created_utc": 1751666400,
          "stickied": false
        }
      }
    ]
  }
}
"""

REDDIT_NEW_JSON = """
{
  "kind": "Listing",
  "data": {
    "after": null,
    "children": [
      {
        "kind": "t3",
        "data": {
          "name": "t3_bbb222",
          "title": "AAPL earnings play",
          "selftext": "",
          "author": "thetagang99",
          "created_utc": 1751666400,
          "stickied": false
        }
      },
      {
        "kind": "t3",
        "data": {
          "name": "t3_ccc333",
          "title": "TSLA delivery numbers",
          "selftext": "Q2 deliveries beat estimates.",
          "author": "ev_bull",
          "created_utc": 1751662800,
          "stickied": false
        }
      }
    ]
  }
}
"""

REDDIT_EMPTY_JSON = '{"kind": "Listing", "data": {"after": null, "children": []}}'

BSKY_SEARCH_JSON = """
{
  "posts": [
    {
      "uri": "at://did:plc:abc123/app.bsky.feed.post/3kxyz1",
      "cid": "bafyreiaaa",
      "author": {"did": "did:plc:abc123", "handle": "trader.bsky.social"},
      "record": {
        "$type": "app.bsky.feed.post",
        "createdAt": "2026-07-04T14:30:00.000Z",
        "text": "Stock market rally continues into July."
      }
    },
    {
      "uri": "at://did:plc:def456/app.bsky.feed.post/3kxyz2",
      "cid": "bafyreibbb",
      "author": {"did": "did:plc:def456", "handle": "macro.watcher"},
      "record": {
        "$type": "app.bsky.feed.post",
        "createdAt": "2026-07-04T09:15:30.500Z",
        "text": "Fed minutes moved the stock indices today."
      }
    }
  ],
  "hitsTotal": 2
}
"""

BSKY_EMPTY_JSON = '{"posts": [], "hitsTotal": 0}'


class FakeResponse:
    def __init__(self, payload: str):
        self._payload = payload

    def json(self):
        return json.loads(self._payload)


def _patch_reddit_get(source: RedditSource, by_listing: dict[str, str]) -> list[str]:
    """Route source._get to fixture payloads keyed by listing name; log URLs."""
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        for listing, payload in by_listing.items():
            if url.endswith(f"/{listing}.json") or url.endswith(f"/{listing}"):
                return FakeResponse(payload)
        raise AssertionError(f"unexpected URL {url}")

    source._get = fake_get
    return calls


# --- Reddit -----------------------------------------------------------------


def test_reddit_recent_posts_shape_dedupe_and_sticky(cfg):
    source = RedditSource(cfg)
    _patch_reddit_get(source, {"hot": REDDIT_HOT_JSON, "new": REDDIT_NEW_JSON})
    df = source.recent_posts(subreddits=["wallstreetbets"])

    assert list(df.columns) == REDDIT_COLUMNS
    # t3_bbb222 appears in both listings -> deduped; stickied t3_sticky1 skipped
    assert sorted(df["id"]) == ["t3_aaa111", "t3_bbb222", "t3_ccc333"]
    assert (df["platform"] == "reddit").all()
    assert df["created"].dt.tz is None  # naive UTC, consistent with other adapters
    assert df["created"].notna().all()

    by_id = df.set_index("id")
    assert by_id.loc["t3_aaa111", "text"] == "NVDA to the moon\nBought calls this morning."
    assert by_id.loc["t3_bbb222", "text"] == "AAPL earnings play"  # empty selftext
    assert by_id.loc["t3_ccc333", "author"] == "ev_bull"


def test_reddit_dedupes_across_subreddits_and_hits_every_listing(cfg):
    source = RedditSource(cfg)
    calls = _patch_reddit_get(source, {"hot": REDDIT_HOT_JSON, "new": REDDIT_NEW_JSON})
    df = source.recent_posts(subreddits=["wallstreetbets", "stocks"])
    assert len(calls) == 4  # 2 subreddits x 2 listings
    assert len(df) == 3  # same posts served for both subs -> deduped by id


def test_reddit_empty_listing_returns_empty_frame_with_columns(cfg):
    source = RedditSource(cfg)
    _patch_reddit_get(source, {"hot": REDDIT_EMPTY_JSON, "new": REDDIT_EMPTY_JSON})
    df = source.recent_posts(subreddits=["stocks"])
    assert df.empty
    assert list(df.columns) == REDDIT_COLUMNS


def test_reddit_always_available_and_mode_sets_interval(cfg):
    unauth = RedditSource(cfg)
    assert unauth.available is False
    assert unauth.min_interval == 6.5

    keyed_cfg = dataclasses.replace(cfg, reddit_client_id="cid", reddit_client_secret="sec")
    authed = RedditSource(keyed_cfg)
    assert authed.available is True
    assert authed.min_interval == 0.7


# --- Bluesky ----------------------------------------------------------------


@pytest.fixture
def bsky_cfg(cfg):
    return dataclasses.replace(
        cfg, bluesky_handle="me.bsky.social", bluesky_app_password="app-pass"
    )


def test_bluesky_search_posts_shape(bsky_cfg, monkeypatch):
    source = BlueskySource(bsky_cfg)
    seen: dict = {}

    def fake_get_json(path, **params):
        seen["path"], seen["params"] = path, params
        return json.loads(BSKY_SEARCH_JSON)

    monkeypatch.setattr(source, "_get_json", fake_get_json)
    df = source.search_posts("stock", limit=25)

    assert seen["path"] == "app.bsky.feed.searchPosts"
    assert seen["params"] == {"q": "stock", "limit": 25}
    assert list(df.columns) == BSKY_COLUMNS
    assert list(df["id"]) == [
        "at://did:plc:abc123/app.bsky.feed.post/3kxyz1",
        "at://did:plc:def456/app.bsky.feed.post/3kxyz2",
    ]
    assert (df["platform"] == "bluesky").all()
    assert df["created"].dt.tz is None  # naive UTC, consistent with reddit frame
    assert str(df["created"].iloc[0]) == "2026-07-04 14:30:00"
    assert list(df["author"]) == ["trader.bsky.social", "macro.watcher"]
    assert df["text"].iloc[1] == "Fed minutes moved the stock indices today."


def test_bluesky_empty_returns_empty_frame_with_columns(bsky_cfg, monkeypatch):
    source = BlueskySource(bsky_cfg)
    monkeypatch.setattr(source, "_get_json", lambda path, **p: json.loads(BSKY_EMPTY_JSON))
    df = source.search_posts("stock")
    assert df.empty
    assert list(df.columns) == BSKY_COLUMNS


def test_bluesky_availability_requires_handle_and_password(cfg, bsky_cfg):
    assert BlueskySource(cfg).available is False
    assert BlueskySource(bsky_cfg).available is True
    half = dataclasses.replace(cfg, bluesky_handle="me.bsky.social")
    assert BlueskySource(half).available is False

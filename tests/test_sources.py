"""Tests for source registry, availability, and offline config loading.

No network: healthcheck() is only called on keyed sources whose key is None,
which short-circuits before any HTTP request. Keyless sources (edgar,
google_news) are never healthchecked here because that would hit
the network.
"""

from __future__ import annotations

from pathlib import Path

from stock_signals.config import load_config
from stock_signals.ingest import all_sources
from stock_signals.ingest.base import SourceStatus

EXPECTED_NAMES = {
    "edgar",
    "fmp",
    "twelvedata",
    "finnhub",
    "tiingo",
    "fred",
    "google_news",
    "reddit",
    "bluesky",
}
KEYLESS = {"edgar", "google_news"}  # reddit needs OAuth creds (anon JSON is blocked)
ENV_VARS = [
    "EDGAR_USER_AGENT",
    "FMP_API_KEY",
    "TWELVEDATA_API_KEY",
    "FINNHUB_API_KEY",
    "TIINGO_API_KEY",
    "FRED_API_KEY",
    "ANTHROPIC_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "BLUESKY_HANDLE",
    "BLUESKY_APP_PASSWORD",
]


def test_all_sources_returns_nine_uniquely_named(cfg):
    sources = all_sources(cfg)
    assert len(sources) == 9
    names = [s.name for s in sources]
    assert len(set(names)) == 9
    assert set(names) == EXPECTED_NAMES


def test_keyless_available_keyed_without_keys_unavailable(cfg):
    for source in all_sources(cfg):
        if source.name in KEYLESS:
            assert source.available is True, source.name
        else:
            assert source.available is False, source.name


def test_healthcheck_on_unavailable_keyed_source_skips_network(cfg):
    for source in all_sources(cfg):
        if source.name in KEYLESS:
            continue  # keyless healthchecks hit the network; never call them here
        status = source.healthcheck()
        assert isinstance(status, SourceStatus)
        assert status.name == source.name
        assert status.available is False
        assert status.ok is None


def test_load_config_defaults_without_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    config = load_config(env_file=Path("/nonexistent"))
    assert config.fmp_key is None
    assert config.twelvedata_key is None
    assert config.finnhub_key is None
    assert config.tiingo_key is None
    assert config.fred_key is None
    assert config.anthropic_key is None
    assert config.reddit_client_id is None
    assert config.reddit_client_secret is None
    assert config.bluesky_handle is None
    assert config.bluesky_app_password is None
    assert isinstance(config.edgar_user_agent, str)
    assert config.edgar_user_agent.strip() != ""

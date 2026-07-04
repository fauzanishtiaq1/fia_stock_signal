"""Shared fixtures. All tests are offline: no network calls."""

from __future__ import annotations

import pytest

from stock_signals import db
from stock_signals.config import Config


@pytest.fixture
def cfg(tmp_path):
    """A Config built directly (no env vars): keyless-only, temp data dir."""
    return Config(
        edgar_user_agent="test-agent test@example.com",
        fmp_key=None,
        twelvedata_key=None,
        finnhub_key=None,
        tiingo_key=None,
        fred_key=None,
        anthropic_key=None,
        reddit_client_id=None,
        reddit_client_secret=None,
        data_dir=tmp_path / "data",
    )


@pytest.fixture
def con(tmp_path):
    """A DuckDB connection to a fresh temp database with the schema applied."""
    connection = db.connect(tmp_path / "t.duckdb")
    yield connection
    connection.close()

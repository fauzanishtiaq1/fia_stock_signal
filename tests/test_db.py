"""Tests for stock_signals.db: schema creation and upsert semantics."""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_signals import db

EXPECTED_TABLES = {
    "universe",
    "prices_daily",
    "xbrl_facts",
    "estimates",
    "news",
    "events",
    "macro",
    "social_posts",
    "scores",
    "picks",
}


def test_schema_creates_all_ten_tables(con):
    assert len(EXPECTED_TABLES) == 10
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert EXPECTED_TABLES <= names


def test_upsert_replaces_on_primary_key(con):
    df = pd.DataFrame(
        {
            "series_id": ["CPI", "GDP"],
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "value": [1.0, 2.0],
        }
    )
    assert db.upsert_df(con, "macro", df) == 2

    changed = pd.DataFrame(
        {
            "series_id": ["CPI"],
            "date": [date(2024, 1, 1)],
            "value": [9.9],
        }
    )
    assert db.upsert_df(con, "macro", changed) == 1

    rows = con.execute(
        "SELECT series_id, value FROM macro ORDER BY series_id"
    ).fetchall()
    assert rows == [("CPI", 9.9), ("GDP", 2.0)]


def test_upsert_missing_optional_columns_become_null(con):
    df = pd.DataFrame({"symbol": ["AAPL"], "name": ["Apple Inc."]})
    assert db.upsert_df(con, "universe", df) == 1

    row = con.execute(
        "SELECT symbol, name, sector, sub_industry, cik, added_at FROM universe"
    ).fetchone()
    assert row == ("AAPL", "Apple Inc.", None, None, None, None)


def test_upsert_empty_df_returns_zero(con):
    assert db.upsert_df(con, "macro", pd.DataFrame()) == 0
    assert con.execute("SELECT count(*) FROM macro").fetchone()[0] == 0

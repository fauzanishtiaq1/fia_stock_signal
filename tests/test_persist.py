"""Tests for stock_signals.persist: parquet export/import round-trips. Offline."""

from __future__ import annotations

from datetime import date

import pandas as pd

from stock_signals import db
from stock_signals.persist import export_tables, import_tables

MACRO = pd.DataFrame(
    {
        "series_id": ["CPI", "CPI", "GDP"],
        "date": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 1, 1)],
        "value": [3.1, 3.2, 27.4],
    }
)

UNIVERSE = pd.DataFrame(
    {
        "symbol": ["AAPL", "MSFT"],
        "name": ["Apple Inc.", "Microsoft"],
        "sector": ["Tech", "Tech"],
        "sub_industry": ["Hardware", "Software"],
        "cik": ["0000320193", "0000789019"],
        "added_at": [date(2024, 1, 1), date(2024, 1, 1)],
    }
)


def _populated_db(path):
    con = db.connect(path)
    db.upsert_df(con, "macro", MACRO)
    db.upsert_df(con, "universe", UNIVERSE)
    return con


def test_export_import_round_trip(tmp_path):
    con_a = _populated_db(tmp_path / "a.duckdb")
    exported = export_tables(con_a, tmp_path / "pq")
    con_a.close()

    assert exported == {"macro": 3, "universe": 2}
    assert (tmp_path / "pq" / "macro.parquet").is_file()
    assert (tmp_path / "pq" / "universe.parquet").is_file()

    con_b = db.connect(tmp_path / "b.duckdb")
    imported = import_tables(con_b, tmp_path / "pq")
    assert imported == {"macro": 3, "universe": 2}

    assert con_b.execute("SELECT count(*) FROM macro").fetchone()[0] == 3
    assert con_b.execute("SELECT count(*) FROM universe").fetchone()[0] == 2
    value = con_b.execute(
        "SELECT value FROM macro WHERE series_id = 'CPI' AND date = DATE '2024-02-01'"
    ).fetchone()[0]
    assert value == 3.2
    name = con_b.execute(
        "SELECT name FROM universe WHERE symbol = 'AAPL'"
    ).fetchone()[0]
    assert name == "Apple Inc."
    con_b.close()


def test_import_missing_dir_returns_empty(tmp_path):
    con = db.connect(tmp_path / "b.duckdb")
    assert import_tables(con, tmp_path / "does-not-exist") == {}
    con.close()


def test_reimport_is_idempotent(tmp_path):
    con_a = _populated_db(tmp_path / "a.duckdb")
    export_tables(con_a, tmp_path / "pq")
    con_a.close()

    con_b = db.connect(tmp_path / "b.duckdb")
    first = import_tables(con_b, tmp_path / "pq")
    second = import_tables(con_b, tmp_path / "pq")
    assert first == second == {"macro": 3, "universe": 2}
    assert con_b.execute("SELECT count(*) FROM macro").fetchone()[0] == 3
    assert con_b.execute("SELECT count(*) FROM universe").fetchone()[0] == 2
    con_b.close()


def test_export_skips_empty_tables(tmp_path):
    con = _populated_db(tmp_path / "a.duckdb")
    exported = export_tables(con, tmp_path / "pq")
    con.close()
    assert "news" not in exported
    assert not (tmp_path / "pq" / "news.parquet").exists()

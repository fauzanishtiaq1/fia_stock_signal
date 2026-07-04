"""DuckDB storage: schema and helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    sub_industry TEXT,
    cik TEXT,
    added_at DATE
);
CREATE TABLE IF NOT EXISTS prices_daily (
    symbol TEXT,
    date DATE,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT,
    source TEXT,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS xbrl_facts (
    cik TEXT,
    tag TEXT,
    unit TEXT,
    period_end DATE,
    fiscal_period TEXT,
    filed DATE,
    form TEXT,
    value DOUBLE,
    PRIMARY KEY (cik, tag, unit, period_end, filed)
);
CREATE TABLE IF NOT EXISTS estimates (
    symbol TEXT,
    date DATE,
    metric TEXT,
    value DOUBLE,
    PRIMARY KEY (symbol, date, metric)
);
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY,
    symbol TEXT,
    published TIMESTAMP,
    title TEXT,
    source TEXT,
    url TEXT
);
CREATE TABLE IF NOT EXISTS events (
    accession TEXT PRIMARY KEY,
    cik TEXT,
    form TEXT,
    items TEXT,
    filed TIMESTAMP,
    title TEXT,
    url TEXT
);
CREATE TABLE IF NOT EXISTS macro (
    series_id TEXT,
    date DATE,
    value DOUBLE,
    PRIMARY KEY (series_id, date)
);
CREATE TABLE IF NOT EXISTS social_posts (
    id TEXT,
    platform TEXT,
    symbol TEXT,
    created TIMESTAMP,
    author TEXT,
    text TEXT,
    sentiment DOUBLE,
    model TEXT,
    PRIMARY KEY (id, symbol)
);
CREATE TABLE IF NOT EXISTS scores (
    run_date DATE,
    horizon TEXT,
    symbol TEXT,
    factor TEXT,
    value DOUBLE,
    pctile DOUBLE,
    PRIMARY KEY (run_date, horizon, symbol, factor)
);
CREATE TABLE IF NOT EXISTS picks (
    run_date DATE,
    horizon TEXT,
    rank INTEGER,
    symbol TEXT,
    composite DOUBLE,
    breakdown TEXT,
    PRIMARY KEY (run_date, horizon, rank)
);
"""


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the DuckDB database and ensure schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    return con


def upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Insert a DataFrame, replacing rows that collide on the table's primary key.

    Column names in df must match table columns (missing columns become NULL).
    Returns number of rows written.
    """
    if df.empty:
        return 0
    cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    for missing in set(cols) - set(df.columns):
        df = df.assign(**{missing: None})
    df = df[cols]
    con.register("_upsert_tmp", df)
    con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _upsert_tmp")
    con.unregister("_upsert_tmp")
    return len(df)

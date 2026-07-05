"""Parquet snapshots of DuckDB tables, so nightly CI runs can commit data to git."""

from __future__ import annotations

from pathlib import Path

import duckdb


def _user_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Names of user tables in the main schema."""
    rows = con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
    ).fetchall()
    return [r[0] for r in rows]


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """Column names of a table, in schema order."""
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]


def export_tables(con: duckdb.DuckDBPyConnection, dir: Path) -> dict[str, int]:
    """Write every non-empty user table to <dir>/<table>.parquet.

    Empty tables are skipped and any existing parquet files for them are left
    alone. Returns {table: rowcount} for the tables exported.
    """
    dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, int] = {}
    for table in _user_tables(con):
        (count,) = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()
        if count == 0:
            continue
        path = str(dir / f"{table}.parquet").replace("'", "''")
        con.execute(f"COPY (SELECT * FROM \"{table}\") TO '{path}' (FORMAT PARQUET)")
        exported[table] = count
    return exported


def import_tables(con: duckdb.DuckDBPyConnection, dir: Path) -> dict[str, int]:
    """Load <dir>/<table>.parquet files into their matching tables.

    Uses INSERT OR REPLACE with an explicit column list, so it is idempotent
    and order-safe. Files without a matching table are skipped. A missing dir
    returns {}. Returns {table: rows in file}.
    """
    if not dir.is_dir():
        return {}
    tables = set(_user_tables(con))
    imported: dict[str, int] = {}
    for file in sorted(dir.glob("*.parquet")):
        table = file.stem
        if table not in tables:
            continue
        cols = ", ".join(f'"{c}"' for c in _table_columns(con, table))
        (count,) = con.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(file)]
        ).fetchone()
        con.execute(
            f'INSERT OR REPLACE INTO "{table}" ({cols}) '
            f"SELECT {cols} FROM read_parquet(?)",
            [str(file)],
        )
        imported[table] = count
    return imported

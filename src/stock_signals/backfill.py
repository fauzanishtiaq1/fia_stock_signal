"""One-off price history backfill into prices_daily.

Usage:
    uv run python -m stock_signals.backfill --source twelvedata --start 2025-06-01
    uv run python -m stock_signals.backfill --source tiingo --limit 40 --db data/dev.duckdb

Resumable: symbols that already have >= --min-rows rows are skipped, so an
interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stock_signals import db
from stock_signals.config import Config, load_config

log = logging.getLogger("backfill")


def _fetcher(source: str, cfg: Config):
    """Return (callable(symbol) -> DataFrame) for the chosen source."""
    if source == "twelvedata":
        from stock_signals.ingest.twelvedata import TwelveDataSource

        src = TwelveDataSource(cfg)
        return lambda s: src.daily_prices(s, outputsize=300)
    if source == "tiingo":
        from stock_signals.ingest.tiingo import TiingoSource

        src = TiingoSource(cfg)
        return lambda s: src.daily_history(s, start=_args.start)
    if source == "fmp":
        from stock_signals.ingest.fmp import FmpSource

        src = FmpSource(cfg)
        return lambda s: src.daily_prices(s, start=_args.start)
    raise SystemExit(f"unknown source: {source}")


def main(argv: list[str] | None = None) -> int:
    global _args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="twelvedata",
                        choices=["twelvedata", "tiingo", "fmp"])
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N universe symbols")
    parser.add_argument("--min-rows", type=int, default=200,
                        help="skip symbols that already have this many rows")
    parser.add_argument("--db", default=None, help="override database path")
    _args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    db_path = Path(_args.db) if _args.db else cfg.db_path
    con = db.connect(db_path)

    symbols = [r[0] for r in
               con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
    if _args.limit:
        symbols = symbols[: _args.limit]
    have = dict(con.execute(
        "SELECT symbol, count(*) FROM prices_daily GROUP BY symbol").fetchall())
    todo = [s for s in symbols if have.get(s, 0) < _args.min_rows]
    log.info("%d symbols total, %d already filled, %d to fetch via %s",
             len(symbols), len(symbols) - len(todo), len(todo), _args.source)

    fetch = _fetcher(_args.source, cfg)
    written = errors = 0
    for i, symbol in enumerate(todo, start=1):
        try:
            rows = db.upsert_df(con, "prices_daily", fetch(symbol))
            written += rows
        except Exception as exc:  # noqa: BLE001 - keep going, report at end
            errors += 1
            log.warning("%s failed: %s", symbol, exc)
        if i % 25 == 0 or i == len(todo):
            log.info("progress %d/%d (%d rows, %d errors)", i, len(todo), written, errors)

    con.close()
    log.info("done: %d rows written, %d symbols failed", written, errors)
    return 0 if (not todo or errors < len(todo)) else 1


if __name__ == "__main__":
    sys.exit(main())

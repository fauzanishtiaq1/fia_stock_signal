"""Smoke test: healthcheck every data source plus the DB and universe loader."""

from __future__ import annotations

import logging
import sys

from stock_signals import db
from stock_signals.config import load_config
from stock_signals.ingest import all_sources

log = logging.getLogger(__name__)


def _print_table(rows: list[tuple[str, str, str]]) -> None:
    """Print an aligned SOURCE | STATUS | DETAIL table."""
    headers = ("SOURCE", "STATUS", "DETAIL")
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(3)
    ]
    fmt = f"{{:<{widths[0]}}} | {{:<{widths[1]}}} | {{:<{widths[2]}}}"
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def main() -> int:
    """Run all healthchecks and print a status table. Exit 1 on any real failure."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()

    rows: list[tuple[str, str, str]] = []
    failed = False

    for source in all_sources(config):
        status = source.healthcheck()
        if not status.available:
            rows.append((status.name, "SKIP-no-key", status.detail))
        elif status.ok:
            rows.append((status.name, "OK", status.detail))
        else:
            rows.append((status.name, "FAIL", status.detail))
            log.warning("source %s failed healthcheck: %s", status.name, status.detail)
            failed = True

    try:
        con = db.connect(config.db_path)
        n_tables = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchone()[0]
        con.close()
        rows.append(("db", "OK", f"{n_tables} tables at {config.db_path}"))
    except Exception as exc:  # noqa: BLE001 - smoke report, not control flow
        rows.append(("db", "FAIL", f"{type(exc).__name__}: {exc}"))
        log.warning("db check failed: %s", exc)
        failed = True

    try:
        from stock_signals.universe import load_universe

        universe = load_universe(config)
        rows.append(("universe", "OK", f"{len(universe)} constituents"))
    except Exception as exc:  # noqa: BLE001 - smoke report, not control flow
        rows.append(("universe", "FAIL", f"{type(exc).__name__}: {exc}"))
        log.warning("universe check failed: %s", exc)
        failed = True

    _print_table(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

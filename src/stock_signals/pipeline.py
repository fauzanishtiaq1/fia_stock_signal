"""Nightly ingestion pipeline: universe, EDGAR events, macro, prices, news."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import duckdb

from stock_signals import db
from stock_signals.config import PROJECT_ROOT, Config, load_config

log = logging.getLogger(__name__)

PRICE_LOOKBACK_DAYS = 400
MACRO_START = "2015-01-01"
NEWS_SYMBOL_LIMIT = 25  # RSS politeness: only the first N universe symbols

StepResult = tuple[int, str]  # (rows written, detail)


def _run_step(
    steps: dict[str, dict[str, Any]], name: str, fn: Callable[[], StepResult]
) -> None:
    """Run one pipeline step, recording ok/rows/detail; never let it raise."""
    try:
        rows, detail = fn()
        steps[name] = {"ok": True, "rows": rows, "detail": detail}
        log.info("step %-8s ok: %d rows (%s)", name, rows, detail)
    except Exception as exc:  # noqa: BLE001 - one step must not kill the run
        steps[name] = {"ok": False, "rows": 0, "detail": f"{type(exc).__name__}: {exc}"}
        log.warning("step %-8s FAILED: %s", name, exc)


def _universe_rows(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str | None]]:
    """(symbol, name) pairs from the universe table, sorted by symbol."""
    return con.execute("SELECT symbol, name FROM universe ORDER BY symbol").fetchall()


def _ingest_events(con: duckdb.DuckDBPyConnection, cfg: Config) -> StepResult:
    """Latest 8-K and Schedule 13D filings from EDGAR into events.

    EDGAR renamed beneficial-ownership form types from "SC 13D" to
    "SCHEDULE 13D" with the structured-XML transition; the old name
    returns an empty feed.
    """
    from stock_signals.ingest.edgar import EdgarSource

    src = EdgarSource(cfg)
    total = 0
    for form in ("8-K", "SCHEDULE 13D", "SCHEDULE 13D/A"):
        total += db.upsert_df(con, "events", src.latest_filings(form))
    return total, "8-K + SCHEDULE 13D filings"


def _ingest_macro(con: duckdb.DuckDBPyConnection, cfg: Config) -> StepResult:
    """FRED default series into macro."""
    from stock_signals.ingest.fred import DEFAULT_SERIES, FredSource

    src = FredSource(cfg)
    total = 0
    errors = 0
    for series_id in DEFAULT_SERIES:
        try:
            total += db.upsert_df(con, "macro", src.series_observations(series_id, start=MACRO_START))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.warning("macro series %s failed: %s", series_id, exc)
    n = len(DEFAULT_SERIES)
    if n and errors == n:
        raise RuntimeError(f"all {n} FRED series failed")
    detail = f"{n} series since {MACRO_START}" + (f", {errors} errors" if errors else "")
    return total, detail


def _ingest_prices(con: duckdb.DuckDBPyConnection, cfg: Config) -> StepResult:
    """FMP daily bars for every universe symbol into prices_daily."""
    from stock_signals.ingest.fmp import FmpSource

    src = FmpSource(cfg)
    symbols = [row[0] for row in _universe_rows(con)]
    start = (date.today() - timedelta(days=PRICE_LOOKBACK_DAYS)).isoformat()
    total = 0
    errors = 0
    for i, symbol in enumerate(symbols, start=1):
        try:
            total += db.upsert_df(con, "prices_daily", src.daily_prices(symbol, start=start))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.warning("prices for %s failed: %s", symbol, exc)
        if i % 50 == 0:
            log.info("prices: %d/%d symbols done", i, len(symbols))
    if symbols and errors == len(symbols):
        raise RuntimeError(f"all {errors} symbols failed")
    detail = f"{len(symbols)} symbols since {start}" + (f", {errors} errors" if errors else "")
    return total, detail


def _ingest_news(con: duckdb.DuckDBPyConnection, cfg: Config) -> StepResult:
    """Google News RSS for the first NEWS_SYMBOL_LIMIT universe symbols into news."""
    from stock_signals.ingest.news_rss import GoogleNewsSource

    src = GoogleNewsSource(cfg)
    pairs = _universe_rows(con)[:NEWS_SYMBOL_LIMIT]
    total = 0
    errors = 0
    for symbol, name in pairs:
        try:
            total += db.upsert_df(con, "news", src.ticker_news(symbol, company_name=name))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.warning("news for %s failed: %s", symbol, exc)
    if pairs and errors == len(pairs):
        raise RuntimeError(f"all {errors} symbols failed")
    detail = f"{len(pairs)} symbols" + (f", {errors} errors" if errors else "")
    return total, detail


def _print_summary(steps: dict[str, dict[str, Any]]) -> None:
    """Print an aligned STEP | STATUS | ROWS | DETAIL table."""
    headers = ("STEP", "STATUS", "ROWS", "DETAIL")
    rows = [
        (name, "OK" if s["ok"] else "FAIL", str(s["rows"]), s["detail"])
        for name, s in steps.items()
    ]
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(4)
    ]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def main() -> int:
    """Run the nightly pipeline. Exit 1 only if setup or universe refresh failed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    steps: dict[str, dict[str, Any]] = {}
    cfg: Config | None = None
    con: duckdb.DuckDBPyConnection | None = None

    # (a) config + db
    def _setup() -> StepResult:
        nonlocal cfg, con
        cfg = load_config()
        con = db.connect(cfg.db_path)
        return 0, f"db at {cfg.db_path}"

    _run_step(steps, "setup", _setup)

    if steps["setup"]["ok"]:
        assert cfg is not None and con is not None

        # (b) restore parquet snapshots committed by previous runs
        def _restore() -> StepResult:
            from stock_signals.persist import import_tables

            pq_dir = cfg.data_dir / "parquet"
            if not pq_dir.is_dir():
                return 0, "no parquet dir"
            counts = import_tables(con, pq_dir)
            return sum(counts.values()), f"{len(counts)} tables restored"

        _run_step(steps, "restore", _restore)

        # (c) universe — always
        def _universe() -> StepResult:
            from stock_signals.universe import refresh_universe

            n = refresh_universe(con, cfg)
            return n, "universe refreshed"

        _run_step(steps, "universe", _universe)

        # (d) EDGAR events
        _run_step(steps, "events", lambda: _ingest_events(con, cfg))

        # (e) macro (FRED)
        if cfg.fred_key:
            _run_step(steps, "macro", lambda: _ingest_macro(con, cfg))
        else:
            steps["macro"] = {"ok": True, "rows": 0, "detail": "skipped (no FRED_API_KEY)"}

        # (f) prices (FMP)
        if cfg.fmp_key:
            _run_step(steps, "prices", lambda: _ingest_prices(con, cfg))
        else:
            steps["prices"] = {"ok": True, "rows": 0, "detail": "skipped (no FMP_API_KEY)"}

        # (g) news RSS
        _run_step(steps, "news", lambda: _ingest_news(con, cfg))

        # (h) factor scores -> scores/picks tables
        def _scores() -> StepResult:
            from stock_signals.factors import compute_and_store

            counts = compute_and_store(con)
            detail = ", ".join(f"{h}:{n}" for h, n in counts.items())
            return sum(counts.values()), f"eligible {detail}"

        _run_step(steps, "scores", _scores)

        # (i) regenerate the static site from the latest picks
        def _site() -> StepResult:
            from stock_signals.sitegen import generate

            path = generate(con)
            return 0, str(path)

        _run_step(steps, "site", _site)

        # (j) persist tables to parquet so CI can commit them back
        def _persist() -> StepResult:
            from stock_signals.persist import export_tables

            counts = export_tables(con, cfg.data_dir / "parquet")
            total = sum(counts.values())
            return total, f"{len(counts)} tables exported ({total} rows)"

        _run_step(steps, "persist", _persist)

        con.close()

    # (i) last_run.json + summary
    payload = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "steps": steps}
    data_dir = cfg.data_dir if cfg is not None else PROJECT_ROOT / "data"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "last_run.json").write_text(json.dumps(payload, indent=2) + "\n")
    except OSError as exc:
        log.warning("could not write last_run.json: %s", exc)

    _print_summary(steps)
    critical_ok = steps["setup"]["ok"] and steps.get("universe", {}).get("ok", False)
    return 0 if critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())

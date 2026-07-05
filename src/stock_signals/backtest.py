"""Cross-sectional backtest harness — the plan's go/no-go gate.

Deliberately a transparent pandas loop, not a backtesting engine (no
backtrader/zipline): at each rebalance date we rank the universe on the
horizon's composite factor, hold the top-N equal-weight until the next
rebalance, and charge a round-trip cost on the traded fraction.  A bottom-N
("sell list") portfolio is tracked the same way so the long-short spread can
be inspected as a diagnostic.

Run: ``python -m stock_signals.backtest --horizon all``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from stock_signals import db
from stock_signals.config import PROJECT_ROOT, load_config

HORIZONS = ("1w", "3m", "1y")

#: Minimum trailing bars required at the first rebalance, per horizon.
_MIN_BARS = {"1w": 6, "3m": 240, "1y": 240}

#: Rebalance periods per year used for annualization (weekly vs monthly).
_PERIODS_PER_YEAR = {"1w": 52, "3m": 12, "1y": 12}

_SHORT_SAMPLE_WARNING = "sample too short for a reliable verdict (<3y)"

# ---------------------------------------------------------------------------
# Factor definitions — DUPLICATED ON PURPOSE. KEEP IN SYNC with
# stock_signals.factors: the backtest must stay a self-contained, auditable
# harness even as the live scoring module evolves.  If you change a
# definition here, mirror it there (and vice versa).
#
# All factors use adj_close per symbol AS OF a date t, i.e. only bars <= t
# (the caller passes ``panel.loc[:t]``).
# ---------------------------------------------------------------------------


def _reversal_5d(window: pd.DataFrame) -> pd.Series:
    """5-day return as of the window's last bar: last / 5-bars-back - 1."""

    def per_symbol(col: pd.Series) -> float:
        s = col.dropna()
        if len(s) < 6:
            return np.nan
        base = s.iloc[-6]
        if base <= 0:
            return np.nan
        return s.iloc[-1] / base - 1.0

    return window.apply(per_symbol)


def _momentum_12_1(window: pd.DataFrame) -> pd.Series:
    """12-1 momentum: bar 21-back / bar 252-back - 1.

    Requires >= 240 bars; if fewer than 253 bars are available the long leg
    falls back to the oldest bar.
    """

    def per_symbol(col: pd.Series) -> float:
        s = col.dropna()
        if len(s) < 240:
            return np.nan
        base = s.iloc[-253] if len(s) >= 253 else s.iloc[0]
        if base <= 0:
            return np.nan
        return s.iloc[-22] / base - 1.0

    return window.apply(per_symbol)


def _trailing_vol(window: pd.DataFrame) -> pd.Series:
    """Std dev of daily returns over the trailing 252 (>= 240) bars."""

    def per_symbol(col: pd.Series) -> float:
        s = col.dropna()
        if len(s) < 240:
            return np.nan
        rets = s.iloc[-252:].pct_change().dropna()
        if len(rets) < 2:
            return np.nan
        return float(rets.std(ddof=1))

    return window.apply(per_symbol)


def _composite(window: pd.DataFrame, horizon: str) -> pd.Series:
    """Cross-sectional composite score (higher = better) as of the last bar.

    - "1w": pct rank of NEGATED 5-day reversal (recent losers score high).
    - "3m": pct rank of 12-1 momentum.
    - "1y": 0.5 * pctrank(momentum_12_1) + 0.5 * pctrank(-trailing vol).
    """
    if horizon == "1w":
        return (-_reversal_5d(window)).rank(pct=True)
    if horizon == "3m":
        return _momentum_12_1(window).rank(pct=True)
    if horizon == "1y":
        mom = _momentum_12_1(window).rank(pct=True)
        low_vol = (-_trailing_vol(window)).rank(pct=True)
        return 0.5 * mom + 0.5 * low_vol
    raise ValueError(f"unknown horizon {horizon!r}; expected one of {HORIZONS}")


# --------------------------- end factor block ------------------------------


def _load_panel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Wide adj_close panel (dates x symbols) from prices_daily."""
    df = con.execute("SELECT symbol, date, adj_close FROM prices_daily").df()
    if df.empty:
        raise ValueError("prices_daily is empty: run the ingestion pipeline first")
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="symbol", values="adj_close").sort_index()


def _rebalance_dates(index: pd.DatetimeIndex, horizon: str) -> pd.DatetimeIndex:
    """Last trading day of each week (W-FRI) for "1w", of each month otherwise."""
    freq = "W-FRI" if horizon == "1w" else "M"
    last_per_period = index.to_series().groupby(index.to_period(freq)).max()
    return pd.DatetimeIndex(last_per_period.values)


def _hold_period(
    panel: pd.DataFrame,
    t: pd.Timestamp,
    t_next: pd.Timestamp,
    members: list[str],
    prev_w: dict[str, float],
    round_trip_cost: float,
) -> tuple[float, float, dict[str, float]]:
    """Net period return, turnover, and new target weights for one leg.

    Equal-weight the members, hold t -> t_next; members with a missing price
    at either end are dropped from the period return.  Cost is charged on the
    traded fraction: return -= turnover * round_trip_cost.
    """
    w_new = {s: 1.0 / len(members) for s in members}
    traded = set(w_new) | set(prev_w)
    turnover = 0.5 * sum(abs(w_new.get(s, 0.0) - prev_w.get(s, 0.0)) for s in traded)
    p0 = panel.loc[t, members].replace(0.0, np.nan)
    p1 = panel.loc[t_next, members]
    rets = (p1 / p0 - 1.0).dropna()
    gross = float(rets.mean()) if not rets.empty else 0.0
    return gross - turnover * round_trip_cost, turnover, w_new


def _leg_metrics(period_returns: list[float], ppy: int) -> dict:
    """Compounded performance metrics for one portfolio leg."""
    r = pd.Series(period_returns, dtype=float)
    if r.empty:
        raise ValueError("no holding periods: need at least two rebalance dates")
    curve = (1.0 + r).cumprod()
    total = float(curve.iloc[-1] - 1.0)
    years = len(r) / ppy
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1.0 else -1.0
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(r.mean()) / std * math.sqrt(ppy) if std > 0 else 0.0
    drawdown = curve / curve.cummax() - 1.0
    return {
        "cagr": cagr,
        "vol": std * math.sqrt(ppy),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "total_return": total,
    }


def _benchmark_metrics(
    panel: pd.DataFrame,
    benchmark: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    years: float,
) -> dict | None:
    """Buy-and-hold benchmark over the evaluation window; None if absent."""
    if benchmark not in panel.columns:
        return None
    b = panel[benchmark].dropna()
    if b.empty:
        return None
    p0, p1 = b.asof(start_date), b.asof(end_date)
    if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
        return None
    total = float(p1 / p0 - 1.0)
    if total > -1.0 and years > 0:
        cagr = float((1.0 + total) ** (1.0 / years) - 1.0)
    else:
        cagr = -1.0
    return {"total_return": total, "cagr": cagr}


def run_backtest(
    con: duckdb.DuckDBPyConnection,
    horizon: str,
    top_n: int = 10,
    cost_bps: float = 10.0,
    start: str | None = None,
    benchmark: str = "SPY",
) -> dict:
    """Backtest one horizon's composite factor; return a metrics dict.

    Longs the top-N by composite at each rebalance, tracks the bottom-N the
    same way for the long-short spread diagnostic, and reports buy-and-hold
    benchmark performance over the same window (None if the benchmark symbol
    is absent from prices_daily — it is never fetched here).
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}; expected one of {HORIZONS}")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    panel = _load_panel(con)
    rankable = panel.drop(columns=[benchmark], errors="ignore")
    if rankable.shape[1] == 0:
        raise ValueError("no symbols to rank after excluding the benchmark")

    rebal = _rebalance_dates(panel.index, horizon)
    min_bars = _MIN_BARS[horizon]
    positions = panel.index.get_indexer(rebal)
    rebal = rebal[positions + 1 >= min_bars]  # enough trailing bars at first rebalance
    if start is not None:
        rebal = rebal[rebal >= pd.Timestamp(start)]
    if len(rebal) < 2:
        raise ValueError(
            f"not enough history for horizon {horizon!r}: need >= {min_bars} trailing "
            "bars before the first rebalance and at least two rebalance dates"
        )

    round_trip_cost = 2.0 * cost_bps / 1e4
    top_rets: list[float] = []
    bot_rets: list[float] = []
    top_turns: list[float] = []
    prev_top: dict[str, float] = {}
    prev_bot: dict[str, float] = {}

    # A benchmark (or any long-history symbol) can stretch the calendar into
    # years before the ranked universe has data; skip rebalance dates where
    # too few symbols are scoreable instead of failing on them.
    min_scoreable = max(2, min(top_n, len(rankable.columns)))
    used: list[pd.Timestamp] = []
    for t, t_next in zip(rebal[:-1], rebal[1:]):
        comp = _composite(rankable.loc[:t], horizon).dropna()
        if len(comp) < min_scoreable:
            continue
        used.extend([t, t_next])
        n = min(top_n, len(comp))
        top_syms = list(comp.nlargest(n).index)
        bot_syms = list(comp.nsmallest(n).index)
        r_top, turn_top, prev_top = _hold_period(
            panel, t, t_next, top_syms, prev_top, round_trip_cost
        )
        r_bot, _, prev_bot = _hold_period(
            panel, t, t_next, bot_syms, prev_bot, round_trip_cost
        )
        top_rets.append(r_top)
        bot_rets.append(r_bot)
        top_turns.append(turn_top)

    if len(top_rets) < 2:
        raise ValueError(
            f"not enough scoreable rebalance periods for horizon {horizon!r} "
            f"(need >= {min_scoreable} rankable symbols per date)"
        )
    ppy = _PERIODS_PER_YEAR[horizon]
    start_date, end_date = used[0], used[-1]
    years_calendar = (end_date - start_date).days / 365.25
    top = _leg_metrics(top_rets, ppy)
    bottom = _leg_metrics(bot_rets, ppy)
    return {
        "horizon": horizon,
        "n_rebalances": len(top_rets),
        "start": str(start_date.date()),
        "end": str(end_date.date()),
        "top": top,
        "bottom": bottom,
        "spread_total_return": top["total_return"] - bottom["total_return"],
        "avg_turnover": float(np.mean(top_turns)),
        "benchmark": _benchmark_metrics(
            panel, benchmark, start_date, end_date, len(top_rets) / ppy
        ),
        "cost_bps": cost_bps,
        "top_n": top_n,
        "warning": _SHORT_SAMPLE_WARNING if years_calendar < 3.0 else None,
    }


def run_all(con: duckdb.DuckDBPyConnection, **kw) -> dict[str, dict]:
    """Run the backtest for all three horizons; keyed by horizon."""
    return {h: run_backtest(con, h, **kw) for h in HORIZONS}


def _json_safe(obj: object) -> object:
    """Recursively replace non-finite floats with None for strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main(argv: list[str] | None = None) -> None:
    """CLI: run the backtest, write JSON results, print a summary table."""
    parser = argparse.ArgumentParser(
        description="Cross-sectional factor backtest (the plan's go/no-go gate)."
    )
    parser.add_argument("--db", type=Path, default=None, help="DuckDB path (default: config db_path)")
    parser.add_argument("--horizon", choices=("all",) + HORIZONS, default="all")
    parser.add_argument("--top", type=int, default=10, help="portfolio size (default 10)")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="one-way cost in bps (default 10)")
    parser.add_argument("--start", default=None, help="earliest rebalance date (YYYY-MM-DD)")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "backtest" / "results.json",
        help="where to write the results dict as JSON",
    )
    args = parser.parse_args(argv)

    db_path = args.db if args.db is not None else load_config().db_path
    con = db.connect(db_path)
    try:
        kw = dict(top_n=args.top, cost_bps=args.cost_bps, start=args.start)
        if args.horizon == "all":
            results = run_all(con, **kw)
        else:
            results = {args.horizon: run_backtest(con, args.horizon, **kw)}
    finally:
        con.close()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(_json_safe(results), indent=2))

    print(
        f"{'horizon':<8} {'period':<24} {'top CAGR':>9} {'bot CAGR':>9} "
        f"{'spread':>8} {'turnover':>9} {'SPY CAGR':>9}  warning"
    )
    for res in results.values():
        bench = res["benchmark"]
        spy = f"{bench['cagr']:.1%}" if bench else "n/a"
        print(
            f"{res['horizon']:<8} "
            f"{res['start'] + ' -> ' + res['end']:<24} "
            f"{res['top']['cagr']:>9.1%} "
            f"{res['bottom']['cagr']:>9.1%} "
            f"{res['spread_total_return']:>8.1%} "
            f"{res['avg_turnover']:>9.2f} "
            f"{spy:>9}  "
            f"{res['warning'] or ''}"
        )
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

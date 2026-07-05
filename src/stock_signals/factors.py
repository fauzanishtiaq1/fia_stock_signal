"""v0 scoring engine: prices-only factors, composites, and top/bottom picks.

Factors (trading-day offsets on each symbol's own adj_close series):

- ``reversal_5d``    last / 5-bars-back - 1 (needs >= 6 bars)
- ``momentum_12_1``  close 21 bars back / close 252 bars back - 1 (needs >= 253
  bars; with >= 240 bars the oldest available bar substitutes the 252-back leg)
- ``low_vol_252``    std of daily pct-change over the last 252 bars (>= 240 bars)

Composites per horizon (higher = stronger buy candidate):

- ``1w``  pctile of negative reversal_5d (recent losers rank high)
- ``3m``  pctile of momentum_12_1
- ``1y``  0.5 * pctile(momentum_12_1) + 0.5 * pctile(negative low_vol_252)

Stored ``pctile`` values are the directional percentiles that feed the
composite (e.g. the 1w reversal_5d pctile is the pct rank of the *negated*
reversal), so each composite is a weighted average of its stored pctiles.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from . import db
from .config import load_config

HORIZONS = ("1w", "3m", "1y")
HORIZON_FACTORS: dict[str, list[str]] = {
    "1w": ["reversal_5d"],
    "3m": ["momentum_12_1"],
    "1y": ["momentum_12_1", "low_vol_252"],
}

REVERSAL_BARS = 5
MOMENTUM_SKIP = 21
MOMENTUM_FULL = 252
MOMENTUM_MIN = 240
STALE_DAYS = 7
N_PICKS = 10


def _price_at(px: pd.DataFrame, bars_back: int) -> pd.Series:
    """adj_close ``bars_back`` trading days before each symbol's last bar."""
    rows = px.loc[px["from_end"] == bars_back, ["symbol", "adj_close"]]
    return rows.set_index("symbol")["adj_close"]


def _raw_factors(px: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol raw factor values from a (symbol, date, adj_close) frame.

    Returns a DataFrame indexed by symbol with one column per factor; symbols
    lacking the required history get NaN for that factor.
    """
    px = px.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    gb = px.groupby("symbol", sort=True)
    px["from_end"] = gb.cumcount(ascending=False)
    px["ret"] = gb["adj_close"].pct_change()
    n_bars = gb.size()

    last = _price_at(px, 0)
    out = pd.DataFrame(index=n_bars.index)

    # Reversal: alignment yields NaN where the 5-back bar does not exist.
    out["reversal_5d"] = last / _price_at(px, REVERSAL_BARS) - 1.0

    # Momentum: 252-back leg, falling back to the oldest bar when >= 240 bars.
    oldest = px.drop_duplicates("symbol", keep="first").set_index("symbol")["adj_close"]
    far = _price_at(px, MOMENTUM_FULL).reindex(n_bars.index)
    far = far.fillna(oldest.where(n_bars >= MOMENTUM_MIN))
    out["momentum_12_1"] = _price_at(px, MOMENTUM_SKIP).reindex(n_bars.index) / far - 1.0

    # Volatility over the last 252 bars (all available when 240 <= n < 252).
    vol = px.loc[px["from_end"] < MOMENTUM_FULL].groupby("symbol")["ret"].std()
    out["low_vol_252"] = vol.where(n_bars >= MOMENTUM_MIN)
    return out


def compute_and_store(
    con: duckdb.DuckDBPyConnection, run_date: date | None = None
) -> dict[str, int]:
    """Compute factors/composites from prices_daily and upsert scores + picks.

    Returns {horizon: eligible symbol count}. Symbols whose latest price is
    more than 7 calendar days older than the global max date are dropped.
    """
    run_date = run_date or date.today()
    px = con.execute("SELECT symbol, date, adj_close FROM prices_daily").df()
    if px.empty:
        return {h: 0 for h in HORIZONS}
    px["date"] = pd.to_datetime(px["date"])
    latest = px.groupby("symbol")["date"].transform("max")
    px = px[latest >= px["date"].max() - pd.Timedelta(days=STALE_DAYS)]

    raw = _raw_factors(px)
    # Directional percentiles: the pct rank that feeds each composite.
    pct = pd.DataFrame(
        {
            "reversal_5d": (-raw["reversal_5d"]).rank(pct=True),
            "momentum_12_1": raw["momentum_12_1"].rank(pct=True),
            "low_vol_252": (-raw["low_vol_252"]).rank(pct=True),
        }
    )
    composites: dict[str, pd.Series] = {
        "1w": pct["reversal_5d"],
        "3m": pct["momentum_12_1"],
        "1y": 0.5 * pct["momentum_12_1"] + 0.5 * pct["low_vol_252"],
    }

    score_frames: list[pd.DataFrame] = []
    pick_frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for horizon in HORIZONS:
        comp = composites[horizon].dropna().sort_values(ascending=False)
        counts[horizon] = len(comp)
        if comp.empty:
            continue
        factors = HORIZON_FACTORS[horizon]
        for factor in factors:
            score_frames.append(
                pd.DataFrame(
                    {
                        "run_date": run_date,
                        "horizon": horizon,
                        "symbol": comp.index,
                        "factor": factor,
                        "value": raw[factor].reindex(comp.index).to_numpy(),
                        "pctile": pct[factor].reindex(comp.index).to_numpy(),
                    }
                )
            )
        score_frames.append(
            pd.DataFrame(
                {
                    "run_date": run_date,
                    "horizon": horizon,
                    "symbol": comp.index,
                    "factor": "composite",
                    "value": comp.to_numpy(),
                    "pctile": comp.rank(pct=True).to_numpy(),
                }
            )
        )

        k = min(N_PICKS, len(comp))
        top, bottom = comp.iloc[:k], comp.iloc[::-1].iloc[:k]
        breakdowns = {
            sym: json.dumps(
                {
                    f: {"value": float(raw.at[sym, f]), "pctile": float(pct.at[sym, f])}
                    for f in factors
                }
            )
            for sym in set(top.index) | set(bottom.index)
        }
        for ranks, side in ((range(1, k + 1), top), (range(-1, -k - 1, -1), bottom)):
            pick_frames.append(
                pd.DataFrame(
                    {
                        "run_date": run_date,
                        "horizon": horizon,
                        "rank": list(ranks),
                        "symbol": side.index,
                        "composite": side.to_numpy(),
                        "breakdown": [breakdowns[s] for s in side.index],
                    }
                )
            )

    if score_frames:
        db.upsert_df(con, "scores", pd.concat(score_frames, ignore_index=True))
    if pick_frames:
        db.upsert_df(con, "picks", pd.concat(pick_frames, ignore_index=True))
    return counts


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``python -m stock_signals.factors``."""
    parser = argparse.ArgumentParser(description="Compute v0 factor scores and picks.")
    parser.add_argument("--db", type=Path, default=None, help="DuckDB path override")
    parser.add_argument(
        "--run-date", type=date.fromisoformat, default=None, help="Run date (YYYY-MM-DD)"
    )
    args = parser.parse_args(argv)
    con = db.connect(args.db or load_config().db_path)
    try:
        counts = compute_and_store(con, run_date=args.run_date)
    finally:
        con.close()
    for horizon in HORIZONS:
        print(f"{horizon}: {counts.get(horizon, 0)} eligible symbols")


if __name__ == "__main__":
    main()

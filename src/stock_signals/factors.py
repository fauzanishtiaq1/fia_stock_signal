"""v0 scoring engine: prices-only factors, composites, and top/bottom picks.

Factors (trading-day offsets on each symbol's own adj_close series):

- ``reversal_5d``    last / 5-bars-back - 1 (needs >= 6 bars)
- ``momentum_12_1``  close 21 bars back / close 252 bars back - 1 (needs >= 253
  bars; with >= 240 bars the oldest available bar substitutes the 252-back leg)
- ``low_vol_252``    std of daily pct-change over the last 252 bars (>= 240 bars)

Fundamental factors for the 1y horizon (from fundamentals.latest_metrics):

- ``value_ey``  earnings yield; its pctile is the earnings-yield pct rank
- ``quality``   mean of the pct ranks of gross_profitability and roe (the
  score is already a [0, 1] rank blend, so value == pctile for this factor)

Social attention factor for the 1w horizon (from the social_posts table):

- ``attention``  attention_spike = mentions in the 24h before the newest post
  divided by max(mean daily mentions over the prior 14 days, 0.5); its pctile
  is the pct rank across symbols with any social mentions. A ``froth`` flag
  (display-only, never part of the composite) marks symbols with
  attention_spike >= 3 and bullish_ratio >= 0.8.

Composites per horizon (higher = stronger buy candidate):

- ``1w``  with social data: 0.7 * pctile(negative reversal_5d)
  + 0.3 * attention component (symbols without mentions get a neutral 0.5);
  with an empty social_posts table: pure pctile of negative reversal_5d
- ``3m``  pctile of momentum_12_1
- ``1y``  full mode (fundamentals coverage >= FUNDAMENTALS_MIN_COVERAGE):
  0.35 * value_ey + 0.35 * quality + 0.15 * pctile(momentum_12_1)
  + 0.15 * pctile(negative low_vol_252); symbols without fundamentals are
  excluded. Preview mode otherwise:
  0.5 * pctile(momentum_12_1) + 0.5 * pctile(negative low_vol_252)

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
HORIZON_FACTORS: dict[str, list[str]] = {  # preview-mode 1y factor set
    "1w": ["reversal_5d"],
    "3m": ["momentum_12_1"],
    "1y": ["momentum_12_1", "low_vol_252"],
}
FULL_1Y_FACTORS = ["value_ey", "quality", "momentum_12_1", "low_vol_252"]
FULL_1Y_WEIGHTS = {"value_ey": 0.35, "quality": 0.35, "momentum_12_1": 0.15, "low_vol_252": 0.15}
FUNDAMENTALS_MIN_COVERAGE = 100  # symbols with fundamentals needed for full 1y mode

REVERSAL_BARS = 5
MOMENTUM_SKIP = 21
MOMENTUM_FULL = 252
MOMENTUM_MIN = 240
STALE_DAYS = 7
N_PICKS = 10

ATTENTION_RECENT_HOURS = 24
ATTENTION_BASELINE_DAYS = 14
ATTENTION_BASELINE_FLOOR = 0.5  # mentions/day; avoids divide-by-tiny spikes
ATTENTION_WEIGHT = 0.3  # 1w composite weight when social data exists
BULLISH_THRESHOLD = 0.05  # sentiment above this counts as bullish
FROTH_SPIKE = 3.0  # attention_spike at/above this ...
FROTH_BULLISH = 0.8  # ... with bullish_ratio at/above this => froth warning


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


def _fundamental_factors(
    con: duckdb.DuckDBPyConnection, symbols: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """value_ey/quality raw values and pctiles for the full 1y composite.

    Returns None (preview mode) unless at least FUNDAMENTALS_MIN_COVERAGE of
    ``symbols`` have earnings_yield, gross_profitability and roe. Ranks are
    computed within that covered set; other symbols get NaN (excluded).
    """
    from . import fundamentals

    metrics = fundamentals.latest_metrics(con)
    if metrics.empty:
        return None
    m = metrics.set_index("symbol").reindex(symbols)
    m = m[["earnings_yield", "gross_profitability", "roe"]].dropna()
    if len(m) < FUNDAMENTALS_MIN_COVERAGE:
        return None
    quality = (m["gross_profitability"].rank(pct=True) + m["roe"].rank(pct=True)) / 2.0
    raw = pd.DataFrame({"value_ey": m["earnings_yield"], "quality": quality})
    pct = pd.DataFrame({"value_ey": m["earnings_yield"].rank(pct=True), "quality": quality})
    return raw.reindex(symbols), pct.reindex(symbols)


def _social_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame | None:
    """Per-symbol attention_spike/bullish_ratio from social_posts, or None.

    Computed directly from the social_posts table (no dependency on the
    sentiment module). attention_spike = mentions in the 24h before the
    newest post, divided by max(mean daily mentions over the prior 14 days,
    ATTENTION_BASELINE_FLOOR) — days with zero mentions count toward the
    mean. bullish_ratio = share of those recent mentions with sentiment >
    BULLISH_THRESHOLD (NaN when none of them carry a sentiment score).
    Returns None when social_posts is empty.
    """
    posts = con.execute("SELECT symbol, created, sentiment FROM social_posts").df()
    if posts.empty:
        return None
    posts["created"] = pd.to_datetime(posts["created"])
    cutoff = posts["created"].max() - pd.Timedelta(hours=ATTENTION_RECENT_HOURS)
    recent = posts[posts["created"] > cutoff]
    prior = posts[
        (posts["created"] <= cutoff)
        & (posts["created"] > cutoff - pd.Timedelta(days=ATTENTION_BASELINE_DAYS))
    ]
    symbols = posts["symbol"].unique()
    mean_daily = (
        prior.groupby("symbol").size().reindex(symbols, fill_value=0)
        / ATTENTION_BASELINE_DAYS
    )
    spike = recent.groupby("symbol").size().reindex(symbols, fill_value=0) / (
        mean_daily.clip(lower=ATTENTION_BASELINE_FLOOR)
    )
    scored = recent.dropna(subset=["sentiment"])
    bullish = (
        (scored["sentiment"] > BULLISH_THRESHOLD)
        .groupby(scored["symbol"])
        .mean()
        .reindex(symbols)
    )
    return pd.DataFrame({"attention_spike": spike, "bullish_ratio": bullish})


def compute_and_store(
    con: duckdb.DuckDBPyConnection, run_date: date | None = None
) -> dict[str, dict[str, int | str | bool]]:
    """Compute factors/composites from prices_daily and upsert scores + picks.

    Returns {horizon: {"eligible": n, "mode": "full" | "preview"}}; only 1y
    can run in preview mode (fundamentals coverage below the threshold). The
    1w entry additionally carries "social": whether social_posts had any
    rows (and thus whether the attention component entered the composite).
    Symbols whose latest price is more than 7 calendar days older than the
    global max date are dropped.
    """
    run_date = run_date or date.today()
    # Join universe so non-universe symbols in prices_daily (e.g. the SPY
    # benchmark) are never ranked or picked.
    px = con.execute(
        "SELECT p.symbol, p.date, p.adj_close FROM prices_daily p "
        "JOIN universe u USING (symbol)"
    ).df()
    if px.empty:
        empty: dict[str, dict[str, int | str | bool]] = {
            h: {"eligible": 0, "mode": "preview" if h == "1y" else "full"} for h in HORIZONS
        }
        empty["1w"]["social"] = False
        return empty
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
    factor_sets = {h: list(f) for h, f in HORIZON_FACTORS.items()}
    modes = {h: "full" for h in HORIZONS}

    # 1w: blend a social-attention component into the reversal rank. NOTE:
    # backtest.py intentionally stays price-only for 1w — social history
    # cannot be backfilled, so the backtest covers the reversal leg only.
    # With no social data at all the composite stays the pure reversal
    # pctile (no 0.7 shrinkage of the spread).
    social = _social_factors(con)
    has_social = social is not None
    if has_social:
        attention = social["attention_spike"].reindex(raw.index)
        bullish = social["bullish_ratio"].reindex(raw.index)
        # Ranked only across symbols with any mentions; the rest are neutral.
        attention_pct = attention.rank(pct=True)
        composite_1w = (1.0 - ATTENTION_WEIGHT) * pct[
            "reversal_5d"
        ] + ATTENTION_WEIGHT * attention_pct.fillna(0.5)
    else:
        attention = attention_pct = bullish = pd.Series(dtype=float)
        composite_1w = pct["reversal_5d"]

    composites: dict[str, pd.Series] = {
        "1w": composite_1w,
        "3m": pct["momentum_12_1"],
    }

    fund = _fundamental_factors(con, raw.index)
    if fund is None:
        modes["1y"] = "preview"
        composites["1y"] = 0.5 * pct["momentum_12_1"] + 0.5 * pct["low_vol_252"]
    else:
        fund_raw, fund_pct = fund
        raw = raw.join(fund_raw)
        pct = pct.join(fund_pct)
        factor_sets["1y"] = FULL_1Y_FACTORS
        composites["1y"] = sum(w * pct[f] for f, w in FULL_1Y_WEIGHTS.items())

    score_frames: list[pd.DataFrame] = []
    pick_frames: list[pd.DataFrame] = []
    counts: dict[str, dict[str, int | str | bool]] = {}
    for horizon in HORIZONS:
        comp = composites[horizon].dropna().sort_values(ascending=False)
        counts[horizon] = {"eligible": len(comp), "mode": modes[horizon]}
        if horizon == "1w":
            counts[horizon]["social"] = has_social
        if comp.empty:
            continue
        factors = factor_sets[horizon]
        social_syms = (
            comp.index.intersection(attention.dropna().index) if horizon == "1w" else []
        )
        if len(social_syms):
            score_frames.append(
                pd.DataFrame(
                    {
                        "run_date": run_date,
                        "horizon": horizon,
                        "symbol": social_syms,
                        "factor": "attention",
                        "value": attention.reindex(social_syms).to_numpy(),
                        "pctile": attention_pct.reindex(social_syms).to_numpy(),
                    }
                )
            )
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

        def _breakdown(sym: str) -> str:
            entry = {
                f: {"value": float(raw.at[sym, f]), "pctile": float(pct.at[sym, f])}
                for f in factors
            }
            if sym in social_syms:
                spike = float(attention[sym])
                entry["attention"] = {
                    "value": spike,
                    "pctile": float(attention_pct[sym]),
                }
                # Display-only crowding warning; never changes the composite.
                froth = spike >= FROTH_SPIKE and (
                    pd.notna(bullish[sym]) and float(bullish[sym]) >= FROTH_BULLISH
                )
                if froth:
                    entry["froth"] = True
            return json.dumps(entry)

        breakdowns = {sym: _breakdown(sym) for sym in set(top.index) | set(bottom.index)}
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
        info = counts.get(horizon, {"eligible": 0, "mode": "preview"})
        print(f"{horizon}: {info['eligible']} eligible symbols ({info['mode']} mode)")


if __name__ == "__main__":
    main()

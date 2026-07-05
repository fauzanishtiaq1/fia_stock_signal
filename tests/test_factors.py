"""Offline tests for the v0 factor engine on synthetic price histories."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_signals import db
from stock_signals.factors import HORIZON_FACTORS, compute_and_store

RUN_DATE = date(2026, 7, 5)
N_DAYS = 300
SYMBOLS = ["STEADY_UP", "CRASHER", "CHOPPY", "FLAT"]


def _synthetic_prices() -> pd.DataFrame:
    """~300 trading days of adj_close for four archetypal symbols."""
    dates = pd.bdate_range(end="2026-07-03", periods=N_DAYS)
    i = np.arange(N_DAYS, dtype=float)

    steady = 100.0 * 1.001**i + 0.05 * np.sin(i)  # monotonic riser, tiny vol
    crasher = np.full(N_DAYS, 100.0)  # flat, then -20% over last 5 days
    crasher[-5:] = 100.0 * np.linspace(0.96, 0.80, 5)
    choppy = 100.0 * (1.0 + 0.08 * np.sin(1.3 * i))  # high vol, flat drift
    flat = np.full(N_DAYS, 100.0)

    frames = [
        pd.DataFrame(
            {
                "symbol": sym,
                "date": [d.date() for d in dates],
                "close": series,
                "adj_close": series,
                "source": "test",
            }
        )
        for sym, series in zip(SYMBOLS, [steady, crasher, choppy, flat])
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def scored_con(con):
    """Connection with synthetic universe + prices loaded and scores computed."""
    db.upsert_df(
        con, "universe", pd.DataFrame({"symbol": SYMBOLS, "name": [f"{s} Inc" for s in SYMBOLS]})
    )
    db.upsert_df(con, "prices_daily", _synthetic_prices())
    counts = compute_and_store(con, run_date=RUN_DATE)
    assert counts == {
        "1w": {"eligible": 4, "mode": "full", "social": False},  # empty social_posts
        "3m": {"eligible": 4, "mode": "full"},
        "1y": {"eligible": 4, "mode": "preview"},  # no fundamentals in this db
    }
    return con


def _rank1(con, horizon: str) -> str:
    return con.execute(
        "SELECT symbol FROM picks WHERE horizon = ? AND rank = 1", [horizon]
    ).fetchone()[0]


def test_3m_buy_rank1_is_steady_up(scored_con):
    assert _rank1(scored_con, "3m") == "STEADY_UP"


def test_1w_buy_rank1_is_crasher(scored_con):
    assert _rank1(scored_con, "1w") == "CRASHER"


def test_1y_composite_steady_up_beats_choppy(scored_con):
    rows = scored_con.execute(
        "SELECT symbol, value FROM scores "
        "WHERE horizon = '1y' AND factor = 'composite' AND symbol IN ('STEADY_UP', 'CHOPPY')"
    ).fetchall()
    comp = dict(rows)
    assert comp["STEADY_UP"] > comp["CHOPPY"]


def test_picks_have_positive_and_negative_ranks(scored_con):
    lo, hi = scored_con.execute("SELECT min(rank), max(rank) FROM picks").fetchone()
    assert lo < 0 < hi
    for horizon in HORIZON_FACTORS:
        ranks = {
            r[0]
            for r in scored_con.execute(
                "SELECT rank FROM picks WHERE horizon = ?", [horizon]
            ).fetchall()
        }
        assert {1, 2, 3, 4, -1, -2, -3, -4} == ranks  # 4 symbols -> 4 up, 4 down


def test_breakdowns_are_json_with_expected_factors(scored_con):
    rows = scored_con.execute("SELECT horizon, breakdown FROM picks").fetchall()
    assert rows
    for horizon, breakdown in rows:
        parsed = json.loads(breakdown)
        assert set(parsed) == set(HORIZON_FACTORS[horizon])
        for entry in parsed.values():
            assert set(entry) == {"value", "pctile"}
            assert 0.0 <= entry["pctile"] <= 1.0


def test_scores_pctiles_within_unit_interval(scored_con):
    lo, hi, n = scored_con.execute(
        "SELECT min(pctile), max(pctile), count(*) FROM scores"
    ).fetchone()
    assert n > 0
    assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------------------------
# 1w social attention factor
# --------------------------------------------------------------------------

ANCHOR = pd.Timestamp("2026-07-04 12:00:00")  # newest post; sets the 24h window


def _social_posts() -> pd.DataFrame:
    """Seed posts: FLAT gets a big bullish spike, CHOPPY a steady trickle.

    FLAT: 20 mentions inside the last 24h, all bullish, zero baseline ->
    attention_spike = 20 / max(0, 0.5) = 40 and bullish_ratio = 1.0 (froth).
    CHOPPY: 2 non-bullish recent mentions on a 1/day baseline over the prior
    14 days -> attention_spike = 2.0, no froth. STEADY_UP/CRASHER: no posts.
    """
    rows = [
        (f"flat-{i}", "reddit", "FLAT", ANCHOR - pd.Timedelta(hours=i), 0.7)
        for i in range(20)
    ]
    rows += [
        ("choppy-r0", "bluesky", "CHOPPY", ANCHOR - pd.Timedelta(hours=2), -0.3),
        ("choppy-r1", "reddit", "CHOPPY", ANCHOR - pd.Timedelta(hours=5), 0.0),
    ]
    rows += [
        (f"choppy-b{k}", "reddit", "CHOPPY", ANCHOR - pd.Timedelta(hours=25 + 24 * k), None)
        for k in range(14)
    ]
    frame = pd.DataFrame(rows, columns=["id", "platform", "symbol", "created", "sentiment"])
    return frame.astype({"sentiment": "float64"})


def test_1w_social_attention_and_froth(scored_con):
    con = scored_con

    def composites_1w():
        return dict(
            con.execute(
                "SELECT symbol, value FROM scores "
                "WHERE horizon = '1w' AND factor = 'composite'"
            ).fetchall()
        )

    def other_composites():
        return con.execute(
            "SELECT horizon, symbol, value FROM scores "
            "WHERE horizon IN ('3m', '1y') AND factor = 'composite' ORDER BY 1, 2"
        ).fetchall()

    # Baseline run (empty social_posts): 1w composite is exactly the pure
    # reversal pctile — no 0.7 shrinkage — and no attention rows exist.
    base = composites_1w()
    reversal_pct = dict(
        con.execute(
            "SELECT symbol, pctile FROM scores "
            "WHERE horizon = '1w' AND factor = 'reversal_5d'"
        ).fetchall()
    )
    assert base == pytest.approx(reversal_pct)
    assert not con.execute("SELECT * FROM scores WHERE factor = 'attention'").fetchall()
    base_other = other_composites()

    db.upsert_df(con, "social_posts", _social_posts())
    counts = compute_and_store(con, run_date=RUN_DATE)
    assert counts["1w"] == {"eligible": 4, "mode": "full", "social": True}

    # Attention score rows exist only for symbols with mentions; the pctile
    # is ranked within that mentioned set (FLAT tops it, CHOPPY is median).
    attn = {
        sym: (value, pctile)
        for sym, value, pctile in con.execute(
            "SELECT symbol, value, pctile FROM scores WHERE factor = 'attention'"
        ).fetchall()
    }
    assert attn == {
        "FLAT": (pytest.approx(40.0), pytest.approx(1.0)),
        "CHOPPY": (pytest.approx(2.0), pytest.approx(0.5)),
    }

    new = composites_1w()
    # The bullish spike boosts FLAT's 1w composite vs the no-social baseline.
    assert new["FLAT"] > base["FLAT"]
    # Symbols without mentions share the neutral 0.5 component, so their
    # ordering relative to each other is unchanged.
    assert (new["CRASHER"] > new["STEADY_UP"]) == (base["CRASHER"] > base["STEADY_UP"])
    # 3m/1y never see social data.
    assert other_composites() == base_other

    breakdowns = {
        sym: json.loads(raw)
        for sym, raw in con.execute(
            "SELECT symbol, breakdown FROM picks WHERE horizon = '1w' AND rank > 0"
        ).fetchall()
    }
    assert breakdowns["FLAT"]["froth"] is True  # spike 40 >= 3, bullish 1.0 >= 0.8
    assert breakdowns["FLAT"]["attention"] == {
        "value": pytest.approx(40.0),
        "pctile": pytest.approx(1.0),
    }
    assert "attention" in breakdowns["CHOPPY"]
    assert "froth" not in breakdowns["CHOPPY"]  # spike 2.0 < 3
    for sym in ("CRASHER", "STEADY_UP"):  # no mentions -> no attention entry
        assert "attention" not in breakdowns[sym]
        assert "froth" not in breakdowns[sym]

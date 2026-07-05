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
        "1w": {"eligible": 4, "mode": "full"},
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

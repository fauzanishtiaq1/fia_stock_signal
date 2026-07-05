"""Offline tests for the cross-sectional backtest harness. Synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_signals import db
from stock_signals.backtest import run_all, run_backtest

N_DAYS = 320
WINNERS = ["WIN1", "WIN2"]
LOSERS = ["LOS1", "LOS2"]
NOISE = ["NSE1", "NSE2", "NSE3", "NSE4"]
BENCHMARK = "SPY"

METRIC_KEYS = {"cagr", "vol", "sharpe", "max_drawdown", "total_return"}
RESULT_KEYS = {
    "horizon",
    "n_rebalances",
    "start",
    "end",
    "top",
    "bottom",
    "spread_total_return",
    "avg_turnover",
    "benchmark",
    "cost_bps",
    "top_n",
    "warning",
}


def _walk(rng: np.random.Generator, drift: float, sigma: float) -> np.ndarray:
    """Geometric-ish price path: 100 * cumprod(1 + drift + noise)."""
    return 100.0 * np.cumprod(1.0 + drift + rng.normal(0.0, sigma, N_DAYS))


@pytest.fixture
def bt_con(con):
    """Temp db seeded with persistent winners, persistent losers, noise, SPY."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=N_DAYS)
    paths = {sym: _walk(rng, 0.002, 0.002) for sym in WINNERS}
    paths |= {sym: _walk(rng, -0.002, 0.002) for sym in LOSERS}
    paths |= {sym: _walk(rng, 0.0, 0.004) for sym in NOISE}
    paths[BENCHMARK] = _walk(rng, 0.0003, 0.001)

    wide = pd.DataFrame(paths, index=dates)
    long_df = wide.reset_index(names="date").melt(
        "date", var_name="symbol", value_name="adj_close"
    )
    long_df["date"] = long_df["date"].dt.date
    long_df["close"] = long_df["adj_close"]
    long_df["source"] = "synthetic"
    db.upsert_df(con, "prices_daily", long_df)
    db.upsert_df(
        con, "universe", pd.DataFrame({"symbol": list(paths), "name": list(paths)})
    )
    return con


def test_3m_momentum_separates_winners_from_losers(bt_con):
    res = run_backtest(bt_con, "3m", top_n=2)
    assert RESULT_KEYS <= set(res)
    assert res["horizon"] == "3m"
    assert res["top"]["total_return"] > res["bottom"]["total_return"]
    assert res["spread_total_return"] > 0
    assert 0.0 <= res["avg_turnover"] <= 1.0
    assert res["benchmark"] is not None
    assert set(res["benchmark"]) == {"total_return", "cagr"}
    assert res["warning"] == "sample too short for a reliable verdict (<3y)"
    # ~320 bars, >=240 trailing bars required, monthly thereafter: a handful.
    assert 2 <= res["n_rebalances"] <= 8


def test_higher_costs_strictly_lower_top_return(bt_con):
    cheap = run_backtest(bt_con, "3m", top_n=2, cost_bps=10.0)
    pricey = run_backtest(bt_con, "3m", top_n=2, cost_bps=100.0)
    assert pricey["top"]["total_return"] < cheap["top"]["total_return"]


def test_1w_runs_and_reports_all_metric_keys(bt_con):
    res = run_backtest(bt_con, "1w", top_n=2)
    assert RESULT_KEYS <= set(res)
    assert set(res["top"]) == METRIC_KEYS
    assert set(res["bottom"]) == METRIC_KEYS
    # weekly rebalances over ~64 weeks
    assert res["n_rebalances"] >= 10


def test_run_all_covers_three_horizons(bt_con):
    results = run_all(bt_con, top_n=2)
    assert set(results) == {"1w", "3m", "1y"}
    for horizon, res in results.items():
        assert res["horizon"] == horizon
        assert RESULT_KEYS <= set(res)


def test_absent_benchmark_reported_as_none(bt_con):
    res = run_backtest(bt_con, "3m", top_n=2, benchmark="NOPE")
    assert res["benchmark"] is None


def test_unknown_horizon_raises(bt_con):
    with pytest.raises(ValueError):
        run_backtest(bt_con, "2d")


def test_empty_db_raises(tmp_path):
    con = db.connect(tmp_path / "empty.duckdb")
    try:
        with pytest.raises(ValueError):
            run_backtest(con, "3m")
    finally:
        con.close()

"""Offline tests for EDGAR fundamentals metrics and the 1y factor modes."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_signals import db, factors
from stock_signals.factors import FULL_1Y_WEIGHTS, compute_and_store
from stock_signals.fundamentals import latest_metrics

RUN_DATE = date(2026, 7, 5)

# Three companies: ALPHA has every tag; BETA exercises the revenue-minus-cost
# gross-profit path, the us-gaap shares fallback and (metrics fixture only)
# zero equity; GAMMA is missing Assets and shares entirely.
CIKS = {"ALPHA": "0000000001", "BETA": "0000000002", "GAMMA": "0000000003"}


def _fact(
    cik: str,
    tag: str,
    value: float,
    end: str = "2023-09-30",
    filed: str = "2023-11-01",
    unit: str = "USD",
    fp: str = "FY",
    form: str = "10-K",
) -> dict:
    return {
        "cik": cik,
        "tag": tag,
        "unit": unit,
        "period_end": end,
        "fiscal_period": fp,
        "filed": filed,
        "form": form,
        "value": value,
    }


def _fact_rows(beta_equity: float = 0.0) -> pd.DataFrame:
    a, b, g = CIKS["ALPHA"], CIKS["BETA"], CIKS["GAMMA"]
    rows = [
        # ALPHA: complete set; superseded/older/quarterly rows must be ignored.
        _fact(a, "GrossProfit", 400.0),
        _fact(a, "Assets", 1000.0),
        _fact(a, "StockholdersEquity", 500.0),
        _fact(a, "EntityCommonStockSharesOutstanding", 100.0, unit="shares"),
        _fact(a, "NetIncomeLoss", 100.0, filed="2024-11-01"),  # re-filed: latest wins
        _fact(a, "NetIncomeLoss", 90.0, filed="2023-11-01"),
        _fact(a, "NetIncomeLoss", 999.0, end="2022-09-30", filed="2022-11-01"),
        _fact(a, "NetIncomeLoss", 9999.0, end="2023-06-30", fp="Q3", form="10-Q"),
        # BETA: no GrossProfit -> revenue - cost; shares via us-gaap fallback.
        _fact(b, "RevenueFromContractWithCustomerExcludingAssessedTax", 1000.0),
        _fact(b, "CostOfGoodsAndServicesSold", 600.0),
        _fact(b, "Assets", 800.0),
        _fact(b, "StockholdersEquity", beta_equity),
        _fact(b, "NetIncomeLoss", 50.0),
        _fact(b, "CommonStockSharesOutstanding", 200.0, unit="shares"),
        # GAMMA: no Assets, no shares tags.
        _fact(g, "NetIncomeLoss", 10.0),
        _fact(g, "StockholdersEquity", 100.0),
        _fact(g, "GrossProfit", 40.0),
    ]
    return pd.DataFrame(rows)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": list(CIKS),
            "name": [f"{s} Inc" for s in CIKS],
            "cik": [CIKS[s] for s in CIKS],
        }
    )


def _spot_prices() -> pd.DataFrame:
    """Two bars per symbol; the later close is the one metrics must use."""
    rows = []
    for symbol, older, latest in [("ALPHA", 15.0, 20.0), ("BETA", 9.0, 5.0), ("GAMMA", 1.0, 2.0)]:
        rows.append({"symbol": symbol, "date": "2026-07-02", "adj_close": older, "source": "test"})
        rows.append({"symbol": symbol, "date": "2026-07-03", "adj_close": latest, "source": "test"})
    return pd.DataFrame(rows)


def _price_history(n_days: int = 260) -> pd.DataFrame:
    """>= 240 bars per symbol so momentum_12_1 and low_vol_252 are defined."""
    dates = pd.bdate_range(end="2026-07-03", periods=n_days)
    i = np.arange(n_days, dtype=float)
    frames = []
    for k, symbol in enumerate(CIKS):
        series = 100.0 * (1.0 + 0.0005 * (k + 1)) ** i + (0.2 + 0.1 * k) * np.sin(i)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": [d.date() for d in dates],
                    "close": series,
                    "adj_close": series,
                    "source": "test",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def metrics_con(con):
    """Fundamentals + minimal prices for hand-checkable latest_metrics values."""
    db.upsert_df(con, "universe", _universe())
    db.upsert_df(con, "prices_daily", _spot_prices())
    db.upsert_df(con, "xbrl_facts", _fact_rows(beta_equity=0.0))
    return con


@pytest.fixture
def full_con(con):
    """Long price history + fundamentals (BETA equity positive) for mode tests."""
    db.upsert_df(con, "universe", _universe())
    db.upsert_df(con, "prices_daily", _price_history())
    db.upsert_df(con, "xbrl_facts", _fact_rows(beta_equity=250.0))
    return con


def _row(df: pd.DataFrame, symbol: str) -> pd.Series:
    return df.set_index("symbol").loc[symbol]


def test_alpha_metrics_hand_computed(metrics_con):
    m = latest_metrics(metrics_con)
    alpha = _row(m, "ALPHA")
    assert alpha["net_income_ttm"] == 100.0  # latest FY, latest filed; not 90/999/9999
    assert alpha["gross_profitability"] == pytest.approx(400.0 / 1000.0)
    assert alpha["roe"] == pytest.approx(100.0 / 500.0)
    assert alpha["earnings_yield"] == pytest.approx(100.0 / (100.0 * 20.0))  # latest close 20
    assert alpha["last_price"] == pytest.approx(20.0)


def test_beta_gross_fallback_shares_fallback_zero_equity(metrics_con):
    beta = _row(latest_metrics(metrics_con), "BETA")
    assert beta["gross_profit_ttm"] == pytest.approx(1000.0 - 600.0)
    assert beta["gross_profitability"] == pytest.approx(400.0 / 800.0)
    assert np.isnan(beta["roe"])  # zero equity -> NaN, not inf
    assert beta["earnings_yield"] == pytest.approx(50.0 / (200.0 * 5.0))


def test_gamma_missing_tags_drop_only_those_metrics(metrics_con):
    gamma = _row(latest_metrics(metrics_con), "GAMMA")
    assert np.isnan(gamma["gross_profitability"])  # no Assets
    assert np.isnan(gamma["earnings_yield"])  # no shares tag
    assert gamma["roe"] == pytest.approx(10.0 / 100.0)


def test_latest_metrics_empty_db(con):
    m = latest_metrics(con)
    assert m.empty
    assert "earnings_yield" in m.columns


def test_1y_preview_mode_below_coverage_threshold(full_con):
    # Fundamentals exist for only 2 symbols: far below the default threshold.
    counts = compute_and_store(full_con, run_date=RUN_DATE)
    assert counts["1y"] == {"eligible": 3, "mode": "preview"}
    assert counts["1w"] == {"eligible": 3, "mode": "full", "social": False}
    factors_1y = {
        r[0]
        for r in full_con.execute(
            "SELECT DISTINCT factor FROM scores WHERE horizon = '1y'"
        ).fetchall()
    }
    assert factors_1y == {"momentum_12_1", "low_vol_252", "composite"}


def test_1y_full_mode_with_lowered_threshold(full_con, monkeypatch):
    monkeypatch.setattr(factors, "FUNDAMENTALS_MIN_COVERAGE", 2)
    counts = compute_and_store(full_con, run_date=RUN_DATE)
    # GAMMA lacks fundamentals -> excluded from 1y but still scored on 1w/3m.
    assert counts["1y"] == {"eligible": 2, "mode": "full"}
    assert counts["3m"] == {"eligible": 3, "mode": "full"}

    scores = full_con.execute(
        "SELECT symbol, factor, value, pctile FROM scores WHERE horizon = '1y'"
    ).df()
    assert set(scores["factor"]) == {
        "value_ey",
        "quality",
        "momentum_12_1",
        "low_vol_252",
        "composite",
    }
    assert "GAMMA" not in set(scores["symbol"])

    # Each stored composite is the weighted average of its stored pctiles.
    pct = scores.pivot(index="symbol", columns="factor", values="pctile")
    comp = scores[scores["factor"] == "composite"].set_index("symbol")["value"]
    for symbol in ("ALPHA", "BETA"):
        expected = sum(w * pct.at[symbol, f] for f, w in FULL_1Y_WEIGHTS.items())
        assert comp[symbol] == pytest.approx(expected)

    breakdowns = full_con.execute(
        "SELECT breakdown FROM picks WHERE horizon = '1y' AND rank = 1"
    ).fetchone()
    import json

    assert set(json.loads(breakdowns[0])) == set(factors.FULL_1Y_FACTORS)

"""EDGAR XBRL fundamentals: targeted ingestion + latest-FY value/quality metrics.

``refresh_fundamentals`` pulls companyfacts for every universe CIK but keeps
only the whitelisted ``TAGS`` on 10-K/10-Q rows since ``min_period_end``, so
xbrl_facts stays small enough to commit as parquet.

``latest_metrics`` reads the most recent annual (10-K, FY) figure per company
and tag — latest FY as a v1 proxy for TTM, no quarter stitching — and derives
gross_profitability, roe and earnings_yield for the 1y factor blend.
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)

# Whitelisted us-gaap/dei concepts. Variants confirmed in a live AAPL
# companyfacts spot check (2026-07-05): AAPL books cost of revenue under
# CostOfGoodsAndServicesSold (it has no CostOfRevenue facts at all) and also
# reports operating cash flow under the ...ContinuingOperations variant.
TAGS: tuple[str, ...] = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "GrossProfit",
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",  # modern variant of CostOfRevenue (AAPL et al.)
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "StockholdersEquity",
    "Assets",
    "Liabilities",
    "CashAndCashEquivalentsAtCarryingValue",
    "EntityCommonStockSharesOutstanding",  # dei
    "CommonStockSharesOutstanding",
    "EarningsPerShareDiluted",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",  # variant
)

_FORMS = ("10-K", "10-Q")  # annual and quarterly filings only
_FACT_PK = ["cik", "tag", "unit", "period_end", "filed"]

METRIC_COLUMNS = [
    "symbol",
    "revenue_ttm",
    "gross_profit_ttm",
    "net_income_ttm",
    "equity",
    "assets",
    "shares",
    "last_price",
    "gross_profitability",
    "roe",
    "earnings_yield",
]


def refresh_fundamentals(
    con: duckdb.DuckDBPyConnection,
    cfg: Config,
    min_period_end: str = "2018-01-01",
    max_symbols: int | None = None,
) -> tuple[int, int]:
    """Ingest whitelisted XBRL facts for every universe CIK into xbrl_facts.

    Returns (symbols_ok, rows written). Per-symbol failures are logged and
    skipped; EdgarSource already rate-limits under SEC's 10 req/s.
    """
    from .ingest.edgar import EdgarSource

    src = EdgarSource(cfg)
    pairs = con.execute(
        "SELECT symbol, cik FROM universe WHERE cik IS NOT NULL AND cik <> '' ORDER BY symbol"
    ).fetchall()
    if max_symbols is not None:
        pairs = pairs[:max_symbols]

    cutoff = pd.Timestamp(min_period_end)
    symbols_ok = 0
    rows = 0
    for i, (symbol, cik) in enumerate(pairs, start=1):
        try:
            facts = src.companyfacts(cik)
            if not facts.empty:
                facts = facts[
                    facts["tag"].isin(TAGS)
                    & (facts["period_end"] >= cutoff)
                    & facts["form"].isin(_FORMS)
                ]
                # companyfacts repeats a fact when it carries both fp=Q4 and
                # fp=FY entries; the PK upsert cannot take intra-batch dupes.
                facts = facts.drop_duplicates(subset=_FACT_PK, keep="last")
            rows += db.upsert_df(con, "xbrl_facts", facts)
            symbols_ok += 1
        except Exception as exc:  # noqa: BLE001 - one company must not kill the refresh
            log.warning("fundamentals for %s (cik %s) failed: %s", symbol, cik, exc)
        if i % 50 == 0:
            log.info("fundamentals: %d/%d symbols done", i, len(pairs))
    return symbols_ok, rows


def _tag_col(wide: pd.DataFrame, tag: str) -> pd.Series:
    """Column for a tag from the pivoted facts, all-NaN when no company has it."""
    if tag in wide.columns:
        return wide[tag]
    return pd.Series(np.nan, index=wide.index)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den with zero/negative denominators mapped to NaN."""
    return num / den.where(den > 0)


def latest_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Latest-FY fundamentals per universe symbol, with derived ratios.

    For each (cik, tag) the most recent annual figure wins (form 10-K,
    fiscal_period FY; latest period_end, then latest filed). Derived columns
    are NaN when an input is missing or a denominator is zero/negative:

    - gross_profitability = (GrossProfit or revenue - cost of revenue) / Assets
    - roe                 = NetIncomeLoss / StockholdersEquity
    - earnings_yield      = NetIncomeLoss / (shares * latest adj_close)
    """
    placeholders = ", ".join("?" for _ in TAGS)
    facts = con.execute(
        f"""
        SELECT cik, tag, period_end, filed, value FROM xbrl_facts
        WHERE form = '10-K' AND fiscal_period = 'FY' AND tag IN ({placeholders})
        """,
        list(TAGS),
    ).df()
    if facts.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    facts = facts.sort_values(["cik", "tag", "period_end", "filed"], kind="mergesort")
    latest = facts.drop_duplicates(subset=["cik", "tag"], keep="last")
    wide = latest.pivot(index="cik", columns="tag", values="value")

    revenue = _tag_col(wide, "RevenueFromContractWithCustomerExcludingAssessedTax")
    revenue = revenue.fillna(_tag_col(wide, "Revenues"))
    cost = _tag_col(wide, "CostOfRevenue").fillna(_tag_col(wide, "CostOfGoodsAndServicesSold"))
    gross = _tag_col(wide, "GrossProfit").fillna(revenue - cost)
    shares = _tag_col(wide, "EntityCommonStockSharesOutstanding")
    shares = shares.fillna(_tag_col(wide, "CommonStockSharesOutstanding"))

    out = pd.DataFrame(
        {
            "cik": wide.index,
            "revenue_ttm": revenue.to_numpy(),
            "gross_profit_ttm": gross.to_numpy(),
            "net_income_ttm": _tag_col(wide, "NetIncomeLoss").to_numpy(),
            "equity": _tag_col(wide, "StockholdersEquity").to_numpy(),
            "assets": _tag_col(wide, "Assets").to_numpy(),
            "shares": shares.to_numpy(),
        }
    )

    uni = con.execute(
        "SELECT symbol, cik FROM universe WHERE cik IS NOT NULL AND cik <> ''"
    ).df()
    px = con.execute(
        "SELECT symbol, adj_close AS last_price FROM prices_daily "
        "QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY date DESC) = 1"
    ).df()
    out = uni.merge(out, on="cik", how="inner").merge(px, on="symbol", how="left")

    out["gross_profitability"] = _safe_div(out["gross_profit_ttm"], out["assets"])
    out["roe"] = _safe_div(out["net_income_ttm"], out["equity"])
    market_cap = out["shares"].where(out["shares"] > 0) * out["last_price"]
    out["earnings_yield"] = _safe_div(out["net_income_ttm"], market_cap)
    return out[METRIC_COLUMNS].sort_values("symbol").reset_index(drop=True)

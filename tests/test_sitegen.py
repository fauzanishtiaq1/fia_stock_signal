"""Offline tests for the static site generator (tmp_path DB only)."""

from __future__ import annotations

import json
import re

from stock_signals import sitegen

RUN_DATE = "2026-07-04"


def _insert_fixture(con) -> None:
    con.execute(
        "INSERT INTO universe (symbol, name, sector) VALUES "
        "('AAPL', 'Apple Inc.', 'Information Technology'), "
        "('XYZ', 'Xyz Corp', 'Industrials'), "
        "('ABC', 'Abc Holdings', 'Financials')"
    )
    top_breakdown = json.dumps(
        {
            "mom12-1": {"value": 0.342, "pctile": 0.87},
            "vol": {"value": 12.5, "pctile": 0.12},
        }
    )
    sell_breakdown = json.dumps({"mom12-1": {"value": -0.21, "pctile": 0.03}})
    for horizon in ("1w", "3m", "1y"):
        rows = [
            (1, "AAPL", 0.91, top_breakdown),
            (-1, "XYZ", 0.08, sell_breakdown),
            (-2, "ABC", 0.12, sell_breakdown),
        ]
        for rank, symbol, composite, breakdown in rows:
            con.execute(
                f"INSERT INTO picks VALUES (DATE '{RUN_DATE}', ?, ?, ?, ?, ?)",
                [horizon, rank, symbol, composite, breakdown],
            )
            con.execute(
                f"INSERT INTO scores VALUES "
                f"(DATE '{RUN_DATE}', ?, ?, 'composite', ?, 0.5)",
                [horizon, symbol, composite],
            )


def test_generate_site(con, tmp_path):
    _insert_fixture(con)

    out = sitegen.generate(con, out_path=tmp_path / "index.html")

    assert out == tmp_path / "index.html"
    assert out.exists()
    html = out.read_text(encoding="utf-8")

    assert "AAPL" in html
    assert "not investment advice" in html.lower()
    for label in ("1 Week", "3 Months", "1 Year"):
        assert label in html

    match = re.search(
        r'<script type="application/json" id="ss-data">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match, "embedded JSON data block missing"
    payload = json.loads(match.group(1))

    assert payload["run_date"] == RUN_DATE
    assert payload["coverage"] == {"scored": 3, "total": 3}
    assert set(payload["horizons"]) == {"1w", "3m", "1y"}
    for horizon in ("1w", "3m", "1y"):
        bucket = payload["horizons"][horizon]
        assert [e["symbol"] for e in bucket["buy"]] == ["AAPL"]
        # Sell list is worst-first: rank -1 before rank -2.
        assert [e["rank"] for e in bucket["sell"]] == [-1, -2]
        assert bucket["sell"][0]["symbol"] == "XYZ"
        top = bucket["buy"][0]
        assert top["name"] == "Apple Inc."
        assert top["sector"] == "Information Technology"
        factors = {f["name"]: f for f in top["factors"]}
        assert factors["mom12-1"]["value"] == 0.342
        assert factors["mom12-1"]["pctile"] == 0.87

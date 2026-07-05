"""Offline tests for the static site generator (tmp_path DB only)."""

from __future__ import annotations

import json
import re

from stock_signals import sitegen

RUN_DATE = "2026-07-04"

RECENT_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000042/aapl-8k.htm"
OUTSIDE_URL = "https://www.sec.gov/Archives/edgar/data/999999/unknown-13d.htm"
STALE_URL = "https://www.sec.gov/Archives/edgar/data/320193/old-8k.htm"


def _insert_fixture(con) -> None:
    con.execute(
        "INSERT INTO universe (symbol, name, sector, cik) VALUES "
        "('AAPL', 'Apple Inc.', 'Information Technology', '0000320193'), "
        "('XYZ', 'Xyz Corp', 'Industrials', '0000789019'), "
        "('ABC', 'Abc Holdings', 'Financials', NULL)"
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


def _insert_events(con) -> None:
    rows = [
        # Unpadded CIK must still join to AAPL's zero-padded universe.cik.
        (
            "acc-recent",
            "320193",
            "8-K",
            "2026-07-04 12:00:00",
            "Apple Inc. 8-K (acquisition agreement)",
            RECENT_URL,
        ),
        # CIK without a universe match: excluded.
        (
            "acc-outside",
            "999999",
            "SCHEDULE 13D",
            "2026-07-03 09:00:00",
            "Unknown Co 13D",
            OUTSIDE_URL,
        ),
        # Older than 7 days relative to the newest filed timestamp: excluded.
        (
            "acc-stale",
            "320193",
            "8-K",
            "2026-06-20 08:00:00",
            "Apple Inc. old 8-K",
            STALE_URL,
        ),
    ]
    for accession, cik, form, filed, title, url in rows:
        con.execute(
            "INSERT INTO events VALUES (?, ?, ?, NULL, CAST(? AS TIMESTAMP), ?, ?)",
            [accession, cik, form, filed, title, url],
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

    # No events rows -> empty list in the payload and no filings section.
    assert payload["events"] == []
    assert "Recent deal filings" not in html


def test_recent_deal_filings(con, tmp_path):
    _insert_fixture(con)
    _insert_events(con)

    out = sitegen.generate(con, out_path=tmp_path / "index.html")
    html = out.read_text(encoding="utf-8")

    match = re.search(
        r'<script type="application/json" id="ss-data">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match, "embedded JSON data block missing"
    payload = json.loads(match.group(1))

    # Exactly the recent in-universe filing, joined across padded/unpadded CIK.
    assert payload["events"] == [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "form": "8-K",
            "filed": "2026-07-04",
            "title": "Apple Inc. 8-K (acquisition agreement)",
            "url": RECENT_URL,
        }
    ]

    # The collapsible section is rendered with a link to the SEC filing.
    assert "Recent deal filings (last 7 days)" in html
    assert '<details class="filings">' in html
    assert f'href="{RECENT_URL}"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener"' in html

    # Out-of-universe and stale filings never reach the page.
    assert OUTSIDE_URL not in html
    assert STALE_URL not in html

"""Static site generator: one self-contained HTML page from picks/scores.

Reads the latest run_date in ``picks``, joins ``universe`` for names and
sectors, and writes a single HTML file with all CSS/JS inline and the data
embedded as a JSON block — it works when opened directly via file://.
Recent deal-relevant SEC filings from ``events`` surface on the 1-Week tab.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import duckdb

from stock_signals.config import PROJECT_ROOT, load_config

#: (horizon key, tab label, caption shown under the tab title)
HORIZONS: tuple[tuple[str, str, str], ...] = (
    (
        "1w",
        "1 Week",
        "Attention watchlist — short-horizon signals are the weakest and "
        "highest-turnover. Backtest pending. Attention = social mention "
        "spikes (Reddit/Bluesky); a 'crowded' badge is a warning, not a "
        "buy signal.",
    ),
    (
        "3m",
        "3 Months",
        "12-1 momentum ranking. Backtest pending.",
    ),
    (
        "1y",
        "1 Year",
        "Preview: momentum + low-volatility blend until fundamentals land "
        "(Phase 1). Backtest pending.",
    ),
)

_COLUMNS = ("Rank", "Symbol", "Name", "Sector", "Score", "Factors")

#: SEC forms considered deal-relevant for the 1-Week tab.
_EVENT_FORMS = ("8-K", "SCHEDULE 13D", "SCHEDULE 13D/A")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _parse_breakdown(raw: str | None) -> tuple[list[dict[str, Any]], bool]:
    """breakdown JSON text -> ([{name, value, pctile}, ...], froth) (tolerant).

    ``froth`` is the display-only crowding warning written by factors.py
    (a bare boolean alongside the per-factor dicts, never a factor itself).
    """
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, dict):
        return [], False
    factors = [
        {"name": name, "value": d.get("value"), "pctile": d.get("pctile")}
        for name, d in parsed.items()
        if isinstance(d, dict)
    ]
    return factors, parsed.get("froth") is True


def _recent_events(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Deal-relevant filings for universe companies from the last 7 days.

    The window is anchored to the newest ``filed`` timestamp in ``events`` so
    stale local data still shows something sensible. ``universe.cik`` is a
    10-digit zero-padded string while ``events.cik`` may be unpadded, so the
    join compares both sides as BIGINT (NULL ciks never match).
    """
    placeholders = ", ".join("?" for _ in _EVENT_FORMS)
    rows = con.execute(
        f"""
        SELECT u.symbol, u.name, e.form, CAST(e.filed AS DATE), e.title, e.url
        FROM events AS e
        JOIN universe AS u
          ON TRY_CAST(e.cik AS BIGINT) = TRY_CAST(u.cik AS BIGINT)
        WHERE TRY_CAST(e.cik AS BIGINT) IS NOT NULL
          AND e.form IN ({placeholders})
          AND e.filed >= (SELECT max(filed) FROM events) - INTERVAL 7 DAY
        ORDER BY e.filed DESC, u.symbol, e.accession
        LIMIT 30
        """,
        list(_EVENT_FORMS),
    ).fetchall()
    return [
        {
            "symbol": symbol,
            "name": name or "",
            "form": form,
            "filed": str(filed),
            "title": title or "",
            "url": url or "",
        }
        for symbol, name, form, filed, title, url in rows
    ]


def _payload(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Collect everything the page needs for the latest run_date in picks."""
    run_date = con.execute("SELECT max(run_date) FROM picks").fetchone()[0]
    if run_date is None:
        raise ValueError("picks table is empty; run the scoring pipeline first")

    rows = con.execute(
        """
        SELECT p.horizon, p.rank, p.symbol, u.name, u.sector, p.composite, p.breakdown
        FROM picks AS p
        LEFT JOIN universe AS u ON u.symbol = p.symbol
        WHERE p.run_date = ?
        """,
        [run_date],
    ).fetchall()
    scored = con.execute(
        "SELECT count(DISTINCT symbol) FROM scores WHERE run_date = ?", [run_date]
    ).fetchone()[0]
    total = con.execute("SELECT count(*) FROM universe").fetchone()[0]

    horizons: dict[str, dict[str, list[dict[str, Any]]]] = {
        hz: {"buy": [], "sell": []} for hz, _, _ in HORIZONS
    }
    for horizon, rank, symbol, name, sector, composite, breakdown in rows:
        factors, froth = _parse_breakdown(breakdown)
        entry = {
            "rank": rank,
            "symbol": symbol,
            "name": name or "",
            "sector": sector or "",
            "composite": composite,
            "factors": factors,
            "froth": froth,
        }
        bucket = horizons.setdefault(horizon, {"buy": [], "sell": []})
        (bucket["buy"] if rank > 0 else bucket["sell"]).append(entry)

    for bucket in horizons.values():
        bucket["buy"].sort(key=lambda e: e["rank"])  # 1 (best) first
        bucket["sell"].sort(key=lambda e: e["rank"], reverse=True)  # -1 (worst) first

    return {
        "run_date": str(run_date),
        "coverage": {"scored": int(scored or 0), "total": int(total or 0)},
        "horizons": horizons,
        "events": _recent_events(con),
    }


# --------------------------------------------------------------------------
# Template pieces (plain strings: braces here never hit .format/f-strings)
# --------------------------------------------------------------------------

_CSS = """
:root {
  --bg: #f6f7f9; --surface: #ffffff; --text: #1b1f24; --muted: #5c6470;
  --border: #e2e5ea; --accent: #2563eb;
  --pos: #15803d; --neg: #b91c1c;
  --banner-bg: #fdf3d7; --banner-text: #6f5205; --banner-border: #eddfad;
  --chip-bg: #edeff3;
  --filing-bg: #e3eafb; --filing-text: #3a5fc0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101215; --surface: #181b20; --text: #e7eaee; --muted: #98a1ac;
    --border: #2a2f37; --accent: #82a7ff;
    --pos: #4ade80; --neg: #f87171;
    --banner-bg: #29230f; --banner-text: #e8cf83; --banner-border: #493e1a;
    --chip-bg: #242932;
    --filing-bg: #212b42; --filing-text: #a3bcf8;
  }
}
* { box-sizing: border-box; }
html, body { max-width: 100%; overflow-x: hidden; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 980px; margin: 0 auto; padding: 2.25rem 1.25rem 3rem; }
header h1 { margin: 0 0 .35rem; font-size: 1.7rem; letter-spacing: -.01em; }
.meta { margin: 0; color: var(--muted); }
.banner {
  margin: 1.4rem 0; padding: .8rem 1.1rem; border-radius: 10px;
  background: var(--banner-bg); color: var(--banner-text);
  border: 1px solid var(--banner-border); font-weight: 600;
}
.tabs {
  display: flex; flex-wrap: wrap; gap: .25rem;
  margin: 1.75rem 0 0; border-bottom: 1px solid var(--border);
}
.tab-btn {
  appearance: none; background: none; border: none; cursor: pointer;
  padding: .6rem .95rem; margin-bottom: -1px;
  font: inherit; font-weight: 600; color: var(--muted);
  border-bottom: 2px solid transparent;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.panel { display: none; padding-top: 1.1rem; }
.panel.active { display: block; }
.caption { margin: .2rem 0 1.2rem; color: var(--muted); max-width: 72ch; }
h2 { margin: 1.6rem 0 .55rem; font-size: 1rem; }
.table-wrap {
  overflow-x: auto; background: var(--surface);
  border: 1px solid var(--border); border-radius: 12px;
}
table { border-collapse: collapse; width: 100%; min-width: 700px; font-size: .92rem; }
th, td {
  text-align: left; padding: .55rem .85rem; vertical-align: top;
  border-bottom: 1px solid var(--border); white-space: nowrap;
}
tbody tr:last-child td { border-bottom: none; }
th {
  color: var(--muted); font-weight: 600; font-size: .74rem;
  text-transform: uppercase; letter-spacing: .05em;
}
td.symbol { font-weight: 600; }
td.score { font-weight: 600; font-variant-numeric: tabular-nums; }
table.buy td.score { color: var(--pos); }
table.sell td.score { color: var(--neg); }
td.factors { white-space: normal; min-width: 260px; }
.chips { display: flex; flex-wrap: wrap; gap: .3rem; }
.chip {
  display: inline-block; padding: .08rem .55rem; border-radius: 999px;
  background: var(--chip-bg); font-size: .78rem; white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.chip.filing { background: var(--filing-bg); color: var(--filing-text); }
.chip.crowded {
  background: var(--banner-bg); color: var(--banner-text);
  border: 1px solid var(--banner-border); font-weight: 600; cursor: help;
}
td.empty { color: var(--muted); font-style: italic; white-space: normal; }
details.filings {
  margin: 0 0 1.3rem; padding: .55rem .95rem;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px;
}
details.filings summary {
  cursor: pointer; font-weight: 600; font-size: .88rem; color: var(--muted);
}
details.filings summary:hover { color: var(--text); }
.filings-list { margin: .55rem 0 .2rem; padding-left: 1.15rem; font-size: .9rem; }
.filings-list li { margin: .3rem 0; overflow-wrap: anywhere; }
.filings-list .filing-symbol { font-weight: 600; }
.filings-list a { color: var(--accent); }
footer {
  margin-top: 2.75rem; padding-top: 1.1rem; border-top: 1px solid var(--border);
  color: var(--muted); font-size: .85rem;
}
footer p { margin: .25rem 0; }
"""

_JS = """
(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("ss-data").textContent);

  function fmtValue(v) {
    if (typeof v !== "number" || !isFinite(v)) return "\\u2014";
    if (Math.abs(v) < 3) {
      return (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
    }
    return v.toFixed(2);
  }

  // attention_spike is a mentions multiplier, not a return: \\u00d7N.N.
  function fmtMult(v) {
    if (typeof v !== "number" || !isFinite(v)) return "\\u2014";
    return "\\u00d7" + v.toFixed(1);
  }

  function fmtPctile(p) {
    if (typeof p !== "number" || !isFinite(p)) return "";
    var x = p <= 1 ? p * 100 : p;
    x = Math.max(0, Math.min(99, Math.round(x)));
    return "p" + x;
  }

  function chip(f) {
    var span = document.createElement("span");
    span.className = "chip";
    var val = f.name === "attention" ? fmtMult(f.value) : fmtValue(f.value);
    var pt = fmtPctile(f.pctile);
    span.textContent = f.name + " " + val + (pt ? " \\u00b7 " + pt : "");
    return span;
  }

  function crowdedChip() {
    var span = document.createElement("span");
    span.className = "chip crowded";
    span.textContent = "crowded";
    span.title = "High mention spike with extreme bullishness \\u2014 " +
      "historically a contrarian signal";
    return span;
  }

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function shortDate(iso) {
    var p = String(iso || "").split("-");
    var m = parseInt(p[1], 10);
    var d = parseInt(p[2], 10);
    if (!(m >= 1 && m <= 12) || !(d >= 1)) return iso || "";
    return MONTHS[m - 1] + " " + d;
  }

  function fill(tbody, rows, filings) {
    if (!rows.length) {
      var tr = document.createElement("tr");
      var td = document.createElement("td");
      td.colSpan = 6;
      td.className = "empty";
      td.textContent = "No picks for this horizon yet.";
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      function cell(text, cls) {
        var td = document.createElement("td");
        if (cls) td.className = cls;
        td.textContent = text;
        tr.appendChild(td);
      }
      cell(String(r.rank));
      cell(r.symbol, "symbol");
      cell(r.name || "\\u2014");
      cell(r.sector || "\\u2014");
      cell((r.composite * 100).toFixed(1), "score");
      var td = document.createElement("td");
      td.className = "factors";
      var chips = document.createElement("div");
      chips.className = "chips";
      (r.factors || []).forEach(function (f) { chips.appendChild(chip(f)); });
      if (r.froth) chips.appendChild(crowdedChip());
      var ev = filings ? filings[r.symbol] : null;
      if (ev) {
        var flag = document.createElement("span");
        flag.className = "chip filing";
        flag.textContent = ev.form + " \\u00b7 " + shortDate(ev.filed);
        chips.appendChild(flag);
      }
      td.appendChild(chips);
      tr.appendChild(td);
      tbody.appendChild(tr);
    });
  }

  var latestFiling = {};
  (data.events || []).forEach(function (e) {
    if (!latestFiling[e.symbol]) latestFiling[e.symbol] = e; // newest first
  });

  document.querySelectorAll("table[data-horizon]").forEach(function (t) {
    var hz = t.getAttribute("data-horizon");
    var h = data.horizons[hz] || { buy: [], sell: [] };
    fill(
      t.querySelector("tbody"),
      h[t.getAttribute("data-list")] || [],
      hz === "1w" ? latestFiling : null
    );
  });

  var buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      buttons.forEach(function (b) {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });
      document.querySelectorAll(".panel").forEach(function (p) {
        p.classList.toggle("active", p.id === "panel-" + btn.getAttribute("data-tab"));
      });
    });
  });
})();
"""


# --------------------------------------------------------------------------
# HTML assembly
# --------------------------------------------------------------------------


def _tabs_html() -> str:
    parts: list[str] = []
    for i, (hz, label, _) in enumerate(HORIZONS):
        active = " active" if i == 0 else ""
        selected = "true" if i == 0 else "false"
        parts.append(
            f'<button class="tab-btn{active}" id="tab-{hz}" data-tab="{hz}" '
            f'role="tab" aria-selected="{selected}" aria-controls="panel-{hz}">'
            f"{label}</button>"
        )
    return "\n      ".join(parts)


def _table_html(hz: str, kind: str, heading: str) -> str:
    head = "".join(f"<th>{c}</th>" for c in _COLUMNS)
    return f"""
      <h2>{heading}</h2>
      <div class="table-wrap">
        <table class="{kind}" data-horizon="{hz}" data-list="{kind}">
          <thead><tr>{head}</tr></thead>
          <tbody></tbody>
        </table>
      </div>"""


def _filings_html(events: list[dict[str, Any]]) -> str:
    """Collapsible list of recent deal filings for the 1-Week tab ("" if none)."""
    if not events:
        return ""
    items: list[str] = []
    for ev in events:
        url = ev["url"]
        text = escape(ev["title"] or url or ev["form"])
        if url:
            href = escape(url, quote=True)
            text = f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'
        items.append(
            f'<li><span class="filing-symbol">{escape(ev["symbol"])}</span>'
            f' &mdash; {escape(ev["form"])} &mdash; {escape(ev["filed"])}'
            f" &mdash; {text}</li>"
        )
    body = "\n          ".join(items)
    return f"""
      <details class="filings">
        <summary>Recent deal filings (last 7 days)</summary>
        <ul class="filings-list">
          {body}
        </ul>
      </details>"""


def _panels_html(filings: str) -> str:
    parts: list[str] = []
    for i, (hz, _, caption) in enumerate(HORIZONS):
        active = " active" if i == 0 else ""
        extras = filings if hz == "1w" else ""
        parts.append(
            f"""
    <section class="panel{active}" id="panel-{hz}" role="tabpanel" aria-labelledby="tab-{hz}">
      <p class="caption">{caption}</p>{extras}{_table_html(hz, "buy", "Buy candidates")}{_table_html(hz, "sell", "Sell candidates")}
    </section>"""
        )
    return "".join(parts)


def _render(payload: dict[str, Any], data_json: str, generated_at: str) -> str:
    cov = payload["coverage"]
    coverage_line = f"{cov['scored']} of {cov['total']} symbols scored"
    panels = _panels_html(_filings_html(payload.get("events", [])))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Stock Signals</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Stock Signals</h1>
      <p class="meta">Run date {payload["run_date"]} &middot; {coverage_line}</p>
    </header>
    <div class="banner" role="note">Personal research tool — not investment advice.</div>
    <nav class="tabs" role="tablist" aria-label="Horizon">
      {_tabs_html()}
    </nav>
{panels}
    <noscript><p class="caption">Enable JavaScript to see the ranked tables.</p></noscript>
    <footer>
      <p>Generated {generated_at}.</p>
      <p>Data sources: EDGAR, FMP, Twelve Data, Tiingo, FRED, Google News.</p>
      <p>Phase 0 — ingestion and preview rankings only; factor backtests land in Phase 1.</p>
    </footer>
  </div>
  <script type="application/json" id="ss-data">{data_json}</script>
  <script>{_JS}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate(con: duckdb.DuckDBPyConnection, out_path: Path | None = None) -> Path:
    """Write the self-contained site for the latest run_date; return its path."""
    if out_path is None:
        out_path = PROJECT_ROOT / "data" / "site" / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _payload(con)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # "</" must not appear inside the inline <script> block; "<\/" is valid JSON.
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    out_path.write_text(_render(payload, data_json, generated_at), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m stock_signals.sitegen",
        description="Generate the static Stock Signals site from the picks tables.",
    )
    parser.add_argument(
        "--db", type=Path, default=None, help="path to the DuckDB database"
    )
    args = parser.parse_args(argv)

    from stock_signals import db

    con = db.connect(args.db or load_config().db_path)
    try:
        out = generate(con)
    finally:
        con.close()
    print(out)


if __name__ == "__main__":
    main()

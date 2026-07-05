"""Static site generator: one self-contained HTML page from picks/scores.

Reads the latest run_date in ``picks``, joins ``universe`` for names and
sectors, and writes a single HTML file with all CSS/JS inline and the data
embedded as a JSON block — it works when opened directly via file://.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
        "highest-turnover. Backtest pending.",
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


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _parse_breakdown(raw: str | None) -> list[dict[str, Any]]:
    """breakdown JSON text -> [{name, value, pctile}, ...] (tolerant)."""
    try:
        parsed = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, dict):
        return []
    return [
        {"name": name, "value": d.get("value"), "pctile": d.get("pctile")}
        for name, d in parsed.items()
        if isinstance(d, dict)
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
        hz: {"top": [], "avoid": []} for hz, _, _ in HORIZONS
    }
    for horizon, rank, symbol, name, sector, composite, breakdown in rows:
        entry = {
            "rank": rank,
            "symbol": symbol,
            "name": name or "",
            "sector": sector or "",
            "composite": composite,
            "factors": _parse_breakdown(breakdown),
        }
        bucket = horizons.setdefault(horizon, {"top": [], "avoid": []})
        (bucket["top"] if rank > 0 else bucket["avoid"]).append(entry)

    for bucket in horizons.values():
        bucket["top"].sort(key=lambda e: e["rank"])  # 1 (best) first
        bucket["avoid"].sort(key=lambda e: e["rank"], reverse=True)  # -1 (worst) first

    return {
        "run_date": str(run_date),
        "coverage": {"scored": int(scored or 0), "total": int(total or 0)},
        "horizons": horizons,
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
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101215; --surface: #181b20; --text: #e7eaee; --muted: #98a1ac;
    --border: #2a2f37; --accent: #82a7ff;
    --pos: #4ade80; --neg: #f87171;
    --banner-bg: #29230f; --banner-text: #e8cf83; --banner-border: #493e1a;
    --chip-bg: #242932;
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
table.top td.score { color: var(--pos); }
table.avoid td.score { color: var(--neg); }
td.factors { white-space: normal; min-width: 260px; }
.chips { display: flex; flex-wrap: wrap; gap: .3rem; }
.chip {
  display: inline-block; padding: .08rem .55rem; border-radius: 999px;
  background: var(--chip-bg); font-size: .78rem; white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
td.empty { color: var(--muted); font-style: italic; white-space: normal; }
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

  function fmtPctile(p) {
    if (typeof p !== "number" || !isFinite(p)) return "";
    var x = p <= 1 ? p * 100 : p;
    x = Math.max(0, Math.min(99, Math.round(x)));
    return "p" + x;
  }

  function chip(f) {
    var span = document.createElement("span");
    span.className = "chip";
    var pt = fmtPctile(f.pctile);
    span.textContent = f.name + " " + fmtValue(f.value) + (pt ? " \\u00b7 " + pt : "");
    return span;
  }

  function fill(tbody, rows) {
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
      td.appendChild(chips);
      tr.appendChild(td);
      tbody.appendChild(tr);
    });
  }

  document.querySelectorAll("table[data-horizon]").forEach(function (t) {
    var h = data.horizons[t.getAttribute("data-horizon")] || { top: [], avoid: [] };
    fill(t.querySelector("tbody"), h[t.getAttribute("data-list")] || []);
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


def _panels_html() -> str:
    parts: list[str] = []
    for i, (hz, _, caption) in enumerate(HORIZONS):
        active = " active" if i == 0 else ""
        parts.append(
            f"""
    <section class="panel{active}" id="panel-{hz}" role="tabpanel" aria-labelledby="tab-{hz}">
      <p class="caption">{caption}</p>{_table_html(hz, "top", "Top candidates")}{_table_html(hz, "avoid", "Avoid / weakest")}
    </section>"""
        )
    return "".join(parts)


def _render(payload: dict[str, Any], data_json: str, generated_at: str) -> str:
    cov = payload["coverage"]
    coverage_line = f"{cov['scored']} of {cov['total']} symbols scored"
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
{_panels_html()}
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

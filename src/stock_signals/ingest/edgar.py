"""SEC EDGAR adapter: tickers, XBRL company facts, full-text search, latest filings."""

from __future__ import annotations

import re
from typing import Any

import feedparser
import pandas as pd

from stock_signals.ingest.base import Source

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
_FULLTEXT_URL = "https://efts.sec.gov/LATEST/search-index"
_LATEST_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form_type}&company=&dateb=&owner=include"
    "&count={count}&output=atom"
)

_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}")
_CIK_IN_LINK_RE = re.compile(r"/edgar/data/(\d+)")
_ITEM_RE = re.compile(r"Item\s+\d+\.\d+", re.IGNORECASE)

_FACT_COLS = ["cik", "tag", "unit", "period_end", "fiscal_period", "filed", "form", "value"]
_SEARCH_COLS = ["accession", "cik", "form", "filed", "title", "url"]
_EVENT_COLS = ["accession", "cik", "form", "items", "filed", "title", "url"]


def _pad_cik(cik: Any) -> str:
    """Normalize a CIK to a 10-digit zero-padded string ('' if no digits)."""
    digits = re.sub(r"\D", "", str(cik))
    return digits.zfill(10) if digits else ""


def _naive_ts(value: Any) -> pd.Timestamp | None:
    """Parse a timestamp, dropping timezone info; None when unparseable."""
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.tz_localize(None)


def _index_url(cik: str, accession: str) -> str:
    """Build the filing-index URL on www.sec.gov for an accession number."""
    if not accession:
        return ""
    cik_part = cik.lstrip("0") or "0"
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_part}/"
        f"{accession.replace('-', '')}/{accession}-index.htm"
    )


class EdgarSource(Source):
    """SEC EDGAR (keyless; requires declared User-Agent, set by base from config)."""

    name = "edgar"
    key_attr = None
    min_interval = 0.12  # stay under SEC's 10 requests/second

    def company_tickers(self) -> pd.DataFrame:
        """CIK/ticker/name mapping for all EDGAR registrants."""
        data = self._get(_TICKERS_URL).json()
        rows = []
        entries = data.values() if isinstance(data, dict) else data
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cik = _pad_cik(entry.get("cik_str", ""))
            symbol = entry.get("ticker")
            if not cik or not symbol:
                continue
            rows.append({"cik": cik, "symbol": symbol, "name": entry.get("title", "")})
        return pd.DataFrame(rows, columns=["cik", "symbol", "name"])

    def companyfacts(self, cik: str) -> pd.DataFrame:
        """All XBRL facts for a company, shaped for the xbrl_facts table."""
        cik10 = _pad_cik(cik)
        data = self._get(_COMPANYFACTS_URL.format(cik10=cik10)).json()
        rows = []
        facts = data.get("facts", {}) if isinstance(data, dict) else {}
        for taxonomy in facts.values():  # us-gaap, dei, ...
            if not isinstance(taxonomy, dict):
                continue
            for tag, tag_data in taxonomy.items():
                units = tag_data.get("units", {}) if isinstance(tag_data, dict) else {}
                for unit, entries in units.items():
                    if not isinstance(entries, list):
                        continue
                    for entry in entries:
                        if not isinstance(entry, dict) or entry.get("val") is None:
                            continue
                        rows.append(
                            {
                                "cik": cik10,
                                "tag": tag,
                                "unit": unit,
                                "period_end": entry.get("end"),
                                "fiscal_period": entry.get("fp"),
                                "filed": entry.get("filed"),
                                "form": entry.get("form"),
                                "value": entry.get("val"),
                            }
                        )
        df = pd.DataFrame(rows, columns=_FACT_COLS)
        if df.empty:
            return df
        df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # period_end/filed/value are required (part of the PK or the point of the row)
        return df.dropna(subset=["period_end", "filed", "value"]).reset_index(drop=True)

    def fulltext_search(
        self,
        query: str,
        forms: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> pd.DataFrame:
        """Full-text search over EDGAR filings (EFTS); one row per hit."""
        params: dict[str, str] = {"q": query}
        if forms:
            params["forms"] = forms
        if date_from or date_to:
            params["dateRange"] = "custom"
            if date_from:
                params["startdt"] = date_from
            if date_to:
                params["enddt"] = date_to
        data = self._get(_FULLTEXT_URL, params=params).json()

        hits = data.get("hits", {}) if isinstance(data, dict) else {}
        if isinstance(hits, dict):
            hits = hits.get("hits", [])
        if not isinstance(hits, list):
            hits = []

        rows = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            source = hit.get("_source") or {}
            if not isinstance(source, dict):
                source = {}
            accession = source.get("adsh") or str(hit.get("_id", "")).split(":", 1)[0]
            match = _ACCESSION_RE.search(str(accession))
            if not match:
                continue
            accession = match.group(0)

            ciks = source.get("ciks")
            cik = _pad_cik(ciks[0]) if isinstance(ciks, list) and ciks else ""

            form = source.get("form") or source.get("file_type") or ""
            if not form:
                root_forms = source.get("root_forms")
                if isinstance(root_forms, list) and root_forms:
                    form = root_forms[0]

            names = source.get("display_names")
            title = names[0] if isinstance(names, list) and names else ""

            rows.append(
                {
                    "accession": accession,
                    "cik": cik,
                    "form": str(form),
                    "filed": source.get("file_date"),
                    "title": str(title),
                    "url": _index_url(cik, accession),
                }
            )
        df = pd.DataFrame(rows, columns=_SEARCH_COLS)
        if not df.empty:
            df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        return df

    def latest_filings(self, form_type: str = "8-K", count: int = 100) -> pd.DataFrame:
        """Most recent filings of a form type (Atom feed), shaped for the events table."""
        resp = self._get(_LATEST_URL.format(form_type=form_type, count=count))
        feed = feedparser.parse(resp.content)

        rows = []
        for entry in feed.entries:
            link = entry.get("link", "")
            match = _ACCESSION_RE.search(entry.get("id", "")) or _ACCESSION_RE.search(link)
            if not match:
                continue
            accession = match.group(0)

            cik_match = _CIK_IN_LINK_RE.search(link)
            cik = _pad_cik(cik_match.group(1)) if cik_match else ""

            title = entry.get("title", "") or ""
            form = title.split(" - ", 1)[0].strip() if " - " in title else form_type

            summary = entry.get("summary", "") or ""
            items = ",".join(_ITEM_RE.findall(summary))

            rows.append(
                {
                    "accession": accession,
                    "cik": cik,
                    "form": form,
                    "items": items,
                    "filed": _naive_ts(entry.get("updated")),
                    "title": title,
                    "url": link or _index_url(cik, accession),
                }
            )
        return pd.DataFrame(rows, columns=_EVENT_COLS)

    def _healthcheck_call(self) -> str:
        df = self.company_tickers()
        return f"{len(df)} ticker mappings"

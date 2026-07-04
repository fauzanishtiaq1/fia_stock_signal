# stock-signals

Personal 3-horizon stock ranking site:

- **1 week** — attention watchlist (news / filings / social spikes)
- **3 months** — workhorse ranking (momentum, revisions, events)
- **1 year** — fundamentals (valuation, quality, growth)

> **Not financial advice.** This is a personal research tool for my own use only.
> Nothing here is a recommendation to buy or sell anything.

## Setup

```sh
uv sync
cp .env.example .env   # then fill in your API keys
uv run stock-signals-smoke     # verify every source / db / universe
uv run stock-signals-nightly   # run the full ingestion pipeline
```

## Data sources

| Source | What | Key required |
| --- | --- | --- |
| SEC EDGAR | Filings (8-K, SCHEDULE 13D), XBRL fundamentals | No (User-Agent only) |
| FMP | Daily prices, estimates | Yes (paid) |
| Twelve Data | Prices (backup) | Yes |
| Finnhub | Prices / estimates (backup) | Yes |
| Tiingo | Prices (backup) | Yes |
| FRED | Macro series | Yes (free) |
| Google News RSS | Ticker news | No |
| Reddit / Bluesky | Social sentiment (phase 3) | Yes |

## Architecture

Nightly GitHub Action → DuckDB/parquet → factor scores → static JSON site.

## Roadmap

- **Phase 0** — plumbing: ingestion, schema, smoke test, nightly action (this repo, now)
- **Phase 1** — factor scores + backtest
- **Phase 2** — static site
- **Phase 3** — sentiment + event signals
- **Phase 3b** — optional X (Twitter) sampling

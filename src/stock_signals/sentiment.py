"""Social-post sentiment: ticker extraction, scoring, ingestion, mention stats.

Turns raw social posts (reddit, bluesky, ...) into per-(post, ticker) rows in
the ``social_posts`` table, then aggregates them into mention/spike/bullishness
stats that downstream factors can consume.
"""

from __future__ import annotations

import re
from datetime import timedelta

import duckdb
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stock_signals import db

# Common English words, finance jargon, and WSB slang that collide with real
# ticker symbols. Bare-word mentions of these are ignored; explicit cashtags
# ($ALL, $IT, ...) still count because the "$" signals intent.
BLACKLIST: frozenset[str] = frozenset(
    {
        "A", "AH", "AI", "ALL", "AM", "AN", "AND", "ANY", "ARE", "AT", "ATH",
        "BE", "BIG", "BY", "CAN", "CASH", "CEO", "CFO", "COO", "DD", "DO",
        "DOJ", "EDIT", "EOD", "EPS", "ETF", "EV", "FD", "FOMO", "FOR", "GAIN",
        "GDP", "GO", "HAS", "HE", "HOLD", "HUGE", "IMO", "IN", "IPO", "IRS",
        "IS", "IT", "KEY", "LFG", "LOL", "LOSS", "LOVE", "LOW", "ME", "MOON",
        "MY", "NEW", "NICE", "NO", "NOT", "NOW", "OF", "OK", "ON", "ONE",
        "OR", "OUT", "PLAY", "PM", "PSA", "PUMP", "REAL", "RH", "RIDE", "ROI",
        "SAFE", "SEC", "SO", "TA", "THE", "TLDR", "TO", "UK", "UP", "US",
        "USA", "WE", "WSB", "WTF", "YOLO", "YOU",
    }
)

# Posts listing more tickers than this are treated as ticker-spam and ignored.
MAX_TICKERS_PER_POST = 8

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z])")
_BARE_RE = re.compile(r"\b[A-Z]{2,5}\b")

SUPPORTED_MODELS = ("vader",)

_analyzer: SentimentIntensityAnalyzer | None = None


def extract_tickers(text: str, valid_symbols: set[str]) -> set[str]:
    """Extract ticker symbols mentioned in a post.

    Cashtags ($aapl, $ALL; case-insensitive) are accepted whenever the symbol
    is in valid_symbols, even for blacklisted words -- the "$" is explicit
    intent. Bare tokens must appear in uppercase in the original text, be in
    valid_symbols, and survive BLACKLIST. Text inside URLs never matches, and
    posts naming more than MAX_TICKERS_PER_POST tickers return an empty set
    (spam guard).
    """
    if not text:
        return set()
    cleaned = _URL_RE.sub(" ", text)

    found: set[str] = set()
    for match in _CASHTAG_RE.finditer(cleaned):
        symbol = match.group(1).upper()
        if symbol in valid_symbols:
            found.add(symbol)
    for token in _BARE_RE.findall(cleaned):
        if token in valid_symbols and token not in BLACKLIST:
            found.add(token)

    if len(found) > MAX_TICKERS_PER_POST:
        return set()
    return found


def _vader() -> SentimentIntensityAnalyzer:
    """Module-level lazy singleton: the analyzer loads its lexicon once."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_texts(texts: list[str], model: str = "vader") -> list[float]:
    """Score each text in [-1, 1] (negative..positive) with the given model."""
    if model == "vader":
        analyzer = _vader()
        return [analyzer.polarity_scores(t or "")["compound"] for t in texts]
    # ---------------------------------------------------------------------
    # EXTENSION POINT: add new sentiment models here and to SUPPORTED_MODELS.
    # Each branch should return one float per text, in [-1, 1].
    # ---------------------------------------------------------------------
    if model == "finbert":
        raise NotImplementedError(
            "finbert scoring is not implemented yet: install `transformers` "
            "and load ProsusAI/finbert here, mapping its positive/negative "
            "probabilities to a [-1, 1] score."
        )
    if model == "haiku":
        raise NotImplementedError(
            "haiku scoring is not implemented yet: requires ANTHROPIC_API_KEY "
            "and an Anthropic client batching texts through a Claude Haiku "
            "prompt that returns a [-1, 1] score."
        )
    raise ValueError(
        f"unknown sentiment model {model!r}; supported models: "
        f"{', '.join(SUPPORTED_MODELS)}"
    )


def process_posts(
    con: duckdb.DuckDBPyConnection,
    posts: pd.DataFrame,
    valid_symbols: set[str],
    model: str = "vader",
) -> int:
    """Extract tickers and sentiment from posts and upsert into social_posts.

    posts must have columns [id, platform, created, author, text]. Posts
    mentioning no valid ticker are skipped; each remaining post is scored
    once and written as one row per (post, ticker), so re-running is
    idempotent on the (id, symbol) primary key. Returns rows written.
    """
    if posts is None or len(posts) == 0:
        return 0

    kept: list[tuple[object, set[str]]] = []
    texts: list[str] = []
    for post in posts.itertuples(index=False):
        tickers = extract_tickers(post.text or "", valid_symbols)
        if not tickers:
            continue
        kept.append((post, tickers))
        texts.append(post.text or "")
    if not kept:
        return 0

    scores = score_texts(texts, model=model)
    rows = [
        {
            "id": post.id,
            "platform": post.platform,
            "symbol": symbol,
            "created": post.created,
            "author": post.author,
            "text": post.text,
            "sentiment": score,
            "model": model,
        }
        for (post, tickers), score in zip(kept, scores)
        for symbol in sorted(tickers)
    ]
    return db.upsert_df(con, "social_posts", pd.DataFrame(rows))


def mention_stats(
    con: duckdb.DuckDBPyConnection, baseline_days: int = 14
) -> pd.DataFrame:
    """Per-symbol mention counts, spike ratio, and bullishness.

    "Recent" is the 24h window ending at max(created) in social_posts; the
    baseline is the mean DAILY mention count over the baseline_days window
    just before it (days with zero mentions count as zero, i.e. total
    mentions / baseline_days). spike is mentions_recent divided by
    max(baseline_daily_avg, 0.5); the floor keeps low-baseline symbols from
    exploding.
    bullish_ratio is the fraction of recent mentions with sentiment > 0.05
    among those with non-null sentiment (NaN if there are none).

    Returns a DataFrame with columns [symbol, mentions_recent,
    baseline_daily_avg, spike, bullish_ratio]; empty table -> empty frame.
    """
    max_created = con.execute("SELECT max(created) FROM social_posts").fetchone()[0]
    if max_created is None:
        return pd.DataFrame(
            {
                "symbol": pd.Series(dtype="str"),
                "mentions_recent": pd.Series(dtype="int64"),
                "baseline_daily_avg": pd.Series(dtype="float64"),
                "spike": pd.Series(dtype="float64"),
                "bullish_ratio": pd.Series(dtype="float64"),
            }
        )
    recent_start = max_created - timedelta(hours=24)
    baseline_start = recent_start - timedelta(days=baseline_days)
    return con.execute(
        """
        WITH recent AS (
            SELECT
                symbol,
                count(*) AS mentions_recent,
                avg(CASE WHEN sentiment > 0.05 THEN 1.0 ELSE 0.0 END)
                    FILTER (WHERE sentiment IS NOT NULL) AS bullish_ratio
            FROM social_posts
            WHERE created > ?
            GROUP BY symbol
        ),
        baseline AS (
            SELECT symbol, count(*)::DOUBLE / ? AS baseline_daily_avg
            FROM social_posts
            WHERE created > ? AND created <= ?
            GROUP BY symbol
        )
        SELECT
            coalesce(r.symbol, b.symbol) AS symbol,
            coalesce(r.mentions_recent, 0) AS mentions_recent,
            coalesce(b.baseline_daily_avg, 0.0) AS baseline_daily_avg,
            coalesce(r.mentions_recent, 0)
                / greatest(coalesce(b.baseline_daily_avg, 0.0), 0.5) AS spike,
            r.bullish_ratio
        FROM recent r
        FULL OUTER JOIN baseline b ON r.symbol = b.symbol
        ORDER BY spike DESC, symbol
        """,
        [recent_start, baseline_days, baseline_start, recent_start],
    ).df()

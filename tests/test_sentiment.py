"""Tests for stock_signals.sentiment: extraction, scoring, ingestion, stats."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd
import pytest

from stock_signals import sentiment

T0 = datetime(2026, 7, 4, 12, 0, 0)

# 4 clearly-positive + 2 clearly-negative recent texts (VADER-verified signs).
RECENT_TEXTS = [
    "GME to the moon, great gains",
    "GME diamond hands, love this play",
    "GME this is amazing, big win",
    "GME strong buy, excellent momentum",
    "GME terrible awful loss incoming",
    "GME dumping hard, sad loss",
]


# ---------------------------------------------------------------- extraction


def test_cashtag_extracted():
    assert sentiment.extract_tickers("$AAPL to the moon", {"AAPL"}) == {"AAPL"}


def test_blacklisted_bare_words_ignored():
    got = sentiment.extract_tickers(
        "I think IT and ALL are overvalued", {"IT", "ALL", "AAPL"}
    )
    assert got == set()


def test_cashtag_overrides_blacklist():
    assert sentiment.extract_tickers("$ALL is a real insurer", {"ALL"}) == {"ALL"}


def test_bare_uppercase_word_extracted():
    assert sentiment.extract_tickers("buy AAPL now", {"AAPL"}) == {"AAPL"}


def test_lowercase_bare_word_not_extracted():
    assert sentiment.extract_tickers("buy aapl now", {"AAPL"}) == set()


def test_ticker_inside_url_not_extracted():
    text = "chart here https://finance.example.com/AAPL/history looks good"
    assert sentiment.extract_tickers(text, {"AAPL"}) == set()


def test_unknown_symbols_not_extracted():
    assert sentiment.extract_tickers("buy ZZZZ now", {"AAPL"}) == set()


def test_ticker_spam_returns_empty():
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD"]
    text = " ".join(f"${s}" for s in symbols)
    assert sentiment.extract_tickers(text, set(symbols)) == set()
    # Exactly 8 is still allowed.
    text8 = " ".join(f"${s}" for s in symbols[:8])
    assert sentiment.extract_tickers(text8, set(symbols)) == set(symbols[:8])


def test_empty_text_extracts_nothing():
    assert sentiment.extract_tickers("", {"AAPL"}) == set()


# ------------------------------------------------------------------- scoring


def test_positive_text_scores_positive():
    assert sentiment.score_texts(["to the moon 🚀 huge gains"])[0] > 0


def test_negative_text_scores_negative():
    assert sentiment.score_texts(["bankruptcy incoming, terrible loss"])[0] < 0


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError, match="vader"):
        sentiment.score_texts(["hello"], model="gpt")


def test_planned_models_raise_not_implemented():
    with pytest.raises(NotImplementedError, match="transformers"):
        sentiment.score_texts(["hello"], model="finbert")
    with pytest.raises(NotImplementedError, match="ANTHROPIC_API_KEY"):
        sentiment.score_texts(["hello"], model="haiku")


# ------------------------------------------------- process_posts + aggregates


def _posts_df() -> pd.DataFrame:
    """Baseline ~1/day for GME over 3 days, then 6 mentions in the last 24h."""
    rows = []
    # Baseline: one $GME post per day, all before the recent 24h window.
    for i, hours_ago in enumerate((30, 54, 78)):
        rows.append(
            {
                "id": f"base{i}",
                "platform": "reddit",
                "created": T0 - timedelta(hours=hours_ago),
                "author": f"u{i}",
                "text": "$GME steady as she goes",
            }
        )
    # Recent: six mentions inside the last 24h with planted sentiment signs.
    for i, text in enumerate(RECENT_TEXTS):
        rows.append(
            {
                "id": f"recent{i}",
                "platform": "reddit",
                "created": T0 - timedelta(hours=i),
                "author": f"r{i}",
                "text": text,
            }
        )
    # Noise that must be skipped: no ticker at all.
    rows.append(
        {
            "id": "noise0",
            "platform": "bluesky",
            "created": T0,
            "author": "n0",
            "text": "nothing to see here folks",
        }
    )
    return pd.DataFrame(rows)


def test_process_posts_writes_one_row_per_post_ticker(con):
    written = sentiment.process_posts(con, _posts_df(), {"GME"})
    assert written == 9  # 3 baseline + 6 recent; the no-ticker post skipped
    count, models = con.execute(
        "SELECT count(*), count(DISTINCT model) FROM social_posts"
    ).fetchone()
    assert (count, models) == (9, 1)
    assert con.execute("SELECT DISTINCT model FROM social_posts").fetchone()[0] == (
        "vader"
    )


def test_process_posts_is_idempotent(con):
    posts = _posts_df()
    sentiment.process_posts(con, posts, {"GME"})
    sentiment.process_posts(con, posts, {"GME"})
    assert con.execute("SELECT count(*) FROM social_posts").fetchone()[0] == 9


def test_process_posts_empty_and_none_safe(con):
    assert sentiment.process_posts(con, None, {"GME"}) == 0
    assert sentiment.process_posts(con, pd.DataFrame(), {"GME"}) == 0
    assert con.execute("SELECT count(*) FROM social_posts").fetchone()[0] == 0


def test_multi_ticker_post_gets_one_row_per_ticker_same_score(con):
    posts = pd.DataFrame(
        [
            {
                "id": "p1",
                "platform": "reddit",
                "created": T0,
                "author": "a",
                "text": "$GME and $AMC both look great",
            }
        ]
    )
    assert sentiment.process_posts(con, posts, {"GME", "AMC"}) == 2
    rows = con.execute(
        "SELECT symbol, sentiment FROM social_posts WHERE id = 'p1' ORDER BY symbol"
    ).fetchall()
    assert [r[0] for r in rows] == ["AMC", "GME"]
    assert rows[0][1] == rows[1][1]


def test_mention_stats_spike_and_bullish_ratio(con):
    sentiment.process_posts(con, _posts_df(), {"GME"})
    stats = sentiment.mention_stats(con, baseline_days=3)
    assert list(stats.columns) == [
        "symbol",
        "mentions_recent",
        "baseline_daily_avg",
        "spike",
        "bullish_ratio",
    ]
    row = stats.loc[stats["symbol"] == "GME"].iloc[0]
    assert row["mentions_recent"] == 6
    assert row["baseline_daily_avg"] == pytest.approx(1.0)  # 3 posts / 3 days
    assert row["spike"] == pytest.approx(6.0)
    assert row["spike"] > 3
    # Expected ratio derives from the same scorer the pipeline used.
    scores = sentiment.score_texts(RECENT_TEXTS)
    expected = sum(s > 0.05 for s in scores) / len(scores)
    assert 0 < expected < 1  # test is meaningless if all texts agree
    assert row["bullish_ratio"] == pytest.approx(expected)


def test_mention_stats_empty_table(con):
    stats = sentiment.mention_stats(con)
    assert stats.empty
    assert list(stats.columns) == [
        "symbol",
        "mentions_recent",
        "baseline_daily_avg",
        "spike",
        "bullish_ratio",
    ]


def test_mention_stats_null_sentiment_gives_nan_ratio(con):
    from stock_signals import db

    df = pd.DataFrame(
        [
            {
                "id": "x1",
                "platform": "reddit",
                "symbol": "GME",
                "created": T0,
                "author": "a",
                "text": "$GME",
                "sentiment": None,
                "model": None,
            }
        ]
    )
    db.upsert_df(con, "social_posts", df)
    stats = sentiment.mention_stats(con)
    row = stats.iloc[0]
    assert row["mentions_recent"] == 1
    assert math.isnan(row["bullish_ratio"])

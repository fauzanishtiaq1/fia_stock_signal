"""Tests for stock_signals.universe symbol normalization."""

from __future__ import annotations

import pytest


def test_normalize_symbol():
    universe = pytest.importorskip("stock_signals.universe")
    assert universe.normalize_symbol("BRK.B") == "BRK-B"
    assert universe.normalize_symbol(" aapl ") == "AAPL"
    assert universe.normalize_symbol("BF.B") == "BF-B"

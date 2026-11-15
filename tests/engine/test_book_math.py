"""Test deterministici di `engine/book_math.py` (T2-A)."""

from __future__ import annotations

import math

import pytest

from engine.book_math import (
    available_shares,
    best_price,
    clean_levels,
    fill_for_budget,
    simulate_fill,
    vwap_for_shares,
)
from engine_helpers import levels, make_book, raw_level

THREE_LEVELS = [(0.40, 10.0), (0.50, 10.0), (0.60, 10.0)]


def test_single_level_vwap_equals_level_price():
    book = make_book("yes", asks=levels((0.40, 100.0)))
    fill = simulate_fill(book, "BUY", 50.0)
    assert fill.fully_filled
    assert fill.filled_shares == 50.0
    assert fill.notional_usdc == pytest.approx(20.0)
    assert fill.vwap == pytest.approx(0.40)
    assert fill.levels_consumed == 1


def test_vwap_walks_multiple_ask_levels():
    book = make_book("yes", asks=levels(*THREE_LEVELS))
    fill = simulate_fill(book, "BUY", 25.0)
    # 10 @ 0.40 + 10 @ 0.50 + 5 @ 0.60 = 12.0 USDC
    assert fill.fully_filled
    assert fill.notional_usdc == pytest.approx(12.0)
    assert fill.vwap == pytest.approx(0.48)
    assert fill.worst_price == pytest.approx(0.60)
    assert fill.levels_consumed == 3
    # Il VWAP reale e peggiore del best ask: niente liquidita infinita sul top.
    assert fill.vwap > best_price(book, "BUY")


def test_empty_book_returns_empty_fill_without_raising():
    book = make_book("yes")
    fill = simulate_fill(book, "BUY", 10.0)
    assert not fill.fully_filled
    assert fill.filled_shares == 0.0
    assert fill.notional_usdc == 0.0
    assert fill.vwap is None
    assert fill.worst_price is None
    assert vwap_for_shares(book, "BUY", 10.0) is None
    assert best_price(book, "BUY") is None
    assert available_shares(book, "BUY") == 0.0


def test_none_book_is_handled():
    assert simulate_fill(None, "BUY", 10.0).filled_shares == 0.0
    assert best_price(None, "SELL") is None
    assert available_shares(None, "BUY") == 0.0


def test_insufficient_liquidity_gives_partial_fill():
    book = make_book("yes", asks=levels(*THREE_LEVELS))
    fill = simulate_fill(book, "BUY", 50.0)
    assert not fill.fully_filled
    assert fill.requested_shares == 50.0
    assert fill.filled_shares == pytest.approx(30.0)
    assert fill.notional_usdc == pytest.approx(15.0)
    assert fill.vwap == pytest.approx(0.50)
    # Il VWAP "per l'intera size" non esiste se la size non e eseguibile.
    assert vwap_for_shares(book, "BUY", 50.0) is None


def test_corrupted_levels_are_skipped():
    asks = [
        raw_level(float("nan"), 10.0),
        raw_level(0.0, 10.0),  # ask == 0
        raw_level(0.30, float("nan")),
        raw_level(0.30, 0.0),
        raw_level(0.30, -5.0),
        raw_level(None, 10.0),
        raw_level(1.5, 10.0),
        raw_level(float("inf"), 10.0),
        raw_level(0.50, 10.0),  # unico livello valido
    ]
    book = make_book("yes").model_copy(update={"asks": asks})
    fill = simulate_fill(book, "BUY", 5.0)
    assert fill.fully_filled
    assert fill.vwap == pytest.approx(0.50)
    assert available_shares(book, "BUY") == pytest.approx(10.0)


def test_book_with_only_corrupted_levels_behaves_as_empty():
    book = make_book("yes").model_copy(
        update={"asks": [raw_level(float("nan"), 10.0), raw_level(0.0, 10.0)]}
    )
    assert simulate_fill(book, "BUY", 1.0).vwap is None
    assert best_price(book, "BUY") is None


def test_unsorted_asks_are_sorted_defensively():
    book = make_book("yes", asks=levels((0.60, 10.0), (0.40, 10.0), (0.50, 10.0)))
    assert best_price(book, "BUY") == pytest.approx(0.40)
    assert simulate_fill(book, "BUY", 10.0).vwap == pytest.approx(0.40)


def test_sell_walks_bids_from_highest():
    book = make_book("yes", bids=levels((0.50, 10.0), (0.55, 10.0)))
    fill = simulate_fill(book, "SELL", 15.0)
    # 10 @ 0.55 + 5 @ 0.50 = 8.0 USDC incassati
    assert fill.fully_filled
    assert fill.notional_usdc == pytest.approx(8.0)
    assert fill.vwap == pytest.approx(8.0 / 15.0)
    assert fill.worst_price == pytest.approx(0.50)
    assert best_price(book, "SELL") == pytest.approx(0.55)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf"), None, "10", True])
def test_invalid_target_shares_never_raise(target):
    book = make_book("yes", asks=levels(*THREE_LEVELS))
    fill = simulate_fill(book, "BUY", target)
    assert fill.filled_shares == 0.0
    assert fill.vwap is None
    assert not fill.fully_filled


def test_limit_price_stops_the_walk():
    book = make_book("yes", asks=levels(*THREE_LEVELS))
    fill = simulate_fill(book, "BUY", 30.0, limit_price=0.50)
    assert not fill.fully_filled
    assert fill.filled_shares == pytest.approx(20.0)
    assert fill.worst_price == pytest.approx(0.50)
    assert available_shares(book, "BUY", limit_price=0.50) == pytest.approx(20.0)
    assert available_shares(book, "BUY", limit_price=0.10) == 0.0
    assert simulate_fill(book, "BUY", 5.0, limit_price=float("nan")).filled_shares == 0.0


def test_exact_fill_is_robust_to_float_drift():
    book = make_book("yes", asks=levels((0.40, 0.1), (0.50, 0.2)))
    fill = simulate_fill(book, "BUY", 0.3)  # 0.1 + 0.2 != 0.3 in floating point
    assert fill.fully_filled
    assert fill.filled_shares == 0.3
    assert 0.0 < fill.vwap <= 1.0


def test_vwap_never_exceeds_one():
    book = make_book("yes", asks=levels((1.0, 0.1), (1.0, 0.2), (1.0, 0.7)))
    fill = simulate_fill(book, "BUY", 1.0)
    assert fill.fully_filled
    assert fill.vwap <= 1.0


def test_fill_for_budget_walks_levels():
    book = make_book("yes", asks=levels((0.40, 10.0), (0.50, 10.0)))
    fill = fill_for_budget(book, 6.5)
    # 10 @ 0.40 = 4.0 USDC, restano 2.5 USDC -> 5 shares @ 0.50
    assert fill.filled_shares == pytest.approx(15.0)
    assert fill.notional_usdc == pytest.approx(6.5)
    assert fill.worst_price == pytest.approx(0.50)


def test_fill_for_budget_is_capped_by_liquidity_and_limit():
    book = make_book("yes", asks=levels((0.40, 10.0), (0.50, 10.0)))
    all_in = fill_for_budget(book, 100.0)
    assert all_in.filled_shares == pytest.approx(20.0)
    assert all_in.notional_usdc == pytest.approx(9.0)
    limited = fill_for_budget(book, 100.0, limit_price=0.45)
    assert limited.filled_shares == pytest.approx(10.0)
    assert limited.notional_usdc == pytest.approx(4.0)


@pytest.mark.parametrize("budget", [0.0, -5.0, float("nan"), float("inf"), None])
def test_fill_for_budget_invalid_budget_never_raises(budget):
    book = make_book("yes", asks=levels((0.40, 10.0)))
    fill = fill_for_budget(book, budget)
    assert fill.filled_shares == 0.0
    assert fill.vwap is None


def test_clean_levels_orders_both_sides():
    raw = levels((0.50, 1.0), (0.40, 2.0), (0.60, 3.0))
    assert [price for price, _ in clean_levels(raw, descending=False)] == [0.40, 0.50, 0.60]
    assert [price for price, _ in clean_levels(raw, descending=True)] == [0.60, 0.50, 0.40]
    assert clean_levels(None, descending=False) == []


def test_results_are_finite_numbers():
    book = make_book("yes", asks=levels(*THREE_LEVELS))
    fill = simulate_fill(book, "BUY", 25.0)
    assert all(math.isfinite(v) for v in (fill.filled_shares, fill.notional_usdc, fill.vwap))

"""Test deterministici di `engine/combinatorial_arb.py` (T2-B)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from engine.combinatorial_arb import evaluate_group
from engine_helpers import T0, levels, make_book, make_group, raw_level


def binary_books(yes_ask: float = 0.48, no_ask: float = 0.49, size: float = 100.0):
    return {
        "yes": make_book("yes", asks=levels((yes_ask, size))),
        "no": make_book("no", asks=levels((no_ask, size))),
    }


def test_binary_yes_no_arbitrage_emits_balanced_signal():
    signal = evaluate_group(make_group("yes", "no"), binary_books(), 50.0, now=T0)
    assert signal is not None
    assert signal.strategy == "COMBINATORIAL_ARB"
    assert signal.event_id == "evt-1"
    assert signal.created_at == T0
    # costo per set 0.48 + 0.49 = 0.97 -> edge 0.03, costo totale 50 * 0.97
    assert signal.expected_net_edge == pytest.approx(0.03)
    assert signal.total_cost_usdc == pytest.approx(48.5)
    assert [leg.token_id for leg in signal.legs] == ["yes", "no"]
    assert all(leg.side == "BUY" for leg in signal.legs)
    assert {leg.target_shares for leg in signal.legs} == {50.0}  # bilanciato
    assert signal.legs[0].expected_vwap == pytest.approx(0.48)
    assert signal.legs[1].expected_vwap == pytest.approx(0.49)


def test_three_outcome_group_arbitrage():
    books = {
        "a": make_book("a", asks=levels((0.30, 100.0))),
        "b": make_book("b", asks=levels((0.30, 100.0))),
        "c": make_book("c", asks=levels((0.35, 100.0))),
    }
    signal = evaluate_group(make_group("a", "b", "c"), books, 20.0, now=T0)
    assert signal is not None
    assert len(signal.legs) == 3
    assert signal.expected_net_edge == pytest.approx(0.05)
    assert signal.total_cost_usdc == pytest.approx(19.0)


def test_sum_of_asks_equal_to_one_is_not_an_opportunity():
    books = binary_books(0.50, 0.50)
    assert evaluate_group(make_group("yes", "no"), books, 10.0, min_net_edge=0.0, now=T0) is None


def test_cost_exactly_at_threshold_is_rejected():
    # 0.49 + 0.495 = 0.985 == 1.0 - 0.015: serve una disuguaglianza STRETTA.
    books = binary_books(0.49, 0.495)
    assert evaluate_group(make_group("yes", "no"), books, 10.0, min_net_edge=0.015, now=T0) is None


def test_sum_above_one_is_rejected():
    assert evaluate_group(make_group("yes", "no"), binary_books(0.55, 0.50), 10.0, now=T0) is None


def test_top_of_book_arb_killed_by_depth():
    # Il best ask promette 0.90 per set, ma solo per 5 shares: a 50 shares il VWAP sale a 1.17.
    thin = levels((0.45, 5.0), (0.60, 100.0))
    books = {"yes": make_book("yes", asks=thin), "no": make_book("no", asks=thin)}
    group = make_group("yes", "no")
    assert evaluate_group(group, books, 50.0, now=T0) is None
    small = evaluate_group(group, books, 5.0, now=T0)
    assert small is not None
    assert small.expected_net_edge == pytest.approx(0.10)


def test_multi_level_vwap_and_max_acceptable_price():
    books = {
        "yes": make_book("yes", asks=levels((0.40, 10.0), (0.44, 10.0))),
        "no": make_book("no", asks=levels((0.50, 100.0))),
    }
    signal = evaluate_group(make_group("yes", "no"), books, 20.0, now=T0)
    assert signal is not None
    yes_leg, no_leg = signal.legs
    assert yes_leg.expected_vwap == pytest.approx(0.42)
    assert yes_leg.max_acceptable_price == pytest.approx(0.44)  # livello peggiore camminato
    assert no_leg.max_acceptable_price == pytest.approx(0.50)
    assert signal.expected_net_edge == pytest.approx(0.08)
    assert signal.total_cost_usdc == pytest.approx(18.4)


def test_fee_buffer_removes_the_edge():
    group, books = make_group("yes", "no"), binary_books()  # costo per set 0.97
    assert evaluate_group(group, books, 50.0, fee_buffer=0.02, now=T0) is None
    assert evaluate_group(group, books, 50.0, slippage_buffer=0.02, now=T0) is None
    kept = evaluate_group(group, books, 50.0, fee_buffer=0.01, now=T0)
    assert kept is not None
    assert kept.expected_net_edge == pytest.approx(0.02)


def test_empty_book_on_one_leg_returns_none():
    books = {"yes": make_book("yes", asks=levels((0.40, 100.0))), "no": make_book("no")}
    assert evaluate_group(make_group("yes", "no"), books, 10.0, now=T0) is None


def test_missing_book_returns_none():
    books = {"yes": make_book("yes", asks=levels((0.40, 100.0)))}
    assert evaluate_group(make_group("yes", "no"), books, 10.0, now=T0) is None
    assert evaluate_group(make_group("yes", "no"), {}, 10.0, now=T0) is None
    assert evaluate_group(make_group("yes", "no"), None, 10.0, now=T0) is None
    assert evaluate_group(None, books, 10.0, now=T0) is None


def test_stale_book_returns_none():
    books = binary_books()
    books["no"] = make_book("no", asks=levels((0.49, 100.0)), is_stale=True)
    assert evaluate_group(make_group("yes", "no"), books, 10.0, now=T0) is None


def test_non_mutually_exclusive_group_returns_none():
    group = make_group("yes", "no", mutually_exclusive=False)
    assert evaluate_group(group, binary_books(), 10.0, now=T0) is None


def test_degenerate_groups_return_none():
    books = binary_books()
    assert evaluate_group(make_group("yes"), books, 10.0, now=T0) is None  # un solo esito
    assert evaluate_group(make_group("yes", "yes"), books, 10.0, now=T0) is None  # token duplicato
    assert evaluate_group(make_group(), books, 10.0, now=T0) is None  # nessun esito


def test_insufficient_liquidity_requires_full_target_by_default():
    books = binary_books(size=100.0)
    books["no"] = make_book("no", asks=levels((0.49, 30.0)))
    assert evaluate_group(make_group("yes", "no"), books, 50.0, now=T0) is None


def test_min_executable_shares_allows_balanced_downsizing():
    books = binary_books(size=100.0)
    books["no"] = make_book("no", asks=levels((0.49, 30.0)))
    group = make_group("yes", "no")
    signal = evaluate_group(group, books, 50.0, min_executable_shares=20.0, now=T0)
    assert signal is not None
    assert [leg.target_shares for leg in signal.legs] == [pytest.approx(30.0)] * 2
    assert signal.total_cost_usdc == pytest.approx(30.0 * 0.97)
    # Sotto la size minima eseguibile non si emette nulla.
    assert evaluate_group(group, books, 50.0, min_executable_shares=40.0, now=T0) is None


def test_corrupted_ask_levels_are_ignored():
    books = binary_books()
    books["yes"] = make_book("yes").model_copy(
        update={"asks": [raw_level(0.0, 500.0), raw_level(float("nan"), 500.0), raw_level(0.48, 100.0)]}
    )
    signal = evaluate_group(make_group("yes", "no"), books, 50.0, now=T0)
    assert signal is not None
    assert signal.legs[0].expected_vwap == pytest.approx(0.48)  # l'ask a 0 NON crea un finto arbitraggio

    books["yes"] = make_book("yes").model_copy(update={"asks": [raw_level(0.0, 500.0)]})
    assert evaluate_group(make_group("yes", "no"), books, 50.0, now=T0) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_shares": 0.0},
        {"target_shares": -10.0},
        {"target_shares": float("nan")},
        {"target_shares": float("inf")},
        {"target_shares": 10.0, "fee_buffer": -0.01},
        {"target_shares": 10.0, "fee_buffer": float("nan")},
        {"target_shares": 10.0, "min_net_edge": float("nan")},
        {"target_shares": 10.0, "slippage_buffer": -1.0},
        {"target_shares": 10.0, "min_executable_shares": 0.0},
    ],
)
def test_invalid_parameters_return_none_without_raising(kwargs):
    target = kwargs.pop("target_shares")
    assert evaluate_group(make_group("yes", "no"), binary_books(), target, now=T0, **kwargs) is None


def test_evaluation_is_deterministic():
    group, books = make_group("yes", "no"), binary_books()
    first = evaluate_group(group, books, 50.0, now=T0)
    second = evaluate_group(group, books, 50.0, now=T0)
    assert first is not None
    assert first.model_dump() == second.model_dump()
    other = evaluate_group(group, books, 25.0, now=T0)
    assert other.signal_id != first.signal_id


def test_created_at_falls_back_to_latest_book_timestamp():
    later = T0 + timedelta(seconds=7)
    books = {
        "yes": make_book("yes", asks=levels((0.48, 100.0)), updated_at=T0),
        "no": make_book("no", asks=levels((0.49, 100.0)), updated_at=later),
    }
    signal = evaluate_group(make_group("yes", "no"), books, 50.0)
    assert signal is not None
    assert signal.created_at == later


def test_missing_book_timestamp_without_now_returns_none():
    books = binary_books()
    books["no"] = books["no"].model_copy(update={"updated_at": None})
    assert evaluate_group(make_group("yes", "no"), books, 50.0) is None
    assert evaluate_group(make_group("yes", "no"), books, 50.0, now=T0) is not None

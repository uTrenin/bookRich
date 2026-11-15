"""Test deterministici della commissione taker (`fee_rate`) nei tre moduli decisionali."""

from __future__ import annotations

import pytest

from engine.combinatorial_arb import evaluate_group
from engine.news_latency import evaluate_news
from engine.risk_manager import RiskReason, approve_signal
from engine_helpers import T0, levels, make_book, make_group, make_news


def binary_books():
    return {
        "yes": make_book("yes", asks=levels((0.48, 100.0))),
        "no": make_book("no", asks=levels((0.49, 100.0))),
    }


# ---------------------------------------------------------------- arbitraggio


def test_arb_fee_rate_is_charged_on_the_cost_of_the_set():
    group, books = make_group("yes", "no"), binary_books()  # 0.97 per set a book
    signal = evaluate_group(group, books, 50.0, fee_rate=0.01, now=T0)
    assert signal is not None
    # 0.97 * 1.01 = 0.9797 -> edge 0.0203
    assert signal.expected_net_edge == pytest.approx(1.0 - 0.97 * 1.01)
    # Il costo del segnale resta quello di book: le commissioni stanno nell'edge.
    assert signal.total_cost_usdc == pytest.approx(48.5)


def test_arb_fee_rate_can_remove_the_opportunity():
    group, books = make_group("yes", "no"), binary_books()
    assert evaluate_group(group, books, 50.0, now=T0) is not None
    # 0.97 * 1.02 = 0.9894 >= 0.985: con il 2% di commissione non c'e piu margine.
    assert evaluate_group(group, books, 50.0, fee_rate=0.02, now=T0) is None


def test_arb_fee_rate_and_fee_buffer_add_up():
    group, books = make_group("yes", "no"), binary_books()
    signal = evaluate_group(group, books, 50.0, fee_rate=0.005, fee_buffer=0.005, min_net_edge=0.01, now=T0)
    assert signal is not None
    assert signal.expected_net_edge == pytest.approx(1.0 - 0.97 * 1.005 - 0.005)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), None, "0.01"])
def test_arb_invalid_fee_rate_returns_none(bad):
    assert evaluate_group(make_group("yes", "no"), binary_books(), 50.0, fee_rate=bad, now=T0) is None


# ---------------------------------------------------------------- notizie


def test_news_budget_cap_includes_fees():
    book = make_book("yes", asks=levels((0.50, 1000.0)))
    signal = evaluate_news(make_news(), book, now=T0, fee_rate=0.02)
    assert signal is not None
    # Prezzo effettivo 0.51: Kelly 0.3878 -> 77.6 USDC, cappato a 50 commissioni incluse.
    assert signal.total_cost_usdc == pytest.approx(50.0 / 1.02)
    assert signal.total_cost_usdc * 1.02 <= 50.0 + 1e-9
    assert signal.legs[0].target_shares == pytest.approx(50.0 / 1.02 / 0.50)
    assert signal.expected_net_edge == pytest.approx(0.70 - 0.50 * 1.02)


def test_news_fee_rate_can_remove_the_edge():
    book = make_book("yes", asks=levels((0.50, 1000.0)))
    news = make_news(fair=0.55)
    assert evaluate_news(news, book, now=T0) is not None  # edge 0.05
    assert evaluate_news(news, book, now=T0, fee_rate=0.06) is None  # prezzo effettivo 0.53: edge 0.02


def test_news_levels_are_limited_by_the_fee_adjusted_price():
    # Senza commissioni il livello a 0.66 lascia 0.04 di edge e viene comprato;
    # al 2% costa 0.6732 e ne lascerebbe 0.027 < 0.03: ci si ferma al primo livello.
    book = make_book("yes", asks=levels((0.50, 20.0), (0.66, 1000.0)))
    free = evaluate_news(make_news(), book, now=T0)
    paid = evaluate_news(make_news(), book, now=T0, fee_rate=0.02)
    assert free.legs[0].max_acceptable_price == pytest.approx(0.66)
    assert paid.legs[0].max_acceptable_price == pytest.approx(0.50)
    assert paid.legs[0].target_shares == pytest.approx(20.0)


def test_news_kelly_uses_the_fee_adjusted_price():
    book = make_book("yes", asks=levels((0.50, 1000.0)))
    free = evaluate_news(make_news(), book, now=T0, bankroll_usdc=100.0)
    paid = evaluate_news(make_news(), book, now=T0, bankroll_usdc=100.0, fee_rate=0.02)
    # 100 * 0.25 * 0.80 * (0.70 - 0.51) / (1 - 0.51) = 7.755 USDC, commissioni incluse
    assert free.total_cost_usdc == pytest.approx(8.0)
    assert paid.total_cost_usdc * 1.02 == pytest.approx(100.0 * 0.25 * 0.80 * (0.70 - 0.51) / 0.49)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), None])
def test_news_invalid_fee_rate_returns_none(bad):
    book = make_book("yes", asks=levels((0.50, 1000.0)))
    assert evaluate_news(make_news(), book, now=T0, fee_rate=bad) is None


# ---------------------------------------------------------------- rischio


def test_risk_required_capital_includes_fees():
    books = binary_books()
    signal = evaluate_group(make_group("yes", "no"), books, 50.0, now=T0)  # 48.5 USDC a book
    limits = {"bankroll_usdc": 1000.0, "max_event_exposure_usdc": 49.0}

    free = approve_signal(signal, books, 0.0, **limits)
    assert free.approved and free.required_usdc == pytest.approx(48.5)

    paid = approve_signal(signal, books, 0.0, fee_rate=0.02, **limits)
    assert paid.reason is RiskReason.EVENT_EXPOSURE_EXCEEDED  # 48.5 * 1.02 = 49.47 > 49
    assert paid.required_usdc == pytest.approx(48.5 * 1.02)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), None])
def test_risk_invalid_fee_rate_is_rejected(bad):
    books = binary_books()
    signal = evaluate_group(make_group("yes", "no"), books, 50.0, now=T0)
    assert approve_signal(signal, books, 0.0, fee_rate=bad).reason is RiskReason.INVALID_INPUT

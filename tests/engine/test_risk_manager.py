"""Test deterministici di `engine/risk_manager.py` (T2-D)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from contracts.models import OrderLeg, TradeSignal
from engine.combinatorial_arb import evaluate_group
from engine.news_latency import evaluate_news
from engine.risk_manager import RiskDecision, RiskReason, approve_signal
from engine_helpers import T0, levels, make_book, make_group, make_news

LIMITS = {"bankroll_usdc": 1000.0, "max_event_exposure_usdc": 100.0}


def arb_books(size: float = 100.0):
    return {
        "yes": make_book("yes", asks=levels((0.48, size))),
        "no": make_book("no", asks=levels((0.49, size))),
    }


@pytest.fixture
def arb_signal() -> TradeSignal:
    signal = evaluate_group(make_group("yes", "no"), arb_books(), 50.0, now=T0)
    assert signal is not None  # costo 48.5 USDC
    return signal


def test_valid_signal_is_approved(arb_signal):
    decision = approve_signal(arb_signal, arb_books(), 0.0, **LIMITS)
    assert isinstance(decision, RiskDecision)
    assert decision.approved
    assert bool(decision) is True
    assert decision.reason is RiskReason.APPROVED
    assert decision.required_usdc == pytest.approx(48.5)


def test_news_signal_is_approved_with_default_limits():
    book = make_book("yes", asks=levels((0.50, 1000.0)))
    signal = evaluate_news(make_news(), book, now=T0)  # 50 USDC, pari al cap di default
    decision = approve_signal(signal, {"yes": book}, 0.0)
    assert decision.approved
    assert decision.required_usdc == pytest.approx(50.0)


def test_single_stale_book_blocks_the_signal(arb_signal):
    books = arb_books()
    books["no"] = make_book("no", asks=levels((0.49, 100.0)), is_stale=True)
    decision = approve_signal(arb_signal, books, 0.0, **LIMITS)
    assert not decision
    assert decision.reason is RiskReason.STALE_BOOK


def test_stale_book_wins_over_every_other_check(arb_signal):
    # Anche con liquidita e capitale insufficienti il motivo riportato resta il book stale.
    books = {
        "yes": make_book("yes", asks=levels((0.48, 1.0))),
        "no": make_book("no", is_stale=True),
    }
    decision = approve_signal(arb_signal, books, 999.0, **LIMITS)
    assert decision.reason is RiskReason.STALE_BOOK


def test_missing_book_blocks_the_signal(arb_signal):
    books = arb_books()
    del books["no"]
    assert approve_signal(arb_signal, books, 0.0, **LIMITS).reason is RiskReason.MISSING_BOOK
    assert approve_signal(arb_signal, {}, 0.0, **LIMITS).reason is RiskReason.MISSING_BOOK


def test_vanished_liquidity_blocks_the_signal(arb_signal):
    decision = approve_signal(arb_signal, arb_books(size=30.0), 0.0, **LIMITS)
    assert not decision.approved
    assert decision.reason is RiskReason.INSUFFICIENT_LIQUIDITY


def test_empty_book_blocks_the_signal(arb_signal):
    books = arb_books()
    books["yes"] = make_book("yes")
    assert approve_signal(arb_signal, books, 0.0, **LIMITS).reason is RiskReason.INSUFFICIENT_LIQUIDITY


def test_price_moved_beyond_max_acceptable_blocks_the_signal(arb_signal):
    books = arb_books()
    books["yes"] = make_book("yes", asks=levels((0.55, 100.0)))  # il limite del leg era 0.48
    decision = approve_signal(arb_signal, books, 0.0, **LIMITS)
    assert decision.reason is RiskReason.INSUFFICIENT_LIQUIDITY


def test_event_exposure_cap(arb_signal):
    decision = approve_signal(
        arb_signal, arb_books(), 60.0, current_event_exposure_usdc=60.0, **LIMITS
    )
    assert decision.reason is RiskReason.EVENT_EXPOSURE_EXCEEDED
    assert decision.required_usdc == pytest.approx(48.5)
    ok = approve_signal(arb_signal, arb_books(), 51.5, current_event_exposure_usdc=51.5, **LIMITS)
    assert ok.approved  # 51.5 + 48.5 == 100: esattamente al cap


def test_bankroll_cap(arb_signal):
    decision = approve_signal(arb_signal, arb_books(), 960.0, **LIMITS)
    assert decision.reason is RiskReason.BANKROLL_EXCEEDED
    assert approve_signal(arb_signal, arb_books(), 951.5, **LIMITS).approved  # esattamente il bankroll


def test_total_exposure_cap(arb_signal):
    decision = approve_signal(arb_signal, arb_books(), 180.0, max_total_exposure_usdc=200.0, **LIMITS)
    assert decision.reason is RiskReason.TOTAL_EXPOSURE_EXCEEDED
    assert approve_signal(arb_signal, arb_books(), 100.0, max_total_exposure_usdc=200.0, **LIMITS).approved


def test_default_event_cap_blocks_second_position_on_same_event(arb_signal):
    decision = approve_signal(arb_signal, arb_books(), 10.0, current_event_exposure_usdc=10.0)
    assert decision.reason is RiskReason.EVENT_EXPOSURE_EXCEEDED  # 10 + 48.5 > 50


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_exposure_usdc": float("nan")},
        {"current_exposure_usdc": -1.0},
        {"current_exposure_usdc": None},
        {"current_exposure_usdc": 0.0, "bankroll_usdc": float("inf")},
        {"current_exposure_usdc": 0.0, "max_event_exposure_usdc": float("nan")},
        {"current_exposure_usdc": 0.0, "current_event_exposure_usdc": -5.0},
        {"current_exposure_usdc": 0.0, "max_total_exposure_usdc": float("nan")},
        {"current_exposure_usdc": 0.0, "max_book_age_sec": 5.0},  # manca `now`
    ],
)
def test_invalid_inputs_are_rejected_without_raising(arb_signal, kwargs):
    exposure = kwargs.pop("current_exposure_usdc")
    decision = approve_signal(arb_signal, arb_books(), exposure, **kwargs)
    assert not decision.approved
    assert decision.reason is RiskReason.INVALID_INPUT


def test_missing_signal_or_books_is_rejected(arb_signal):
    assert approve_signal(None, arb_books(), 0.0).reason is RiskReason.INVALID_INPUT
    assert approve_signal(arb_signal, None, 0.0).reason is RiskReason.INVALID_INPUT


def test_malformed_signals_are_rejected(arb_signal):
    books = arb_books()
    no_legs = arb_signal.model_copy(update={"legs": []})
    assert approve_signal(no_legs, books, 0.0, **LIMITS).reason is RiskReason.INVALID_SIGNAL
    nan_cost = arb_signal.model_copy(update={"total_cost_usdc": float("nan")})
    assert approve_signal(nan_cost, books, 0.0, **LIMITS).reason is RiskReason.INVALID_SIGNAL
    zero_cost = arb_signal.model_copy(update={"total_cost_usdc": 0.0})
    assert approve_signal(zero_cost, books, 0.0, **LIMITS).reason is RiskReason.INVALID_SIGNAL
    nan_edge = arb_signal.model_copy(update={"expected_net_edge": float("nan")})
    assert approve_signal(nan_edge, books, 0.0, **LIMITS).reason is RiskReason.INVALID_SIGNAL


def test_non_positive_edge_is_rejected(arb_signal):
    for edge in (0.0, -0.01):
        signal = arb_signal.model_copy(update={"expected_net_edge": edge})
        decision = approve_signal(signal, arb_books(), 0.0, **LIMITS)
        assert decision.reason is RiskReason.NON_POSITIVE_EDGE


def test_understated_signal_cost_is_recomputed_on_current_book(arb_signal):
    cheap = arb_signal.model_copy(update={"total_cost_usdc": 1.0})
    decision = approve_signal(cheap, arb_books(), 0.0, **LIMITS)
    assert decision.approved
    assert decision.required_usdc == pytest.approx(48.5)  # non ci si fida del costo dichiarato
    assert approve_signal(cheap, arb_books(), 960.0, **LIMITS).reason is RiskReason.BANKROLL_EXCEEDED


def test_optional_book_age_check(arb_signal):
    books = arb_books()  # updated_at == T0
    fresh = approve_signal(
        arb_signal, books, 0.0, now=T0 + timedelta(seconds=2), max_book_age_sec=5.0, **LIMITS
    )
    assert fresh.approved
    old = approve_signal(
        arb_signal, books, 0.0, now=T0 + timedelta(seconds=30), max_book_age_sec=5.0, **LIMITS
    )
    assert old.reason is RiskReason.STALE_BOOK
    # `now` da solo non attiva il controllo.
    assert approve_signal(arb_signal, books, 0.0, now=T0 + timedelta(days=1), **LIMITS).approved
    books["no"] = books["no"].model_copy(update={"updated_at": None})
    missing_ts = approve_signal(arb_signal, books, 0.0, now=T0, max_book_age_sec=5.0, **LIMITS)
    assert missing_ts.reason is RiskReason.STALE_BOOK


def test_sell_leg_is_checked_against_bids():
    leg = OrderLeg(
        token_id="yes", side="SELL", target_shares=20.0, expected_vwap=0.60, max_acceptable_price=0.58
    )
    signal = TradeSignal(
        signal_id="manual-1",
        strategy="NEWS_LATENCY",
        event_id="evt-1",
        legs=[leg],
        expected_net_edge=0.05,
        total_cost_usdc=12.0,
        created_at=T0,
        rationale="test",
    )
    deep = {"yes": make_book("yes", bids=levels((0.60, 50.0)))}
    assert approve_signal(signal, deep, 0.0, **LIMITS).approved
    weak = {"yes": make_book("yes", bids=levels((0.60, 5.0), (0.50, 50.0)))}  # sotto il limite 0.58
    assert approve_signal(signal, weak, 0.0, **LIMITS).reason is RiskReason.INSUFFICIENT_LIQUIDITY


def test_decision_is_deterministic(arb_signal):
    first = approve_signal(arb_signal, arb_books(), 10.0, **LIMITS)
    second = approve_signal(arb_signal, arb_books(), 10.0, **LIMITS)
    assert first == second

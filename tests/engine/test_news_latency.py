"""Test deterministici di `engine/news_latency.py` (T2-C)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from engine.news_latency import evaluate_news, kelly_fraction_for_binary, news_age_seconds
from engine_helpers import T0, levels, make_book, make_news, raw_level


def deep_book(price: float = 0.50, size: float = 1000.0):
    return make_book("yes", asks=levels((price, size)))


def test_fresh_news_with_edge_emits_capped_buy_signal():
    signal = evaluate_news(make_news(), deep_book(), now=T0)
    assert signal is not None
    assert signal.strategy == "NEWS_LATENCY"
    assert signal.created_at == T0
    # Kelly pieno (0.70-0.50)/(1-0.50)=0.40 -> 1000 * 0.25 * 0.80 * 0.40 = 80 USDC, cappato a 50.
    assert signal.total_cost_usdc == pytest.approx(50.0)
    assert signal.expected_net_edge == pytest.approx(0.20)
    (leg,) = signal.legs
    assert leg.side == "BUY"
    assert leg.token_id == "yes"
    assert leg.target_shares == pytest.approx(100.0)
    assert leg.expected_vwap == pytest.approx(0.50)
    assert leg.max_acceptable_price == pytest.approx(0.50)


def test_expired_news_is_discarded():
    assert evaluate_news(make_news(age_sec=121.0), deep_book(), now=T0) is None
    assert evaluate_news(make_news(age_sec=3600.0), deep_book(), now=T0) is None


def test_news_exactly_at_max_age_is_still_valid():
    assert evaluate_news(make_news(age_sec=120.0), deep_book(), now=T0) is not None
    assert evaluate_news(make_news(age_sec=10.0), deep_book(), now=T0, max_age_sec=5.0) is None


def test_low_confidence_is_filtered():
    assert evaluate_news(make_news(confidence=0.59), deep_book(), now=T0) is None
    assert evaluate_news(make_news(confidence=0.60), deep_book(), now=T0) is not None


def test_edge_below_minimum_returns_none():
    assert evaluate_news(make_news(fair=0.52), deep_book(0.50), now=T0) is None  # edge 0.02 < 0.03
    assert evaluate_news(make_news(fair=0.40), deep_book(0.50), now=T0) is None  # edge negativo
    assert evaluate_news(make_news(fair=0.50), deep_book(0.50), now=T0, min_net_edge=0.0) is None


def test_fee_buffer_is_subtracted_from_edge():
    news = make_news(fair=0.55)
    assert evaluate_news(news, deep_book(0.50), now=T0) is not None  # edge 0.05
    assert evaluate_news(news, deep_book(0.50), now=T0, fee_buffer=0.03) is None  # edge 0.02


def test_fractional_kelly_sizing_below_cap():
    signal = evaluate_news(make_news(), deep_book(), now=T0, bankroll_usdc=100.0)
    # 100 * 0.25 * 0.80 * 0.40 = 8 USDC -> 16 shares @ 0.50
    assert signal is not None
    assert signal.total_cost_usdc == pytest.approx(8.0)
    assert signal.legs[0].target_shares == pytest.approx(16.0)


def test_sizing_scales_with_confidence():
    low = evaluate_news(make_news(confidence=0.6), deep_book(), now=T0, bankroll_usdc=100.0)
    high = evaluate_news(make_news(confidence=1.0), deep_book(), now=T0, bankroll_usdc=100.0)
    assert low.total_cost_usdc == pytest.approx(6.0)
    assert high.total_cost_usdc == pytest.approx(10.0)


def test_position_never_exceeds_cap():
    signal = evaluate_news(
        make_news(fair=0.99, confidence=1.0),
        deep_book(0.10, 100000.0),
        now=T0,
        bankroll_usdc=1_000_000.0,
        kelly_fraction=1.0,
        max_position_usdc=50.0,
    )
    assert signal is not None
    assert signal.total_cost_usdc <= 50.0 + 1e-9


def test_vwap_walks_multiple_levels():
    book = make_book("yes", asks=levels((0.50, 40.0), (0.60, 1000.0)))
    signal = evaluate_news(make_news(), book, now=T0)
    # 40 @ 0.50 = 20 USDC + 50 @ 0.60 = 30 USDC -> 90 shares per 50 USDC
    assert signal is not None
    (leg,) = signal.legs
    assert leg.target_shares == pytest.approx(90.0)
    assert signal.total_cost_usdc == pytest.approx(50.0)
    assert leg.expected_vwap == pytest.approx(50.0 / 90.0)
    assert leg.max_acceptable_price == pytest.approx(0.60)
    assert signal.expected_net_edge == pytest.approx(0.70 - 50.0 / 90.0)


def test_levels_without_minimum_edge_are_not_bought():
    # Il secondo livello (0.69) lascerebbe solo 0.01 di edge: ci si ferma al primo.
    book = make_book("yes", asks=levels((0.50, 20.0), (0.69, 1000.0)))
    signal = evaluate_news(make_news(), book, now=T0)
    assert signal is not None
    assert signal.legs[0].target_shares == pytest.approx(20.0)
    assert signal.total_cost_usdc == pytest.approx(10.0)
    assert signal.legs[0].max_acceptable_price == pytest.approx(0.50)


def test_budget_is_refined_on_executable_vwap():
    # Kelly sul best ask (0.20): 0.625 * 250 = 156 USDC; il VWAP reale e molto peggiore
    # e il budget va ridotto al Kelly calcolato sul prezzo effettivo.
    book = make_book("yes", asks=levels((0.20, 10.0), (0.60, 10000.0)))
    signal = evaluate_news(
        make_news(fair=0.70, confidence=1.0), book, now=T0, max_position_usdc=1000.0
    )
    assert signal is not None
    vwap = signal.legs[0].expected_vwap
    kelly_at_vwap = 1000.0 * 0.25 * 1.0 * kelly_fraction_for_binary(0.70, vwap)
    assert signal.total_cost_usdc <= kelly_at_vwap + 1e-6
    assert signal.total_cost_usdc < 156.0


def test_position_below_minimum_returns_none():
    assert evaluate_news(make_news(), deep_book(), now=T0, bankroll_usdc=10.0) is None  # 0.8 USDC
    thin = make_book("yes", asks=levels((0.50, 1.0)))  # solo 0.5 USDC eseguibili
    assert evaluate_news(make_news(), thin, now=T0) is None


def test_empty_stale_or_mismatched_book_returns_none():
    news = make_news()
    assert evaluate_news(news, make_book("yes"), now=T0) is None
    assert evaluate_news(news, None, now=T0) is None
    assert evaluate_news(None, deep_book(), now=T0) is None
    stale = make_book("yes", asks=levels((0.50, 1000.0)), is_stale=True)
    assert evaluate_news(news, stale, now=T0) is None
    other = make_book("other-token", asks=levels((0.50, 1000.0)))
    assert evaluate_news(news, other, now=T0) is None


def test_corrupted_ask_levels_do_not_create_fake_edge():
    book = make_book("yes").model_copy(
        update={"asks": [raw_level(0.0, 1000.0), raw_level(float("nan"), 1000.0)]}
    )
    assert evaluate_news(make_news(), book, now=T0) is None


def test_missing_or_incompatible_timestamps_return_none():
    news = make_news()
    assert evaluate_news(news.model_copy(update={"published_at": None}), deep_book(), now=T0) is None
    assert evaluate_news(news, deep_book(), now=None) is None
    naive_now = datetime(2026, 1, 1, 12, 0, 0)  # naive vs aware: nessun TypeError
    assert evaluate_news(news, deep_book(), now=naive_now) is None
    assert news_age_seconds(None, T0) is None


def test_news_from_the_future_tolerates_only_small_clock_skew():
    assert evaluate_news(make_news(age_sec=-2.0), deep_book(), now=T0) is not None
    assert evaluate_news(make_news(age_sec=-60.0), deep_book(), now=T0) is None


def test_no_division_by_zero_when_price_reaches_one():
    news = make_news(fair=1.0, confidence=1.0)
    assert evaluate_news(news, deep_book(1.0), now=T0, min_net_edge=0.0) is None
    # ask 0.99 + fee 0.01 = prezzo effettivo 1.0 -> denominatore di Kelly nullo
    assert evaluate_news(news, deep_book(0.99), now=T0, min_net_edge=0.0, fee_buffer=0.01) is None
    assert kelly_fraction_for_binary(1.0, 1.0) == 0.0
    assert kelly_fraction_for_binary(0.5, 0.0) == 0.0
    assert kelly_fraction_for_binary(float("nan"), 0.5) == 0.0


def test_kelly_fraction_formula():
    assert kelly_fraction_for_binary(0.70, 0.50) == pytest.approx(0.40)
    assert kelly_fraction_for_binary(0.50, 0.50) == 0.0
    assert kelly_fraction_for_binary(0.30, 0.50) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_age_sec": float("nan")},
        {"min_net_edge": -0.1},
        {"kelly_fraction": 1.5},
        {"kelly_fraction": float("inf")},
        {"bankroll_usdc": -100.0},
        {"max_position_usdc": float("nan")},
        {"fee_buffer": -0.01},
    ],
)
def test_invalid_parameters_return_none_without_raising(kwargs):
    assert evaluate_news(make_news(), deep_book(), now=T0, **kwargs) is None


def test_invalid_assessment_values_return_none():
    news = make_news()
    for update in ({"estimated_fair_prob": float("nan")}, {"confidence": None}, {"estimated_fair_prob": 1.2}):
        assert evaluate_news(news.model_copy(update=update), deep_book(), now=T0) is None


def test_event_id_defaults_to_target_token_and_is_deterministic():
    news, book = make_news(), deep_book()
    default = evaluate_news(news, book, now=T0)
    assert default.event_id == "yes"
    explicit = evaluate_news(news, book, now=T0, event_id="evt-42")
    assert explicit.event_id == "evt-42"
    assert default.model_dump() == evaluate_news(news, book, now=T0).model_dump()
    later = evaluate_news(news, book, now=T0 + timedelta(seconds=1))
    assert later.signal_id != default.signal_id

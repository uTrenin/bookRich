"""Virtual portfolio: mark-to-market, locked arbitrage profit, drawdown, settlement, restart."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from contracts.models import BookLevel, MarketGroup, MarketOutcome, OrderBookSnapshot, OrderLeg, PaperFill, TradeSignal
from paper_trader.logger import PaperTradeLogger
from paper_trader.portfolio import PaperPortfolio
from paper_trader.runner import PaperTradingRunner

T0 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)


def _book(token_id: str, bid: float | None = None, ask: float | None = None, stale: bool = False) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        bids=[BookLevel(price=bid, size=500.0)] if bid is not None else [],
        asks=[BookLevel(price=ask, size=500.0)] if ask is not None else [],
        updated_at=datetime.now(timezone.utc),
        is_stale=stale,
    )


def _signal(event_id: str, strategy: str, legs: list[tuple[str, float, float]], signal_id: str = "sig") -> TradeSignal:
    return TradeSignal(
        signal_id=signal_id,
        strategy=strategy,  # type: ignore[arg-type]
        event_id=event_id,
        legs=[
            OrderLeg(token_id=t, side="BUY", target_shares=s, expected_vwap=p, max_acceptable_price=p)
            for t, s, p in legs
        ],
        expected_net_edge=0.03,
        total_cost_usdc=sum(s * p for _, s, p in legs),
        created_at=T0,
        rationale="pnl test",
    )


def _fill(signal: TradeSignal, cost: float, status: str = "FILLED") -> PaperFill:
    return PaperFill(
        signal_id=signal.signal_id,
        strategy=signal.strategy,
        executed_at=T0,
        filled_legs=list(signal.legs) if status == "FILLED" else [],
        actual_simulated_cost_usdc=cost,
        slippage_vs_best_ask=0.0,
        status=status,  # type: ignore[arg-type]
        note="",
    )


def _arb(portfolio: PaperPortfolio, cost: float = 24.25, event_id: str = "evt-arb") -> TradeSignal:
    signal = _signal(event_id, "COMBINATORIAL_ARB", [("yes", 25.0, 0.48), ("no", 25.0, 0.49)], signal_id=f"arb-{cost}")
    portfolio.record_fill(signal, _fill(signal, cost), fees_usdc=0.0)
    return signal


def _news(portfolio: PaperPortfolio, shares: float = 100.0, cost: float = 50.0) -> TradeSignal:
    signal = _signal("evt-news", "NEWS_LATENCY", [("fed", shares, cost / shares)], signal_id="news-1")
    portfolio.record_fill(signal, _fill(signal, cost), fees_usdc=0.0)
    return signal


# ---------------------------------------------------------------- valuation


def test_arbitrage_lot_locks_its_profit_whatever_the_market_does() -> None:
    portfolio = PaperPortfolio(starting_bankroll_usdc=1000.0)
    _arb(portfolio, cost=24.25)

    cheap = portfolio.snapshot({"yes": _book("yes", bid=0.40), "no": _book("no", bid=0.45)})
    assert cheap["locked_arb_pnl_usdc"] == pytest.approx(0.75)  # 25 sets pay 25, cost 24.25
    assert cheap["market_value_usdc"] == pytest.approx(25 * 0.85)
    assert cheap["unrealized_pnl_usdc"] == pytest.approx(25 * 0.85 - 24.25)  # selling now would lose
    assert cheap["equity_usdc"] == pytest.approx(1000.0 + 25 * 0.85 - 24.25)

    rich = portfolio.snapshot({"yes": _book("yes", bid=0.50), "no": _book("no", bid=0.49)})
    assert rich["locked_arb_pnl_usdc"] == pytest.approx(0.75)  # unchanged: it does not depend on bids
    assert rich["unrealized_pnl_usdc"] == pytest.approx(25 * 0.99 - 24.25)


def test_news_lot_is_marked_at_the_best_bid_and_has_no_locked_profit() -> None:
    portfolio = PaperPortfolio()
    _news(portfolio, shares=100.0, cost=50.0)

    snap = portfolio.snapshot({"fed": _book("fed", bid=0.62, ask=0.64)})
    assert snap["open_lots"] == 1
    assert snap["cost_basis_usdc"] == pytest.approx(50.0)
    assert snap["market_value_usdc"] == pytest.approx(62.0)  # the bid, not the ask
    assert snap["unrealized_pnl_usdc"] == pytest.approx(12.0)
    assert snap["locked_arb_pnl_usdc"] == 0.0


def test_stale_or_empty_books_keep_the_last_mark_and_unpriced_tokens_count_as_zero() -> None:
    portfolio = PaperPortfolio()
    _news(portfolio)

    unpriced = portfolio.snapshot({})
    assert unpriced["unpriced_tokens"] == 1
    assert unpriced["market_value_usdc"] == 0.0

    portfolio.snapshot({"fed": _book("fed", bid=0.60)})
    for unusable in (_book("fed", bid=0.10, stale=True), _book("fed", bid=None)):
        snap = portfolio.snapshot({"fed": unusable})
        assert snap["market_value_usdc"] == pytest.approx(60.0)  # a stale or empty book is not a price
        assert snap["unpriced_tokens"] == 0


def test_max_drawdown_remembers_the_worst_fall_from_the_peak() -> None:
    portfolio = PaperPortfolio(starting_bankroll_usdc=1000.0)
    _news(portfolio, shares=100.0, cost=50.0)

    assert portfolio.snapshot({"fed": _book("fed", bid=0.60)})["equity_usdc"] == pytest.approx(1010.0)  # new peak
    down = portfolio.snapshot({"fed": _book("fed", bid=0.35)})
    assert down["equity_usdc"] == pytest.approx(985.0)
    assert down["max_drawdown_usdc"] == pytest.approx(25.0)

    recovered = portfolio.snapshot({"fed": _book("fed", bid=0.58)})
    assert recovered["equity_usdc"] == pytest.approx(1008.0)
    assert recovered["peak_equity_usdc"] == pytest.approx(1010.0)
    assert recovered["max_drawdown_usdc"] == pytest.approx(25.0)  # the worst fall is not forgotten
    assert recovered["max_drawdown_pct"] == pytest.approx(25.0 / 1010.0 * 100.0)


def test_fees_are_part_of_the_cost_and_reported_apart() -> None:
    portfolio = PaperPortfolio()
    signal = _signal("evt-arb", "COMBINATORIAL_ARB", [("yes", 25.0, 0.48), ("no", 25.0, 0.49)])
    portfolio.record_fill(signal, _fill(signal, cost=24.25 * 1.02), fees_usdc=24.25 * 0.02)

    snap = portfolio.snapshot({})
    assert snap["fees_paid_usdc"] == pytest.approx(0.485)
    assert snap["locked_arb_pnl_usdc"] == pytest.approx(25.0 - 24.25 * 1.02)


def test_only_filled_buy_legs_open_a_position() -> None:
    portfolio = PaperPortfolio()
    signal = _signal("evt", "NEWS_LATENCY", [("fed", 10.0, 0.5)])
    assert portfolio.record_fill(signal, _fill(signal, 5.0, status="PARTIAL_REJECTED")) is None
    assert portfolio.record_fill(signal, _fill(signal, 0.0)) is None

    sell_only = signal.model_copy(update={"legs": [
        OrderLeg(token_id="fed", side="SELL", target_shares=10.0, expected_vwap=0.5, max_acceptable_price=0.5)
    ]})
    assert portfolio.record_fill(sell_only, _fill(sell_only, 5.0)) is None
    assert portfolio.snapshot({})["open_lots"] == 0


# ---------------------------------------------------------------- settlement


def test_settle_event_realizes_arbitrage_and_leaves_single_outcome_lots_open() -> None:
    portfolio = PaperPortfolio(starting_bankroll_usdc=1000.0)
    _arb(portfolio, cost=24.25, event_id="evt-mixed")
    news = _signal("evt-mixed", "NEWS_LATENCY", [("yes", 40.0, 0.5)], signal_id="news-mixed")
    portfolio.record_fill(news, _fill(news, 20.0))

    result = portfolio.settle_event("evt-mixed")
    assert result == {"realized_pnl_usdc": pytest.approx(0.75), "settled_lots": 1, "awaiting_outcome": 1}

    snap = portfolio.snapshot({})
    assert snap["realized_pnl_usdc"] == pytest.approx(0.75)
    assert snap["open_lots"] == 1 and snap["settled_lots"] == 1
    assert snap["locked_arb_pnl_usdc"] == 0.0  # no longer locked: it is realized
    assert portfolio.settle_event("evt-other") == {"realized_pnl_usdc": 0.0, "settled_lots": 0, "awaiting_outcome": 0}


def test_settle_tokens_uses_real_payouts() -> None:
    won = PaperPortfolio(starting_bankroll_usdc=1000.0)
    _news(won, shares=100.0, cost=50.0)
    assert won.settle_tokens({"fed": 1.0}) == {"realized_pnl_usdc": pytest.approx(50.0), "settled_lots": 1}
    assert won.snapshot({})["equity_usdc"] == pytest.approx(1050.0)

    lost = PaperPortfolio(starting_bankroll_usdc=1000.0)
    _news(lost, shares=100.0, cost=50.0)
    assert lost.settle_tokens({"fed": 0.0})["realized_pnl_usdc"] == pytest.approx(-50.0)
    assert lost.snapshot({})["equity_usdc"] == pytest.approx(950.0)

    # A lot is settled only when every one of its tokens has a valid payout.
    partial = PaperPortfolio()
    _arb(partial)
    assert partial.settle_tokens({"yes": 1.0})["settled_lots"] == 0
    assert partial.settle_tokens({"yes": 1.0, "no": float("nan")})["settled_lots"] == 0
    assert partial.settle_tokens({"yes": 1.0, "no": 0.0}) == {"realized_pnl_usdc": pytest.approx(0.75), "settled_lots": 1}


# ---------------------------------------------------------------- runner and log


class _Engine:
    """Minimal engine: one arbitrage signal per update, with exposure tracking like the adapter."""

    bankroll_usdc = 500.0

    def __init__(self) -> None:
        self.event_exposure_usdc: dict[str, float] = {}

    def evaluate_group(self, group: MarketGroup, books: dict[str, OrderBookSnapshot]) -> TradeSignal | None:
        if self.event_exposure_usdc.get(group.event_id):
            return None
        return _signal(group.event_id, "COMBINATORIAL_ARB", [("yes", 25.0, 0.48), ("no", 25.0, 0.49)])

    def evaluate_news(self, assessment: Any, book: Any) -> None:
        return None

    def approve_signal(self, signal: TradeSignal, books: dict[str, OrderBookSnapshot]) -> tuple[bool, str]:
        return True, "ok"

    def record_fill(self, signal: TradeSignal, fill: PaperFill) -> None:
        self.event_exposure_usdc[signal.event_id] = fill.actual_simulated_cost_usdc

    def release_event_exposure(self, event_id: str) -> float:
        return float(self.event_exposure_usdc.pop(event_id, 0.0))


class _Discovery:
    def __init__(self, closed: bool | None) -> None:
        self.closed = closed

    async def discover_market_groups(self, limit: int = 50) -> list[MarketGroup]:
        return [_group("evt-other")]

    async def is_event_closed(self, event_id: str) -> bool | None:
        return self.closed


def _group(event_id: str) -> MarketGroup:
    return MarketGroup(
        event_id=event_id,
        title=event_id,
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(market_id=event_id, question="Q?", outcome_label="Yes", token_id="yes"),
            MarketOutcome(market_id=event_id, question="Q?", outcome_label="No", token_id="no"),
        ],
    )


def _runner(tmp_path: Path, closed: bool | None = None) -> PaperTradingRunner:
    books = {"yes": _book("yes", bid=0.47, ask=0.48), "no": _book("no", bid=0.48, ask=0.49)}
    return PaperTradingRunner(
        engine=_Engine(),
        discovery_client=_Discovery(closed),
        market_groups=[_group("evt-arb")],
        books=books,
        trade_logger=PaperTradeLogger(tmp_path / "pnl.jsonl"),
    )


def test_runner_tracks_pnl_in_the_heartbeat(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    assert runner.portfolio.starting_bankroll_usdc == 500.0  # taken from the engine

    fills = asyncio.run(runner.handle_book_update(_book("yes", bid=0.47, ask=0.48)))
    assert len(fills) == 1

    pnl = runner.get_heartbeat_stats()["pnl"]
    assert pnl["open_lots"] == 1
    assert pnl["cost_basis_usdc"] == pytest.approx(24.25)
    assert pnl["locked_arb_pnl_usdc"] == pytest.approx(0.75)
    assert pnl["market_value_usdc"] == pytest.approx(25 * (0.47 + 0.48))
    assert pnl["unrealized_pnl_usdc"] == pytest.approx(25 * 0.95 - 24.25)
    assert pnl["equity_usdc"] == pytest.approx(500.0 + 25 * 0.95 - 24.25)


def test_closed_event_settlement_reports_realized_pnl(tmp_path: Path) -> None:
    runner = _runner(tmp_path, closed=True)
    asyncio.run(runner.handle_book_update(_book("yes", bid=0.47, ask=0.48)))
    _, settled = asyncio.run(runner.refresh_markets_once())
    assert settled == ["evt-arb"]

    entries = [json.loads(line) for line in (tmp_path / "pnl.jsonl").read_text(encoding="utf-8").splitlines()]
    settled_entry = next(e for e in entries if e["event_type"] == "PAPER_SETTLED")
    assert settled_entry["payload"]["realized_pnl_usdc"] == pytest.approx(0.75)
    assert settled_entry["payload"]["settled_lots"] == 1
    assert settled_entry["payload"]["awaiting_outcome"] == 0

    pnl = runner.get_heartbeat_stats()["pnl"]
    assert pnl["realized_pnl_usdc"] == pytest.approx(0.75)
    assert pnl["open_lots"] == 0
    assert pnl["equity_usdc"] == pytest.approx(500.75)


def test_unconfirmed_event_keeps_its_position_open(tmp_path: Path) -> None:
    runner = _runner(tmp_path, closed=None)
    asyncio.run(runner.handle_book_update(_book("yes", bid=0.47, ask=0.48)))
    _, settled = asyncio.run(runner.refresh_markets_once())
    assert settled == []
    pnl = runner.get_heartbeat_stats()["pnl"]
    assert pnl["open_lots"] == 1 and pnl["realized_pnl_usdc"] == 0.0


def test_daily_summary_carries_the_portfolio_snapshot(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    asyncio.run(runner.handle_book_update(_book("yes", bid=0.47, ask=0.48)))
    runner._last_summary_utc_date = "2026-11-14"

    summary = runner._maybe_emit_daily_summary(now=datetime(2026, 11, 15, 0, 0, 5, tzinfo=timezone.utc))
    assert summary is not None and summary["date"] == "2026-11-14"
    assert summary["portfolio"]["locked_arb_pnl_usdc"] == pytest.approx(0.75)
    assert summary["portfolio"]["open_lots"] == 1


def test_portfolio_survives_a_restart(tmp_path: Path) -> None:
    first = _runner(tmp_path, closed=True)
    asyncio.run(first.handle_book_update(_book("yes", bid=0.47, ask=0.48)))  # fill on evt-arb
    news = _signal("evt-news", "NEWS_LATENCY", [("fed", 100.0, 0.5)], signal_id="news-restart")
    first.trade_logger.log_paper_fill(_fill(news, 51.0), signal=news, fees_usdc=1.0)

    # New process, same log: positions and fees come back, marks do not (no books yet).
    second = _runner(tmp_path)
    assert second.portfolio.open_lots == []
    assert second._restore_portfolio_from_log() == 2
    restored = second.portfolio.snapshot({})
    assert restored["open_lots"] == 2
    assert restored["cost_basis_usdc"] == pytest.approx(24.25 + 51.0)
    assert restored["fees_paid_usdc"] == pytest.approx(1.0)
    assert restored["locked_arb_pnl_usdc"] == pytest.approx(0.75)
    assert restored["unpriced_tokens"] == 3
    assert second._restore_portfolio_from_log() == 0  # never restored twice

    # A settlement written before the restart is replayed too.
    asyncio.run(first.refresh_markets_once())
    third = PaperPortfolio.from_log(tmp_path / "pnl.jsonl", starting_bankroll_usdc=500.0)
    snap = third.snapshot({})
    assert snap["realized_pnl_usdc"] == pytest.approx(0.75)
    assert snap["open_lots"] == 1 and snap["settled_lots"] == 1


def test_report_shows_the_pnl_of_the_latest_heartbeat(tmp_path: Path) -> None:
    from paper_trader.report import analyze_paper_log, format_report

    runner = _runner(tmp_path)
    asyncio.run(runner.handle_book_update(_book("yes", bid=0.47, ask=0.48)))

    before = format_report(analyze_paper_log(tmp_path / "pnl.jsonl"))
    assert "Virtual P&L" not in before  # no heartbeat logged yet

    runner.trade_logger.log_heartbeat(runner.get_heartbeat_stats())
    text = format_report(analyze_paper_log(tmp_path / "pnl.jsonl"))
    assert "Virtual P&L (latest heartbeat): equity=499.50 USDC" in text
    assert "locked arbitrage=+0.75" in text
    assert "Open lots: 1 (cost 24.25 USDC" in text


def test_from_log_tolerates_missing_and_malformed_logs(tmp_path: Path) -> None:
    assert PaperPortfolio.from_log(tmp_path / "missing.jsonl").open_lots == []
    broken = tmp_path / "broken.jsonl"
    broken.write_text('not json\n{"event_type": "PAPER_FILL"}\n{"event_type": "PAPER_FILL", "payload": {"status": "FILLED"}}\n[]\n', encoding="utf-8")
    assert PaperPortfolio.from_log(broken).snapshot({})["open_lots"] == 0

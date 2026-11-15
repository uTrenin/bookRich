"""Trading costs end to end: one taker fee rate shared by engine, simulator and log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from contracts.models import BookLevel, MarketGroup, MarketOutcome, OrderBookSnapshot, OrderLeg, TradeSignal
from paper_trader.logger import PaperTradeLogger
from paper_trader.runner import DynamicEngineAdapter, PaperTradingRunner
from paper_trader.simulator import PaperOrderSimulator

T0 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)


def _book(token_id: str, ask: float, size: float = 100.0) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id, asks=[BookLevel(price=ask, size=size)], bids=[], updated_at=datetime.now(timezone.utc)
    )


def _group() -> MarketGroup:
    return MarketGroup(
        event_id="evt-cost",
        title="Cost event",
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(market_id="m", question="Q?", outcome_label="Yes", token_id="yes"),
            MarketOutcome(market_id="m", question="Q?", outcome_label="No", token_id="no"),
        ],
    )


def _signal() -> TradeSignal:
    return TradeSignal(
        signal_id="sig-cost",
        strategy="COMBINATORIAL_ARB",
        event_id="evt-cost",
        legs=[
            OrderLeg(token_id="yes", side="BUY", target_shares=10.0, expected_vwap=0.40, max_acceptable_price=0.40),
            OrderLeg(token_id="no", side="BUY", target_shares=10.0, expected_vwap=0.50, max_acceptable_price=0.50),
        ],
        expected_net_edge=0.05,
        total_cost_usdc=9.0,
        created_at=T0,
        rationale="cost test",
    )


def test_simulator_adds_the_taker_fee_to_the_fill_cost() -> None:
    books = {"yes": _book("yes", 0.40), "no": _book("no", 0.50)}

    free = PaperOrderSimulator().simulate_signal(_signal(), books, executed_at=T0)
    assert free.actual_simulated_cost_usdc == pytest.approx(9.0)
    assert "taker fee" not in free.note

    paid = PaperOrderSimulator(fee_rate=0.02).simulate_signal(_signal(), books, executed_at=T0)
    assert paid.status == "FILLED"
    assert paid.actual_simulated_cost_usdc == pytest.approx(9.0 * 1.02)
    assert "taker fee 0.1800 USDC" in paid.note
    # Prices and slippage are untouched: the fee is a cost on top, not a worse fill.
    assert [leg.expected_vwap for leg in paid.filled_legs] == [0.40, 0.50]
    assert paid.slippage_vs_best_ask == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_simulator_rejects_an_invalid_fee_rate(bad: float) -> None:
    with pytest.raises(ValueError, match="fee_rate"):
        PaperOrderSimulator(fee_rate=bad)


def test_adapter_reads_costs_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAKER_FEE_BPS", "150")
    monkeypatch.setenv("SLIPPAGE_BUFFER", "0.005")
    adapter = DynamicEngineAdapter()
    assert adapter.fee_rate == pytest.approx(0.015)
    assert adapter.slippage_buffer == pytest.approx(0.005)

    # Explicit arguments win over the environment.
    explicit = DynamicEngineAdapter(fee_rate=0.0, slippage_buffer=0.0)
    assert explicit.fee_rate == 0.0 and explicit.slippage_buffer == 0.0

    monkeypatch.delenv("TAKER_FEE_BPS")
    monkeypatch.delenv("SLIPPAGE_BUFFER")
    defaults = DynamicEngineAdapter()
    assert defaults.fee_rate == 0.0 and defaults.slippage_buffer == 0.0


@pytest.mark.parametrize("name,value", [("TAKER_FEE_BPS", "-5"), ("SLIPPAGE_BUFFER", "nan")])
def test_adapter_refuses_nonsense_costs(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        DynamicEngineAdapter()


def test_fee_rate_reaches_engine_simulator_and_log(tmp_path: Path) -> None:
    pytest.importorskip("engine.combinatorial_arb", reason="needs the real /engine")
    log_path = tmp_path / "costs.jsonl"
    books = {"yes": _book("yes", 0.47), "no": _book("no", 0.49)}  # 0.96 per set at book prices

    adapter = DynamicEngineAdapter(
        target_shares=25.0, min_arb_net_edge=0.015, bankroll_usdc=1000.0, max_position_usdc=50.0,
        slippage_buffer=0.0, fee_rate=0.01,
    )
    runner = PaperTradingRunner(
        engine=adapter, market_groups=[_group()], books=dict(books), trade_logger=PaperTradeLogger(log_path)
    )
    # The default simulator prices fees exactly like the engine.
    assert runner.simulator.fee_rate == pytest.approx(0.01)
    assert runner.get_heartbeat_stats()["taker_fee_bps"] == pytest.approx(100.0)

    import asyncio

    fills = asyncio.run(runner.handle_book_update(books["yes"]))
    assert len(fills) == 1 and fills[0].status == "FILLED"
    notional = 25.0 * 0.96
    assert fills[0].actual_simulated_cost_usdc == pytest.approx(notional * 1.01)
    # Exposure counts what really left the bankroll, fees included.
    assert adapter.event_exposure_usdc["evt-cost"] == pytest.approx(notional * 1.01)

    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["event_type"] == "PAPER_FILL"
    assert entry["payload"]["fees_usdc"] == pytest.approx(notional * 0.01)
    assert entry["payload"]["signal_total_cost_usdc"] == pytest.approx(notional)  # book cost, fee-free
    assert entry["payload"]["expected_net_edge"] == pytest.approx(1.0 - 0.96 * 1.01)

    # With a 3% fee the same book is no longer an opportunity: 0.96 * 1.03 = 0.9888.
    expensive = DynamicEngineAdapter(target_shares=25.0, min_arb_net_edge=0.015, slippage_buffer=0.0, fee_rate=0.03)
    assert expensive.evaluate_group(_group(), books) is None


def test_daily_summary_totals_fees(tmp_path: Path) -> None:
    books = {"yes": _book("yes", 0.40), "no": _book("no", 0.50)}
    logger = PaperTradeLogger(tmp_path / "summary.jsonl")
    fill = PaperOrderSimulator(fee_rate=0.02).simulate_signal(_signal(), books, executed_at=T0)
    logger.log_paper_fill(fill, signal=_signal(), fees_usdc=0.18)
    logger.log_paper_fill(fill, signal=_signal(), fees_usdc=0.18)

    summary = logger.generate_daily_summary(date_str="2026-11-15", write_to_log=False)
    assert summary["fills_count"] == 2
    assert summary["total_fees_usdc"] == pytest.approx(0.36)
    assert summary["total_virtual_cost_usdc"] == pytest.approx(2 * 9.0 * 1.02)

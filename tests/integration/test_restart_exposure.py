"""After a restart the risk manager must still see the positions opened before it."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from contracts.models import BookLevel, MarketGroup, MarketOutcome, OrderBookSnapshot
from paper_trader.logger import PaperTradeLogger
from paper_trader.portfolio import PaperPortfolio
from paper_trader.runner import DynamicEngineAdapter, PaperTradingRunner

pytest.importorskip("engine.risk_manager", reason="needs the real /engine")


def _book(token_id: str, ask: float, bid: float) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        asks=[BookLevel(price=ask, size=500.0)],
        bids=[BookLevel(price=bid, size=500.0)],
        updated_at=datetime.now(timezone.utc),
    )


def _books() -> dict[str, OrderBookSnapshot]:
    return {"yes": _book("yes", ask=0.48, bid=0.47), "no": _book("no", ask=0.49, bid=0.48)}


def _group(event_id: str = "evt-restart") -> MarketGroup:
    return MarketGroup(
        event_id=event_id,
        title="Restart event",
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(market_id="m", question="Q?", outcome_label="Yes", token_id="yes"),
            MarketOutcome(market_id="m", question="Q?", outcome_label="No", token_id="no"),
        ],
    )


def _adapter() -> DynamicEngineAdapter:
    return DynamicEngineAdapter(
        target_shares=25.0, min_arb_net_edge=0.015, bankroll_usdc=1000.0, max_position_usdc=50.0,
        fee_rate=0.0, fee_buffer=0.0, slippage_buffer=0.0,
    )


def _runner(log_path: Path, **kwargs) -> PaperTradingRunner:
    return PaperTradingRunner(
        engine=_adapter(), market_groups=[_group()], books=_books(),
        trade_logger=PaperTradeLogger(log_path), **kwargs,
    )


async def _updates(runner: PaperTradingRunner, count: int) -> int:
    fills = 0
    for _ in range(count):
        fills += sum(1 for f in await runner.handle_book_update(_books()["yes"]) if f.status == "FILLED")
    return fills


def _events(log_path: Path) -> list[str]:
    return [json.loads(line)["event_type"] for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_restart_keeps_the_per_event_cap(tmp_path: Path) -> None:
    log_path = tmp_path / "restart.jsonl"

    # Before the restart: two fills of 24.25 USDC reach 48.5 of the 50 USDC cap, the third is refused.
    first = _runner(log_path)
    assert asyncio.run(_updates(first, 3)) == 2
    assert first.engine.event_exposure_usdc == pytest.approx({"evt-restart": 48.5})

    # A fresh process that does not restore anything has forgotten the cap and buys again.
    forgetful = _runner(tmp_path / "other.jsonl")
    assert asyncio.run(_updates(forgetful, 1)) == 1

    # A fresh process on the same log restores the exposure and keeps refusing.
    second = _runner(log_path)
    assert second.engine.event_exposure_usdc == {}
    assert second._restore_portfolio_from_log() == 2
    assert second.engine.event_exposure_usdc == pytest.approx({"evt-restart": 48.5})
    assert second.engine.current_exposure_usdc == pytest.approx(48.5)

    assert asyncio.run(_updates(second, 2)) == 0
    assert second.get_heartbeat_stats()["signals_rejected_risk"] == 2
    assert _events(log_path) == ["PAPER_FILL", "PAPER_FILL", "SIGNAL_REJECTED", "SIGNAL_REJECTED"]
    last = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["payload"]["reason"].startswith("EVENT_EXPOSURE_EXCEEDED")


def test_closed_events_are_not_restored_as_exposure(tmp_path: Path) -> None:
    log_path = tmp_path / "closed.jsonl"
    first = _runner(log_path)
    asyncio.run(_updates(first, 2))
    # The event is confirmed closed before the restart: exposure released, arbitrage realized.
    first.engine.release_event_exposure("evt-restart")
    first.trade_logger.log_event("PAPER_SETTLED", {"event_id": "evt-restart", **first.portfolio.settle_event("evt-restart")})

    second = _runner(log_path)
    second._restore_portfolio_from_log()
    assert second.engine.event_exposure_usdc == {}
    assert second.portfolio.snapshot({})["realized_pnl_usdc"] == pytest.approx(1.5)


def test_exposure_is_not_overwritten_once_the_engine_is_tracking(tmp_path: Path) -> None:
    log_path = tmp_path / "live.jsonl"
    runner = _runner(log_path)
    asyncio.run(_updates(runner, 1))  # this process already holds 24.25 USDC of exposure
    assert runner._restore_portfolio_from_log() == 0  # positions are already in memory
    assert runner._restore_engine_exposure() == 0.0
    assert runner.engine.event_exposure_usdc == pytest.approx({"evt-restart": 24.25})


def test_portfolio_exposure_by_event_leaves_out_closed_events() -> None:
    portfolio = PaperPortfolio.from_log("does-not-exist.jsonl")
    assert portfolio.exposure_by_event() == {}

    adapter = _adapter()
    assert adapter.restore_exposure({"a": 10.0, "b": "12.5", "c": 0.0, "d": float("nan"), "": 5.0, "e": None}) == pytest.approx(22.5)
    assert adapter.event_exposure_usdc == {"a": 10.0, "b": 12.5}
    assert adapter.current_exposure_usdc == pytest.approx(22.5)


@pytest.mark.asyncio
async def test_run_restores_positions_and_exposure_at_startup(tmp_path: Path) -> None:
    log_path = tmp_path / "startup.jsonl"
    await _updates(_runner(log_path), 2)  # a previous process left two fills in the log

    class _Discovery:
        async def discover_market_groups(self, limit: int = 50) -> list[MarketGroup]:
            return [_group()]

    class _IdleStream:
        token_ids = ["yes", "no"]
        on_book_update = None

        def get_all_books(self) -> dict[str, OrderBookSnapshot]:
            return _books()

        async def run(self, stop_event: asyncio.Event | None = None) -> None:
            if stop_event is not None:
                await stop_event.wait()

    stop_event = asyncio.Event()
    runner = PaperTradingRunner(
        engine=_adapter(), discovery_client=_Discovery(), book_stream=_IdleStream(),
        trade_logger=PaperTradeLogger(log_path), heartbeat_interval_sec=3600.0,
    )
    run_task = asyncio.create_task(runner.run(stop_event=stop_event))
    await asyncio.sleep(0.05)

    assert runner.engine.event_exposure_usdc == pytest.approx({"evt-restart": 48.5})
    assert runner.portfolio.snapshot({})["open_lots"] == 2
    assert await runner.handle_book_update(_books()["yes"]) == []  # the cap still holds

    stop_event.set()
    await asyncio.wait_for(run_task, timeout=2.0)
    final_pnl = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])["payload"]["pnl"]
    assert final_pnl["locked_arb_pnl_usdc"] == pytest.approx(1.5)

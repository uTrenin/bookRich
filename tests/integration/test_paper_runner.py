"""Deterministic integration and unit tests for paper_trader (zero live network calls)."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from contracts.models import (
    BookLevel,
    MarketGroup,
    MarketOutcome,
    NewsAssessment,
    OrderBookSnapshot,
    OrderLeg,
    PaperFill,
    TradeSignal,
)
from paper_trader.logger import PaperTradeLogger
from paper_trader.simulator import PaperOrderSimulator


def _import_runner_module() -> Any:
    """Import paper_trader.runner, temporarily stubbing missing collector modules pre-merge."""
    try:
        return importlib.import_module("paper_trader.runner")
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("collector"):
            raise

    inserted: list[str] = []

    class _StubDiscovery:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def discover_market_groups(self, limit: int = 100) -> list[MarketGroup]:
            return []

    class _StubBookStream:
        def __init__(self, token_ids: list[str] | None = None, on_book_update: Any = None, **kwargs: Any) -> None:
            self.token_ids = token_ids or []
            self.on_book_update = on_book_update

        def get_all_books(self, now: datetime | None = None) -> dict[str, OrderBookSnapshot]:
            return {}

        def get_book(self, token_id: str, now: datetime | None = None) -> OrderBookSnapshot | None:
            return None

        async def run(self, stop_event: asyncio.Event | None = None, ws_connect: Any = None) -> None:
            if stop_event is not None:
                await stop_event.wait()

    class _StubClassifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _StubNewsCollector:
        def __init__(
            self,
            feed_urls: list[str] | None = None,
            classifier: Any = None,
            on_assessment: Any = None,
            **kwargs: Any,
        ) -> None:
            self.feed_urls = feed_urls or []
            self.classifier = classifier
            self.on_assessment = on_assessment

        async def run(
            self,
            market_groups_provider: Any,
            stop_event: asyncio.Event | None = None,
        ) -> None:
            if stop_event is not None:
                await stop_event.wait()

    stubs: dict[str, dict[str, Any]] = {
        "collector.gamma_discovery": {"GammaDiscoveryClient": _StubDiscovery},
        "collector.clob_ws_stream": {"ClobOrderBookStream": _StubBookStream},
        "collector.news_stream": {
            "GeminiNewsClassifier": _StubClassifier,
            "NewsStreamCollector": _StubNewsCollector,
        },
    }

    try:
        for mod_name, attrs in stubs.items():
            if mod_name not in sys.modules:
                stub = types.ModuleType(mod_name)
                for attr_name, attr_val in attrs.items():
                    setattr(stub, attr_name, attr_val)
                sys.modules[mod_name] = stub
                inserted.append(mod_name)
        return importlib.import_module("paper_trader.runner")
    finally:
        for mod_name in inserted:
            sys.modules.pop(mod_name, None)


runner_mod = _import_runner_module()
DynamicEngineAdapter = runner_mod.DynamicEngineAdapter
EngineProtocol = runner_mod.EngineProtocol
PaperTradingRunner = runner_mod.PaperTradingRunner
validate_paper_only_safety = runner_mod.validate_paper_only_safety


UTC = timezone.utc


def _ts(hour: int = 12, minute: int = 0, second: int = 0, day: int = 14) -> datetime:
    return datetime(2026, 11, day, hour, minute, second, tzinfo=UTC)


def _make_book(
    token_id: str,
    asks: list[tuple[float, float]] | None = None,
    bids: list[tuple[float, float]] | None = None,
    is_stale: bool = False,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        bids=[BookLevel(price=p, size=s) for p, s in (bids or [])],
        asks=[BookLevel(price=p, size=s) for p, s in (asks or [])],
        updated_at=_ts(),
        is_stale=is_stale,
    )


def _make_signal(
    legs: list[OrderLeg],
    signal_id: str = "sig-1",
    strategy: str = "COMBINATORIAL_ARB",
    event_id: str = "evt-1",
    expected_net_edge: float = 0.03,
    total_cost_usdc: float = 10.0,
) -> TradeSignal:
    return TradeSignal(
        signal_id=signal_id,
        strategy=strategy,  # type: ignore[arg-type]
        event_id=event_id,
        legs=legs,
        expected_net_edge=expected_net_edge,
        total_cost_usdc=total_cost_usdc,
        created_at=_ts(),
        rationale="deterministic test signal",
    )


# ---------------------------------------------------------------------------
# PaperOrderSimulator tests
# ---------------------------------------------------------------------------


def test_simulator_buy_multi_level_vwap_and_slippage() -> None:
    sim = PaperOrderSimulator()
    books = {
        "tok-yes": _make_book(
            "tok-yes",
            asks=[(0.40, 10.0), (0.44, 15.0), (0.50, 20.0)],
        )
    }
    signal = _make_signal(
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="BUY",
                target_shares=30.0,
                expected_vwap=0.44,
                max_acceptable_price=0.45,
            )
        ]
    )

    fill = sim.simulate_signal(signal, books, executed_at=_ts(12, 5))

    expected_cost = 10.0 * 0.40 + 15.0 * 0.44 + 5.0 * 0.50  # 13.10
    expected_vwap = expected_cost / 30.0
    expected_slippage = expected_vwap - 0.40

    assert fill.status == "FILLED"
    assert fill.executed_at == _ts(12, 5)
    assert len(fill.filled_legs) == 1
    assert fill.filled_legs[0].expected_vwap == pytest.approx(expected_vwap)
    assert fill.actual_simulated_cost_usdc == pytest.approx(expected_cost)
    assert fill.slippage_vs_best_ask == pytest.approx(expected_slippage)


def test_simulator_multi_leg_weighted_slippage() -> None:
    sim = PaperOrderSimulator()
    books = {
        "tok-yes": _make_book("tok-yes", asks=[(0.40, 10.0), (0.46, 20.0)]),
        "tok-no": _make_book("tok-no", asks=[(0.50, 10.0)]),
    }
    signal = _make_signal(
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="BUY",
                target_shares=20.0,
                expected_vwap=0.43,
                max_acceptable_price=0.45,
            ),
            OrderLeg(
                token_id="tok-no",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.50,
                max_acceptable_price=0.52,
            ),
        ]
    )

    fill = sim.simulate_signal(signal, books, executed_at=_ts())

    yes_cost = 10.0 * 0.40 + 10.0 * 0.46  # 8.60, vwap 0.43, slip 0.03
    no_cost = 10.0 * 0.50  # 5.00, vwap 0.50, slip 0.00
    weighted_slippage = (0.03 * 20.0 + 0.00 * 10.0) / 30.0

    assert fill.status == "FILLED"
    assert fill.actual_simulated_cost_usdc == pytest.approx(yes_cost + no_cost)
    assert fill.slippage_vs_best_ask == pytest.approx(weighted_slippage)


def test_simulator_sell_bid_walk_handles_unsorted_bids() -> None:
    sim = PaperOrderSimulator()
    # Provide bids intentionally unsorted to confirm descending sort
    books = {
        "tok-yes": _make_book(
            "tok-yes",
            bids=[(0.52, 20.0), (0.60, 10.0), (0.56, 10.0)],
        )
    }
    signal = _make_signal(
        strategy="NEWS_LATENCY",
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="SELL",
                target_shares=20.0,
                expected_vwap=0.58,
                max_acceptable_price=0.57,
            )
        ],
    )

    fill = sim.simulate_signal(signal, books, executed_at=_ts())

    expected_cost = 10.0 * 0.60 + 10.0 * 0.56  # 11.60
    expected_vwap = 0.58
    expected_slippage = 0.60 - expected_vwap  # 0.02

    assert fill.status == "FILLED"
    assert fill.filled_legs[0].expected_vwap == pytest.approx(expected_vwap)
    assert fill.actual_simulated_cost_usdc == pytest.approx(expected_cost)
    assert fill.slippage_vs_best_ask == pytest.approx(expected_slippage)


def test_simulator_empty_book_returns_partial_rejected() -> None:
    sim = PaperOrderSimulator()
    books = {"tok-yes": _make_book("tok-yes", asks=[])}
    signal = _make_signal(
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.5,
                max_acceptable_price=0.6,
            )
        ]
    )

    fill = sim.simulate_signal(signal, books, executed_at=_ts())
    assert fill.status == "PARTIAL_REJECTED"
    assert fill.filled_legs == []
    assert fill.actual_simulated_cost_usdc == 0.0


def test_simulator_zero_best_ask_avoids_zero_division_and_rejects() -> None:
    sim = PaperOrderSimulator()
    books = {"tok-yes": _make_book("tok-yes", asks=[(0.0, 100.0), (0.45, 50.0)])}
    signal = _make_signal(
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.45,
                max_acceptable_price=0.50,
            )
        ]
    )

    fill = sim.simulate_signal(signal, books, executed_at=_ts())
    assert fill.status == "PARTIAL_REJECTED"
    assert fill.filled_legs == []
    assert fill.actual_simulated_cost_usdc == 0.0


def test_simulator_insufficient_depth_returns_partial_rejected() -> None:
    sim = PaperOrderSimulator()
    books = {"tok-yes": _make_book("tok-yes", asks=[(0.40, 5.0), (0.45, 4.0)])}
    signal = _make_signal(
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.42,
                max_acceptable_price=0.60,
            )
        ]
    )

    fill = sim.simulate_signal(signal, books, executed_at=_ts())
    assert fill.status == "PARTIAL_REJECTED"
    assert "Insufficient L2 depth" in fill.note


def test_simulator_max_acceptable_price_exceeded_rejects() -> None:
    sim = PaperOrderSimulator()
    books = {"tok-yes": _make_book("tok-yes", asks=[(0.40, 5.0), (0.60, 10.0)])}
    buy_sig = _make_signal(
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.45,
                max_acceptable_price=0.48,  # actual VWAP = 0.50 > 0.48
            )
        ]
    )
    buy_fill = sim.simulate_signal(buy_sig, books, executed_at=_ts())
    assert buy_fill.status == "PARTIAL_REJECTED"

    sell_books = {"tok-yes": _make_book("tok-yes", bids=[(0.60, 5.0), (0.40, 10.0)])}
    sell_sig = _make_signal(
        legs=[
            OrderLeg(
                token_id="tok-yes",
                side="SELL",
                target_shares=10.0,
                expected_vwap=0.55,
                max_acceptable_price=0.52,  # actual VWAP = 0.50 < 0.52
            )
        ]
    )
    sell_fill = sim.simulate_signal(sell_sig, sell_books, executed_at=_ts())
    assert sell_fill.status == "PARTIAL_REJECTED"


def test_simulator_stale_missing_or_empty_legs_returns_skipped_risk() -> None:
    sim = PaperOrderSimulator()
    leg = OrderLeg(
        token_id="tok-yes",
        side="BUY",
        target_shares=10.0,
        expected_vwap=0.5,
        max_acceptable_price=0.6,
    )
    signal = _make_signal(legs=[leg])

    # Missing book
    missing_fill = sim.simulate_signal(signal, {}, executed_at=_ts())
    assert missing_fill.status == "SKIPPED_RISK"

    # Stale book
    stale_fill = sim.simulate_signal(
        signal,
        {"tok-yes": _make_book("tok-yes", asks=[(0.40, 20.0)], is_stale=True)},
        executed_at=_ts(),
    )
    assert stale_fill.status == "SKIPPED_RISK"

    # No legs
    no_legs_fill = sim.simulate_signal(_make_signal(legs=[]), {}, executed_at=_ts())
    assert no_legs_fill.status == "SKIPPED_RISK"
    assert no_legs_fill.note == "Signal has no legs"


# ---------------------------------------------------------------------------
# PaperTradeLogger tests
# ---------------------------------------------------------------------------


def test_logger_append_jsonl_and_daily_summary(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "paper_trades.jsonl"
    trade_logger = PaperTradeLogger(log_path=log_file)
    assert log_file.parent.exists()

    arb_sig = _make_signal(
        signal_id="sig-arb",
        strategy="COMBINATORIAL_ARB",
        legs=[
            OrderLeg(
                token_id="tok-a",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.45,
                max_acceptable_price=0.46,
            )
        ],
    )
    arb_fill = PaperFill(
        signal_id="sig-arb",
        strategy="COMBINATORIAL_ARB",
        executed_at=_ts(10, 0, 0, day=14),
        filled_legs=arb_sig.legs,
        actual_simulated_cost_usdc=4.50,
        slippage_vs_best_ask=0.01,
        status="FILLED",
        note="ok",
    )

    news_pub = _ts(11, 0, 0, day=14)
    news_exec = news_pub + timedelta(milliseconds=1250)
    news_sig = _make_signal(
        signal_id="sig-news",
        strategy="NEWS_LATENCY",
        legs=[
            OrderLeg(
                token_id="tok-b",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.55,
                max_acceptable_price=0.58,
            )
        ],
    )
    news_fill = PaperFill(
        signal_id="sig-news",
        strategy="NEWS_LATENCY",
        executed_at=news_exec,
        filled_legs=news_sig.legs,
        actual_simulated_cost_usdc=5.50,
        slippage_vs_best_ask=0.03,
        status="FILLED",
        note="news fill",
    )

    rejected_fill = PaperFill(
        signal_id="sig-rej",
        strategy="COMBINATORIAL_ARB",
        executed_at=_ts(12, 0, 0, day=14),
        filled_legs=[],
        actual_simulated_cost_usdc=0.0,
        slippage_vs_best_ask=0.0,
        status="PARTIAL_REJECTED",
        note="depth",
    )

    other_day_fill = PaperFill(
        signal_id="sig-next-day",
        strategy="COMBINATORIAL_ARB",
        executed_at=_ts(1, 0, 0, day=15),
        filled_legs=arb_sig.legs,
        actual_simulated_cost_usdc=99.0,
        slippage_vs_best_ask=0.10,
        status="FILLED",
        note="different UTC date",
    )

    entry1 = trade_logger.log_paper_fill(arb_fill, signal=arb_sig)
    entry2 = trade_logger.log_paper_fill(news_fill, signal=news_sig, news_published_at=news_pub)
    entry3 = trade_logger.log_paper_fill(rejected_fill)
    entry4 = trade_logger.log_signal_rejected(arb_sig, "exposure_cap", ts=_ts(13, 0, 0, day=14))
    entry5 = trade_logger.log_heartbeat({"book_updates": 42}, ts=_ts(14, 0, 0, day=14))
    trade_logger.log_paper_fill(other_day_fill)

    assert entry1["event_type"] == "PAPER_FILL"
    assert entry2["payload"]["latency_ms"] == pytest.approx(1250.0)
    assert entry3["event_type"] == "SIGNAL_REJECTED"
    assert entry4["event_type"] == "SIGNAL_REJECTED"
    assert entry5["event_type"] == "HEARTBEAT_STATS"

    summary = trade_logger.generate_daily_summary(date_str="2026-11-14", write_to_log=True)
    assert summary["date"] == "2026-11-14"
    assert summary["fills_count"] == 2
    assert summary["rejections_count"] == 2
    assert summary["heartbeats_count"] == 1
    assert summary["total_virtual_cost_usdc"] == pytest.approx(10.0)
    assert summary["avg_slippage_vs_best_ask"] == pytest.approx(0.02)
    assert summary["avg_news_latency_ms"] == pytest.approx(1250.0)
    assert summary["fills_by_strategy"] == {"COMBINATORIAL_ARB": 1, "NEWS_LATENCY": 1}

    # Verify JSONL schema on disk and idempotency of DAILY_SUMMARY aggregation
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 7
    assert lines[-1]["event_type"] == "DAILY_SUMMARY"
    for row in lines:
        assert set(row.keys()) == {"ts", "event_type", "payload"}
        datetime.fromisoformat(row["ts"])

    summary_again = trade_logger.generate_daily_summary(date_str="2026-11-14", write_to_log=False)
    assert summary_again["fills_count"] == 2
    assert summary_again["total_virtual_cost_usdc"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# EngineProtocol, DynamicEngineAdapter, Safety Guards & PaperTradingRunner tests
# ---------------------------------------------------------------------------


def test_dynamic_engine_adapter_fallback_and_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = DynamicEngineAdapter()
    assert isinstance(adapter, EngineProtocol)

    group = MarketGroup(
        event_id="evt-1",
        title="Test Event",
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(market_id="m1", question="Q?", outcome_label="Yes", token_id="t1"),
            MarketOutcome(market_id="m1", question="Q?", outcome_label="No", token_id="t2"),
        ],
    )
    assessment = NewsAssessment(
        news_id="n1",
        source="rss",
        headline="Headline",
        published_at=_ts(),
        target_token_id="t1",
        estimated_fair_prob=0.7,
        confidence=0.9,
    )
    sig = _make_signal(
        legs=[
            OrderLeg(
                token_id="t1",
                side="BUY",
                target_shares=5.0,
                expected_vwap=0.45,
                max_acceptable_price=0.5,
            )
        ]
    )

    assert adapter.evaluate_group(group, {}) is None
    assert adapter.evaluate_news(assessment, None) is None
    assert adapter.approve_signal(sig, {}) == (False, "engine_not_loaded")

    # Simulate merged /engine modules via monkeypatched sys.modules
    arb_mod = types.ModuleType("engine.combinatorial_arb")
    arb_mod.evaluate_group = lambda grp, bks: sig  # type: ignore[attr-defined]

    news_mod = types.ModuleType("engine.news_latency")
    news_mod.evaluate_news = lambda assess, bk: sig  # type: ignore[attr-defined]

    risk_mod = types.ModuleType("engine.risk_manager")

    class RiskManager:
        def approve_signal(
            self, signal: TradeSignal, books: dict[str, OrderBookSnapshot]
        ) -> tuple[bool, str]:
            return True, "within_limits"

    RiskManager.__module__ = "engine.risk_manager"
    risk_mod.RiskManager = RiskManager  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "engine.combinatorial_arb", arb_mod)
    monkeypatch.setitem(sys.modules, "engine.news_latency", news_mod)
    monkeypatch.setitem(sys.modules, "engine.risk_manager", risk_mod)

    loaded_adapter = DynamicEngineAdapter()
    assert loaded_adapter.evaluate_group(group, {}) == sig
    assert loaded_adapter.evaluate_news(assessment, None) == sig
    assert loaded_adapter.approve_signal(sig, {}) == (True, "within_limits")


def test_paper_only_and_private_key_safety_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert runner_mod.PAPER_ONLY is True

    monkeypatch.setenv("PRIVATE_KEY", "0x0123456789abcdef")
    with pytest.raises(RuntimeError, match="PRIVATE_KEY"):
        validate_paper_only_safety()

    with pytest.raises(RuntimeError, match="PRIVATE_KEY"):
        PaperTradingRunner(log_path=tmp_path / "blocked.jsonl")

    monkeypatch.delenv("PRIVATE_KEY", raising=False)
    monkeypatch.setattr(runner_mod, "PAPER_ONLY", False)
    with pytest.raises((AssertionError, RuntimeError)):
        validate_paper_only_safety()


class _MockEngine:
    def __init__(self, approve: bool = True, reason: str = "ok") -> None:
        self.approve = approve
        self.reason = reason
        self.group_calls = 0
        self.news_calls = 0

    def evaluate_group(
        self, group: MarketGroup, books: dict[str, OrderBookSnapshot]
    ) -> TradeSignal | None:
        self.group_calls += 1
        yes_token = group.outcomes[0].token_id
        no_token = group.outcomes[1].token_id
        if yes_token not in books or no_token not in books:
            return None
        return _make_signal(
            signal_id=f"arb-{group.event_id}",
            strategy="COMBINATORIAL_ARB",
            event_id=group.event_id,
            legs=[
                OrderLeg(
                    token_id=yes_token,
                    side="BUY",
                    target_shares=10.0,
                    expected_vwap=0.45,
                    max_acceptable_price=0.48,
                ),
                OrderLeg(
                    token_id=no_token,
                    side="BUY",
                    target_shares=10.0,
                    expected_vwap=0.48,
                    max_acceptable_price=0.50,
                ),
            ],
        )

    def evaluate_news(
        self, assessment: NewsAssessment, book: OrderBookSnapshot | None
    ) -> TradeSignal | None:
        self.news_calls += 1
        if book is None or not book.asks:
            return None
        return _make_signal(
            signal_id=f"news-{assessment.news_id}",
            strategy="NEWS_LATENCY",
            event_id="evt-news",
            legs=[
                OrderLeg(
                    token_id=assessment.target_token_id,
                    side="BUY",
                    target_shares=10.0,
                    expected_vwap=0.50,
                    max_acceptable_price=0.55,
                )
            ],
        )

    def approve_signal(
        self, signal: TradeSignal, books: dict[str, OrderBookSnapshot]
    ) -> tuple[bool, str]:
        return self.approve, self.reason


@pytest.mark.asyncio
async def test_runner_handle_book_and_news_updates(tmp_path: Path) -> None:
    log_file = tmp_path / "paper_trades.jsonl"
    group = MarketGroup(
        event_id="evt-arb",
        title="Binary Arb Event",
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(market_id="m1", question="Q", outcome_label="Yes", token_id="tok-yes"),
            MarketOutcome(market_id="m1", question="Q", outcome_label="No", token_id="tok-no"),
        ],
    )
    initial_no = _make_book("tok-no", asks=[(0.48, 25.0)])
    yes_update = _make_book("tok-yes", asks=[(0.44, 5.0), (0.46, 15.0)])

    runner = PaperTradingRunner(
        discovery_client=object(),
        engine=_MockEngine(approve=True),
        trade_logger=PaperTradeLogger(log_path=log_file),
        market_groups=[group],
        books={"tok-no": initial_no},
    )

    fills = await runner.handle_book_update(yes_update)
    assert len(fills) == 1
    assert fills[0].status == "FILLED"
    assert fills[0].actual_simulated_cost_usdc == pytest.approx(
        (5.0 * 0.44 + 5.0 * 0.46) + (10.0 * 0.48)
    )

    published_at = datetime.now(UTC) - timedelta(seconds=1)
    assessment = NewsAssessment(
        news_id="n-100",
        source="reuters",
        headline="Breaking positive news",
        published_at=published_at,
        target_token_id="tok-yes",
        estimated_fair_prob=0.80,
        confidence=0.95,
    )
    news_fill = await runner.handle_news_assessment(assessment)
    assert news_fill is not None
    assert news_fill.status == "FILLED"

    # Risk rejection path
    reject_runner = PaperTradingRunner(
        discovery_client=object(),
        engine=_MockEngine(approve=False, reason="max_exposure_reached"),
        trade_logger=PaperTradeLogger(log_path=log_file),
        market_groups=[group],
        books={"tok-no": initial_no},
    )
    rejected_fills = await reject_runner.handle_book_update(yes_update)
    assert rejected_fills == []

    entries = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [e["event_type"] for e in entries] == [
        "PAPER_FILL",
        "PAPER_FILL",
        "SIGNAL_REJECTED",
    ]
    assert entries[0]["payload"]["signal_id"] == "arb-evt-arb"
    assert entries[1]["payload"]["signal_id"] == "news-n-100"
    assert entries[1]["payload"]["latency_ms"] >= 500.0
    assert entries[2]["payload"]["reason"] == "max_exposure_reached"


@pytest.mark.asyncio
async def test_runner_run_lifecycle_and_heartbeat(tmp_path: Path) -> None:
    log_file = tmp_path / "lifecycle.jsonl"
    group = MarketGroup(
        event_id="evt-life",
        title="Lifecycle Event",
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(market_id="m1", question="Q", outcome_label="Yes", token_id="t-yes"),
            MarketOutcome(market_id="m1", question="Q", outcome_label="No", token_id="t-no"),
        ],
    )

    class _FakeDiscovery:
        def __init__(self) -> None:
            self.called_with_limit: int | None = None

        async def discover_market_groups(self, limit: int = 50) -> list[MarketGroup]:
            self.called_with_limit = limit
            return [group]

    class _FakeBookStream:
        def __init__(self) -> None:
            self.token_ids = ["t-yes", "t-no"]
            self.on_book_update = None
            self.started = False

        def get_all_books(self) -> dict[str, OrderBookSnapshot]:
            return {}

        async def run(self, stop_event: asyncio.Event | None = None) -> None:
            self.started = True
            if self.on_book_update is not None:
                await self.on_book_update(_make_book("t-no", asks=[(0.48, 20.0)]))
                await self.on_book_update(_make_book("t-yes", asks=[(0.45, 20.0)]))
            if stop_event is not None:
                await stop_event.wait()

    class _FakeNewsCollector:
        def __init__(self) -> None:
            self.feed_urls = ["https://example.invalid/rss"]
            self.on_assessment = None
            self.started = False

        async def run(
            self,
            market_groups_provider: Any,
            stop_event: asyncio.Event | None = None,
        ) -> None:
            self.started = True
            assert len(market_groups_provider()) == 1
            if stop_event is not None:
                await stop_event.wait()

    discovery = _FakeDiscovery()
    book_stream = _FakeBookStream()
    news_collector = _FakeNewsCollector()
    stop_event = asyncio.Event()

    runner = PaperTradingRunner(
        discovery_client=discovery,
        book_stream=book_stream,
        news_collector=news_collector,
        engine=_MockEngine(approve=True),
        trade_logger=PaperTradeLogger(log_path=log_file),
        heartbeat_interval_sec=0.05,
    )

    async def _stop_soon() -> None:
        await asyncio.sleep(0.15)
        stop_event.set()

    stopper = asyncio.create_task(_stop_soon())
    await runner.run(stop_event=stop_event, discovery_limit=25)
    await stopper

    assert discovery.called_with_limit == 25
    assert book_stream.started is True
    assert news_collector.started is True

    entries = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    event_types = [e["event_type"] for e in entries]
    assert "PAPER_FILL" in event_types
    assert "HEARTBEAT_STATS" in event_types

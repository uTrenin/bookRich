"""The cost of one book update must not grow with the number of books followed.

Found on the real recording of 2026-11-18 (20 active events, 540 books, about 590 WebSocket
messages per second): the runner copied every tracked book on each update, fell behind the
feed and lost the connection every 44 seconds. These tests count the reads, they do not
measure time, so they stay deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pytest

from collector.clob_ws_stream import ClobOrderBookStream
from contracts.models import (
    BookLevel,
    MarketGroup,
    MarketOutcome,
    NewsAssessment,
    OrderBookSnapshot,
    TradeSignal,
)
from paper_trader.logger import PaperTradeLogger
from paper_trader.runner import DynamicEngineAdapter, PaperTradingRunner, _LiveBooks


def _book(token_id: str, ask: float, size: float = 100.0) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        asks=[BookLevel(price=ask, size=size)],
        bids=[BookLevel(price=max(ask - 0.02, 0.01), size=size)],
        updated_at=datetime.now(timezone.utc),
    )


def _group(event_id: str, token_ids: list[str], exclusive: bool = True) -> MarketGroup:
    return MarketGroup(
        event_id=event_id,
        title=f"Event {event_id}",
        mutually_exclusive=exclusive,
        outcomes=[
            MarketOutcome(market_id=f"m-{tid}", question="Q?", outcome_label=tid, token_id=tid)
            for tid in token_ids
        ],
    )


class _CountingStream(ClobOrderBookStream):
    """The real stream, counting how the runner reads it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.single_reads: list[str] = []
        self.full_copies = 0

    def get_book(self, token_id: str, now: datetime | None = None) -> OrderBookSnapshot | None:
        self.single_reads.append(token_id)
        return super().get_book(token_id, now=now)

    def get_all_books(self, now: datetime | None = None) -> dict[str, OrderBookSnapshot]:
        self.full_copies += 1
        reads_before = list(self.single_reads)
        books = super().get_all_books(now=now)
        self.single_reads = reads_before  # the full copy is counted on its own
        return books


def _stream(asks: dict[str, float]) -> _CountingStream:
    stream = _CountingStream(token_ids=list(asks))
    for token_id, ask in asks.items():
        stream.apply_ws_message(
            {
                "event_type": "book",
                "asset_id": token_id,
                "asks": [{"price": str(ask), "size": "100"}],
                "bids": [{"price": "0.01", "size": "100"}],
            }
        )
    return stream


def _engine() -> DynamicEngineAdapter:
    return DynamicEngineAdapter(
        target_shares=10.0,
        min_arb_net_edge=0.015,
        bankroll_usdc=1000.0,
        max_position_usdc=50.0,
        fee_buffer=0.0,
        slippage_buffer=0.0,
        fee_rate=0.0,
    )


def _setup(
    tmp_path: Path, small_ask: float, unrelated_books: int = 0, engine: Any = None
) -> tuple[PaperTradingRunner, _CountingStream]:
    """One small exclusive group, one large non-exclusive group and unrelated exclusive pairs."""
    small = ["a1", "a2", "a3"]
    sports = [f"s{i}" for i in range(40)]
    groups = [_group("evt-small", small), _group("evt-sports", sports, exclusive=False)]
    asks = {tid: small_ask for tid in small} | {tid: 0.50 for tid in sports}
    for i in range(unrelated_books // 2):
        pair = [f"u{i}-yes", f"u{i}-no"]
        groups.append(_group(f"evt-u{i}", pair))
        asks |= {pair[0]: 0.55, pair[1]: 0.55}
    stream = _stream(asks)
    runner = PaperTradingRunner(
        engine=engine if engine is not None else _engine(),
        book_stream=stream,
        market_groups=groups,
        trade_logger=PaperTradeLogger(log_path=tmp_path / "sync.jsonl"),
    )
    return runner, stream


def _update(stream: _CountingStream, token_id: str, ask: float) -> OrderBookSnapshot:
    """Apply a real price_change to the stream and return the snapshot it would emit."""
    updated = stream.apply_ws_message(
        {
            "event_type": "price_change",
            "price_changes": [{"asset_id": token_id, "price": str(ask), "side": "SELL", "size": "100"}],
        }
    )
    assert [snapshot.token_id for snapshot in updated] == [token_id]
    return updated[0]


# ---------------------------------------------------------------------------
# cost of one update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("unrelated_books", [0, 600])
async def test_book_update_reads_only_the_other_books_of_its_group(tmp_path: Path, unrelated_books: int) -> None:
    runner, stream = _setup(tmp_path, small_ask=0.40, unrelated_books=unrelated_books)

    fills = await runner.handle_book_update(_update(stream, "a1", 0.41))

    assert fills == []  # 0.41 + 0.40 + 0.40: no arbitrage
    assert stream.full_copies == 0
    # Same two reads whether the bot follows 43 books or 643.
    assert stream.single_reads == ["a2", "a3"]


@pytest.mark.asyncio
async def test_update_on_a_non_exclusive_group_reads_no_book_at_all(tmp_path: Path) -> None:
    runner, stream = _setup(tmp_path, small_ask=0.40, unrelated_books=100)

    for i in range(40):
        assert await runner.handle_book_update(_update(stream, f"s{i}", 0.51)) == []

    # The engine discards a non-exclusive group before looking at prices: nothing to read.
    assert stream.full_copies == 0
    assert stream.single_reads == []
    assert runner.get_heartbeat_stats()["book_updates"] == 40


@pytest.mark.asyncio
async def test_update_for_a_token_outside_every_group_reads_nothing(tmp_path: Path) -> None:
    runner, stream = _setup(tmp_path, small_ask=0.40)

    assert await runner.handle_book_update(_book("not-followed", 0.10)) == []

    assert stream.full_copies == 0
    assert stream.single_reads == []


@pytest.mark.asyncio
async def test_news_assessment_reads_only_the_target_book(tmp_path: Path) -> None:
    seen: list[OrderBookSnapshot | None] = []

    class _NewsEngine:
        def evaluate_group(self, group: MarketGroup, books: Mapping[str, OrderBookSnapshot]) -> TradeSignal | None:
            return None

        def evaluate_news(self, assessment: NewsAssessment, book: OrderBookSnapshot | None) -> TradeSignal | None:
            seen.append(book)
            return None

        def approve_signal(self, signal: TradeSignal, books: Mapping[str, OrderBookSnapshot]) -> tuple[bool, str]:
            return False, "not used"

    runner, stream = _setup(tmp_path, small_ask=0.40, unrelated_books=100, engine=_NewsEngine())
    assessment = NewsAssessment(
        news_id="n1",
        source="test",
        headline="Headline",
        published_at=datetime.now(timezone.utc),
        target_token_id="a2",
        estimated_fair_prob=0.9,
        confidence=0.9,
    )

    assert await runner.handle_news_assessment(assessment) is None

    assert stream.full_copies == 0
    assert stream.single_reads == ["a2"]
    assert seen[0] is not None and seen[0].token_id == "a2" and seen[0].asks[0].price == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# same decisions as before
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arbitrage_is_still_found_with_the_other_legs_read_from_the_stream(tmp_path: Path) -> None:
    runner, stream = _setup(tmp_path, small_ask=0.30, unrelated_books=20)

    fills = await runner.handle_book_update(_update(stream, "a1", 0.29))

    assert len(fills) == 1 and fills[0].status == "FILLED"
    assert sorted(leg.token_id for leg in fills[0].filled_legs) == ["a1", "a2", "a3"]
    assert fills[0].actual_simulated_cost_usdc == pytest.approx(10.0 * (0.29 + 0.30 + 0.30))
    # After a fill the portfolio marks every position: that is the one moment a full copy is due.
    assert stream.full_copies == 1


@pytest.mark.asyncio
async def test_a_stale_leg_in_the_stream_still_blocks_the_signal(tmp_path: Path) -> None:
    runner, stream = _setup(tmp_path, small_ask=0.30)
    stream.mark_all_stale()  # what a disconnect does

    # A fresh snapshot for one leg is not enough: the other two are stale in the stream.
    assert await runner.handle_book_update(_book("a1", 0.29)) == []
    assert runner.get_heartbeat_stats()["signals_generated"] == 0


@pytest.mark.asyncio
async def test_the_snapshot_that_triggered_the_update_wins_over_the_stream(tmp_path: Path) -> None:
    runner, stream = _setup(tmp_path, small_ask=0.30)

    # The stream still says 0.30 for a1 (arbitrage), the update being handled says 0.90.
    assert await runner.handle_book_update(_book("a1", 0.90)) == []
    assert "a1" not in stream.single_reads


@pytest.mark.asyncio
async def test_a_stream_without_single_book_reads_still_works(tmp_path: Path) -> None:
    class _CopyOnlyStream:
        """Older test doubles expose only the full copy."""

        def __init__(self) -> None:
            self.full_copies = 0

        def get_all_books(self) -> dict[str, OrderBookSnapshot]:
            self.full_copies += 1
            return {tid: _book(tid, 0.30) for tid in ("a1", "a2", "a3")}

    stream = _CopyOnlyStream()
    runner = PaperTradingRunner(
        engine=_engine(),
        book_stream=stream,
        market_groups=[_group("evt-small", ["a1", "a2", "a3"])],
        trade_logger=PaperTradeLogger(log_path=tmp_path / "copy_only.jsonl"),
    )

    fills = await runner.handle_book_update(_book("a1", 0.29))

    assert len(fills) == 1 and fills[0].status == "FILLED"
    assert stream.full_copies >= 1


@pytest.mark.asyncio
async def test_heartbeat_still_sees_every_book(tmp_path: Path) -> None:
    runner, stream = _setup(tmp_path, small_ask=0.40, unrelated_books=10)
    await runner.handle_book_update(_update(stream, "a1", 0.41))
    assert stream.full_copies == 0

    stats = runner.get_heartbeat_stats()

    # 3 small + 40 sports + 10 unrelated: the heartbeat refreshes them all, once.
    assert stats["tracked_books"] == 53
    assert stream.full_copies == 1


# ---------------------------------------------------------------------------
# the mapping handed to engine, risk check and simulator
# ---------------------------------------------------------------------------


def test_live_books_reads_lazily_and_behaves_like_a_read_only_dict() -> None:
    store = {tid: _book(tid, ask) for tid, ask in (("a", 0.1), ("b", 0.2), ("c", 0.3))}
    single_reads: list[str] = []
    full_reads: list[int] = []

    def read_one(token_id: str) -> OrderBookSnapshot | None:
        single_reads.append(token_id)
        return store.get(token_id)

    def read_all() -> dict[str, OrderBookSnapshot]:
        full_reads.append(1)
        return store

    pinned = _book("a", 0.9)
    view = _LiveBooks(read_one, read_all, {"a": pinned})

    # Pinned entry: never read from the source.
    assert view["a"] is pinned and view.get("a") is pinned and "a" in view
    assert single_reads == []

    # One read per token, however many times it is asked for.
    assert view.get("b") is store["b"] and view["b"] is store["b"] and "b" in view
    assert single_reads == ["b"]

    # A missing token is a miss in every form, and is not asked for twice.
    assert view.get("missing") is None and view.get("missing", "fallback") == "fallback"
    assert "missing" not in view
    with pytest.raises(KeyError):
        view["missing"]
    assert single_reads == ["b", "missing"]
    assert view.get(123) is None and view.get(["not", "hashable"]) is None

    # Walking the mapping is the only thing that loads everything; pinned entries survive it.
    assert full_reads == []
    assert len(view) == 3 and sorted(view) == ["a", "b", "c"]
    assert dict(view) == {"a": pinned, "b": store["b"], "c": store["c"]}
    assert full_reads == [1]
    assert view.get("c") is store["c"] and view.get("missing") is None
    assert single_reads == ["b", "missing"]

"""Regression tests for the cross-review fixes on PR #1 (zero live network calls)."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx
import pytest

from collector.clob_ws_stream import ClobOrderBookStream
from collector.news_stream import GeminiNewsClassifier, NewsStreamCollector, RawNewsItem
from contracts.models import (
    BookLevel,
    MarketGroup,
    MarketOutcome,
    NewsAssessment,
    OrderBookSnapshot,
    OrderLeg,
    TradeSignal,
)
from paper_trader.logger import PaperTradeLogger
from paper_trader.runner import PaperTradingRunner
from paper_trader.simulator import PaperOrderSimulator

T0 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)


def _book(token_id: str, asks: list[tuple[float, float]] = (), bids: list[tuple[float, float]] = ()) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        asks=[BookLevel(price=p, size=s) for p, s in asks],
        bids=[BookLevel(price=p, size=s) for p, s in bids],
        updated_at=T0,
    )


def _group(*token_ids: str, event_id: str = "evt-fix") -> MarketGroup:
    return MarketGroup(
        event_id=event_id,
        title="Review fix event",
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(market_id=f"m-{tid}", question="Q?", outcome_label=tid, token_id=tid)
            for tid in token_ids
        ],
    )


def _signal(legs: list[OrderLeg], event_id: str = "evt-fix", signal_id: str = "sig-fix") -> TradeSignal:
    return TradeSignal(
        signal_id=signal_id,
        strategy="COMBINATORIAL_ARB",
        event_id=event_id,
        legs=legs,
        expected_net_edge=0.03,
        total_cost_usdc=10.0,
        created_at=T0,
        rationale="review fix test",
    )


def _book_event(token_id: str, ask: str = "0.50") -> dict[str, Any]:
    return {
        "event_type": "book",
        "asset_id": token_id,
        "asks": [{"price": ask, "size": "100"}],
        "bids": [{"price": "0.40", "size": "100"}],
    }


def _delta(token_id: str, price: str, size: str, side: str = "SELL", timestamp: datetime | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_type": "price_change",
        "asset_id": token_id,
        "changes": [{"price": price, "side": side, "size": size}],
    }
    if timestamp is not None:
        event["timestamp"] = str(int(timestamp.timestamp() * 1000))
    return event


# ---------------------------------------------------------------------------
# collector/clob_ws_stream.py
# ---------------------------------------------------------------------------


def test_ws_delta_never_refreshes_a_stale_or_missing_book() -> None:
    stream = ClobOrderBookStream(token_ids=["tok"])
    stream.apply_ws_message(_book_event("tok"), now=T0)
    stream.mark_all_stale()  # disconnection: deltas may have been missed

    assert stream.apply_ws_message(_delta("tok", "0.60", "5"), now=T0) == []
    stale = stream.get_book("tok", now=T0)
    assert stale is not None and stale.is_stale is True
    assert [(lvl.price, lvl.size) for lvl in stale.asks] == [(0.50, 100.0)]  # delta not applied

    # A delta for a token that never had a snapshot must not create a one-level book.
    assert stream.apply_ws_message(_delta("never-seen", "0.10", "3"), now=T0) == []
    assert stream.get_book("never-seen", now=T0) is None
    # Only the delta for the token we follow counts as ignored: the other one is not ours.
    assert stream.ignored_delta_count == 1

    # Only a full `book` snapshot makes the book fresh again; deltas then apply normally.
    stream.apply_ws_message(_book_event("tok", ask="0.52"), now=T0)
    updated = stream.apply_ws_message(_delta("tok", "0.60", "5"), now=T0)
    assert len(updated) == 1 and updated[0].is_stale is False
    assert [(lvl.price, lvl.size) for lvl in updated[0].asks] == [(0.52, 100.0), (0.60, 5.0)]


def test_ws_quiet_token_stays_fresh_while_the_connection_is_alive() -> None:
    stream = ClobOrderBookStream(token_ids=["yes", "no"], stale_threshold_sec=30.0)
    stream.apply_ws_message(_book_event("yes"), now=T0)
    stream.apply_ws_message(_book_event("no"), now=T0)

    later = T0 + timedelta(seconds=31)
    stream.apply_ws_message(_delta("yes", "0.50", "101"), now=later)  # only `yes` changes

    view = stream.get_all_books(now=later)
    assert view["yes"].is_stale is False
    assert view["no"].is_stale is False  # unchanged book on a live connection is valid

    silent = stream.get_all_books(now=later + timedelta(seconds=30.5))
    assert silent["yes"].is_stale is True and silent["no"].is_stale is True


def test_ws_feed_lag_marks_books_stale_until_caught_up() -> None:
    stream = ClobOrderBookStream(token_ids=["tok"], stale_threshold_sec=30.0)
    stream.apply_ws_message(_book_event("tok"), now=T0)

    # Processing an event generated 10 minutes ago: we are draining a backlog.
    stream.apply_ws_message(_delta("tok", "0.60", "5", timestamp=T0 - timedelta(minutes=10)), now=T0)
    assert stream.feed_lag_sec == pytest.approx(600.0)
    assert stream.get_book("tok", now=T0).is_stale is True  # type: ignore[union-attr]

    # Caught up: a current event clears the lag.
    t1 = T0 + timedelta(seconds=1)
    stream.apply_ws_message(_delta("tok", "0.61", "5", timestamp=t1 - timedelta(milliseconds=200)), now=t1)
    assert stream.feed_lag_sec == pytest.approx(0.2)
    assert stream.get_book("tok", now=t1).is_stale is False  # type: ignore[union-attr]

    # An old measurement expires once events stop flowing and the connection stays alive.
    stream.apply_ws_message(_delta("tok", "0.62", "5", timestamp=t1 - timedelta(minutes=10)), now=t1)
    t2 = t1 + timedelta(seconds=31)
    stream.apply_ws_message("PONG", now=t2)
    assert stream.get_book("tok", now=t2).is_stale is False  # type: ignore[union-attr]


class _IdleWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._blocker: asyncio.Future[str] | None = None

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        self._blocker = asyncio.get_running_loop().create_future()
        return await self._blocker

    async def close(self) -> None:
        if self._blocker is not None and not self._blocker.done():
            self._blocker.cancel()


@pytest.mark.asyncio
async def test_ws_run_survives_rest_snapshot_failures_at_startup() -> None:
    def rest_handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["token_id"] == "b":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"bids": [], "asks": [{"price": "0.5", "size": "1"}]})

    stop_event = asyncio.Event()
    sockets: list[_IdleWebSocket] = []

    @asynccontextmanager
    async def fake_ws_connect(url: str) -> AsyncIterator[_IdleWebSocket]:
        ws = _IdleWebSocket()
        sockets.append(ws)
        stop_event.set()
        yield ws

    async with httpx.AsyncClient(transport=httpx.MockTransport(rest_handler)) as client:
        stream = ClobOrderBookStream(token_ids=["a", "b", "c"], http_client=client)
        await asyncio.wait_for(stream.run(stop_event=stop_event, ws_connect=fake_ws_connect), timeout=2.0)

    # The failing token did not kill run(): the WS was still opened and subscribed.
    assert len(sockets) == 1
    assert json.loads(sockets[0].sent[0]) == {"assets_ids": ["a", "b", "c"], "type": "market"}
    assert stream.get_book("a") is not None and stream.get_book("c") is not None
    assert stream.get_book("b") is None


@pytest.mark.asyncio
async def test_ws_resync_refetches_only_books_without_a_fresh_snapshot() -> None:
    requested: list[str] = []

    def rest_handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.params["token_id"])
        return httpx.Response(200, json={"bids": [], "asks": [{"price": "0.7", "size": "9"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(rest_handler)) as client:
        stream = ClobOrderBookStream(
            token_ids=["fresh", "stale", "missing"], http_client=client, resync_grace_sec=0.0
        )
        stream.apply_ws_message(_book_event("fresh"), now=T0)
        stream.apply_ws_message(_book_event("stale"), now=T0)
        stream.mark_all_stale()
        stream.apply_ws_message(_book_event("fresh"), now=T0)  # WS re-snapshotted this one

        await stream._resync_after_grace(stop_event=None)

    assert sorted(requested) == ["missing", "stale"]
    for token_id in ("stale", "missing"):
        snapshot = stream._books[token_id]
        assert snapshot.is_stale is False
        assert [(lvl.price, lvl.size) for lvl in snapshot.asks] == [(0.7, 9.0)]


# ---------------------------------------------------------------------------
# paper_trader/simulator.py
# ---------------------------------------------------------------------------


def test_simulator_never_fills_beyond_the_leg_limit_price() -> None:
    sim = PaperOrderSimulator()

    # Only 15 shares rest at or below the 0.44 limit; the average of a 20-share walk (0.425)
    # would still be under the limit, but a limit order cannot trade the 0.50 level.
    buy = _signal([OrderLeg(token_id="tok", side="BUY", target_shares=20.0, expected_vwap=0.42, max_acceptable_price=0.44)])
    fill = sim.simulate_signal(buy, {"tok": _book("tok", asks=[(0.40, 15.0), (0.50, 100.0)])}, executed_at=T0)
    assert fill.status == "PARTIAL_REJECTED"
    assert "within max_acceptable_price" in fill.note
    assert fill.actual_simulated_cost_usdc == 0.0

    enough = sim.simulate_signal(buy, {"tok": _book("tok", asks=[(0.40, 15.0), (0.44, 5.0), (0.50, 100.0)])}, executed_at=T0)
    assert enough.status == "FILLED"
    assert enough.actual_simulated_cost_usdc == pytest.approx(15.0 * 0.40 + 5.0 * 0.44)

    sell = _signal([OrderLeg(token_id="tok", side="SELL", target_shares=20.0, expected_vwap=0.58, max_acceptable_price=0.56)])
    rejected = sim.simulate_signal(sell, {"tok": _book("tok", bids=[(0.60, 15.0), (0.50, 100.0)])}, executed_at=T0)
    assert rejected.status == "PARTIAL_REJECTED"


# ---------------------------------------------------------------------------
# collector/news_stream.py
# ---------------------------------------------------------------------------


def _fresh_items(count: int) -> list[RawNewsItem]:
    now = datetime.now(timezone.utc)
    return [
        RawNewsItem(news_id=f"news-{i}", source="test", headline=f"Headline {i}", published_at=now)
        for i in range(count)
    ]


def _assessment_for(item: RawNewsItem) -> NewsAssessment:
    return NewsAssessment(
        news_id=item.news_id,
        source=item.source,
        headline=item.headline,
        published_at=item.published_at,
        target_token_id="tok",
        estimated_fair_prob=0.7,
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_news_poll_classifies_concurrently_with_a_bound() -> None:
    running = 0
    peak = 0

    async def classifier(item: RawNewsItem, groups: list[MarketGroup]) -> NewsAssessment:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return _assessment_for(item)

    items = _fresh_items(8)
    collector = NewsStreamCollector(feed_urls=["feed"], classifier=classifier, max_concurrent_classifications=3)

    async def fake_fetch(url: str, source_name: str | None = None) -> list[RawNewsItem]:
        return items

    collector.fetch_feed = fake_fetch  # type: ignore[method-assign]
    assessments = await collector.poll_once([_group("tok")])

    assert [a.news_id for a in assessments] == [item.news_id for item in items]  # order preserved
    assert peak == 3  # concurrent, but never above the configured bound


@pytest.mark.asyncio
async def test_news_feeds_are_fetched_concurrently_and_failures_are_isolated() -> None:
    started: list[str] = []
    release = asyncio.Event()

    collector = NewsStreamCollector(feed_urls=["slow", "broken", "fast"], classifier=lambda item, groups: None)

    async def fake_fetch(url: str, source_name: str | None = None) -> list[RawNewsItem]:
        started.append(url)
        if url == "broken":
            raise httpx.ConnectError("boom")
        if url == "slow":
            await release.wait()
        return []

    collector.fetch_feed = fake_fetch  # type: ignore[method-assign]
    poll = asyncio.create_task(collector.poll_once([_group("tok")]))
    await asyncio.sleep(0.01)
    assert started == ["slow", "broken", "fast"]  # nobody waited for the slow feed
    release.set()
    assert await asyncio.wait_for(poll, timeout=1.0) == []


@pytest.mark.asyncio
async def test_news_callback_error_does_not_abort_the_poll() -> None:
    handled: list[str] = []

    def exploding_callback(assessment: NewsAssessment) -> None:
        handled.append(assessment.news_id)
        raise TypeError("handler failed")

    items = _fresh_items(5)
    collector = NewsStreamCollector(
        feed_urls=["feed"],
        classifier=lambda item, groups: _assessment_for(item),
        on_assessment=exploding_callback,
    )

    async def fake_fetch(url: str, source_name: str | None = None) -> list[RawNewsItem]:
        return items

    collector.fetch_feed = fake_fetch  # type: ignore[method-assign]
    assessments = await collector.poll_once([_group("tok")])

    assert len(assessments) == 5
    assert sorted(handled) == [item.news_id for item in items]


@pytest.mark.asyncio
async def test_classifier_bounds_prompt_candidates_and_skips_unrelated_news() -> None:
    groups = [
        MarketGroup(
            event_id=f"evt-{i}",
            title=f"Generic market number {i}",
            mutually_exclusive=True,
            outcomes=[
                MarketOutcome(market_id=f"m{i}", question=f"Will thing {i} happen?", outcome_label="Yes", token_id=f"tok-{i}-yes"),
                MarketOutcome(market_id=f"m{i}", question=f"Will thing {i} happen?", outcome_label="No", token_id=f"tok-{i}-no"),
            ],
        )
        for i in range(60)
    ]
    groups.append(
        MarketGroup(
            event_id="evt-fed",
            title="Federal Reserve December decision",
            mutually_exclusive=True,
            outcomes=[
                MarketOutcome(market_id="fed", question="Will the Fed cut rates?", outcome_label="Cut", token_id="tok-fed-cut"),
                MarketOutcome(market_id="fed", question="Will the Fed cut rates?", outcome_label="Hold", token_id="tok-fed-hold"),
            ],
        )
    )
    prompts: list[str] = []

    async def fake_generate(**kwargs: Any) -> Any:
        prompts.append(kwargs["contents"])
        return SimpleNamespace(text='{"target_token_id": "tok-fed-cut", "estimated_fair_prob": 0.9, "confidence": 0.9}')

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate)))
    classifier = GeminiNewsClassifier(client=client, max_prompt_candidates=10)
    now = datetime.now(timezone.utc)

    relevant = RawNewsItem(news_id="n1", source="s", headline="Fed cuts rates by 25 bps", published_at=now)
    assessment = await classifier.classify(relevant, groups)
    assert assessment is not None and assessment.target_token_id == "tok-fed-cut"
    assert prompts[0].count("- token_id=") <= 10
    assert "tok-fed-cut" in prompts[0] and "tok-0-yes" not in prompts[0]

    unrelated = RawNewsItem(news_id="n2", source="s", headline="Volcano erupts overseas", published_at=now)
    assert await classifier.classify(unrelated, groups) is None
    assert len(prompts) == 1  # no LLM call when nothing matches

    # A token left out of the prompt is never accepted, even if the model names it.
    async def off_list(**kwargs: Any) -> Any:
        return SimpleNamespace(text='{"target_token_id": "tok-0-yes", "estimated_fair_prob": 0.9, "confidence": 0.9}')

    client.aio.models.generate_content = off_list
    assert await classifier.classify(relevant, groups) is None


# ---------------------------------------------------------------------------
# paper_trader/runner.py
# ---------------------------------------------------------------------------


class _StubEngine:
    def __init__(self, approve: bool = True, reason: str = "ok") -> None:
        self.approve = approve
        self.reason = reason

    def evaluate_group(self, group: MarketGroup, books: dict[str, OrderBookSnapshot]) -> TradeSignal | None:
        return _signal(
            [OrderLeg(token_id=o.token_id, side="BUY", target_shares=10.0, expected_vwap=0.48, max_acceptable_price=0.50) for o in group.outcomes],
            event_id=group.event_id,
        )

    def evaluate_news(self, assessment: NewsAssessment, book: OrderBookSnapshot | None) -> TradeSignal | None:
        return None

    def approve_signal(self, signal: TradeSignal, books: dict[str, OrderBookSnapshot]) -> tuple[bool, str]:
        return self.approve, self.reason


class _OneGroupDiscovery:
    async def discover_market_groups(self, limit: int = 50) -> list[MarketGroup]:
        return [_group("yes", "no")]


def _runner(tmp_path: Path, **kwargs: Any) -> PaperTradingRunner:
    return PaperTradingRunner(
        discovery_client=_OneGroupDiscovery(),
        trade_logger=PaperTradeLogger(log_path=tmp_path / "fixes.jsonl"),
        heartbeat_interval_sec=0.05,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_runner_refuses_to_start_without_a_usable_engine(tmp_path: Path) -> None:
    engine = _StubEngine()
    engine.missing_components = ["engine.combinatorial_arb.evaluate_group"]  # type: ignore[attr-defined]
    runner = _runner(tmp_path, engine=engine)
    with pytest.raises(RuntimeError, match="Quant engine not usable"):
        # The timeout keeps a regression from hanging the suite: run() would never return.
        await asyncio.wait_for(runner.run(stop_event=asyncio.Event()), timeout=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", [True, False], ids=["stream-raises", "stream-returns-early"])
async def test_runner_aborts_when_a_background_task_dies(tmp_path: Path, crash: bool) -> None:
    class _DyingStream:
        token_ids = ["yes", "no"]
        on_book_update = None

        def get_all_books(self) -> dict[str, OrderBookSnapshot]:
            return {}

        async def run(self, stop_event: asyncio.Event | None = None) -> None:
            if crash:
                raise ConnectionError("stream died")

    runner = _runner(tmp_path, engine=_StubEngine(), book_stream=_DyingStream())
    with pytest.raises(RuntimeError, match="clob_book_stream stopped unexpectedly"):
        # Before the fix the runner kept running with heartbeats only: this would time out.
        await asyncio.wait_for(runner.run(stop_event=asyncio.Event()), timeout=2.0)


@pytest.mark.asyncio
async def test_runner_throttles_identical_risk_rejections(tmp_path: Path) -> None:
    engine = _StubEngine(approve=False, reason="EVENT_EXPOSURE_EXCEEDED: evento evt-fix oltre il cap")
    runner = PaperTradingRunner(
        engine=engine,
        market_groups=[_group("yes", "no")],
        books={"yes": _book("yes", asks=[(0.48, 50.0)]), "no": _book("no", asks=[(0.49, 50.0)])},
        trade_logger=PaperTradeLogger(log_path=tmp_path / "throttle.jsonl"),
    )
    for _ in range(6):
        await runner.handle_book_update(_book("yes", asks=[(0.48, 50.0)]))

    lines = (tmp_path / "throttle.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # one log line, not one per book update
    stats = runner.get_heartbeat_stats()
    assert stats["signals_rejected_risk"] == 6
    assert stats["rejections_not_logged"] == 5

    # A different reason for the same event is a new fact and is logged immediately.
    engine.reason = "STALE_BOOK: book stale per token no"
    await runner.handle_book_update(_book("yes", asks=[(0.48, 50.0)]))
    assert len((tmp_path / "throttle.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    # With the interval elapsed the next rejection is logged and reports what was skipped.
    runner.rejection_log_interval_sec = 0.0
    engine.reason = "EVENT_EXPOSURE_EXCEEDED: evento evt-fix oltre il cap"
    await runner.handle_book_update(_book("yes", asks=[(0.48, 50.0)]))
    last = json.loads((tmp_path / "throttle.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["payload"]["reason"].endswith("(+5 identical rejections not logged)")


def test_runner_heartbeat_reports_feed_health(tmp_path: Path) -> None:
    stream = ClobOrderBookStream(token_ids=["tok", "no-snapshot-yet"])
    stream.apply_ws_message(_book_event("tok"), now=T0)
    stream.apply_ws_message(_delta("tok", "0.6", "1", timestamp=T0 - timedelta(seconds=4)), now=T0)
    stream.apply_ws_message(_delta("no-snapshot-yet", "0.6", "1"), now=T0)  # ours, but no book yet
    stream.apply_ws_message(_delta("not-followed", "0.6", "1"), now=T0)  # not ours: not counted

    runner = _runner(tmp_path, engine=_StubEngine(), book_stream=stream)
    stats = runner.get_heartbeat_stats()
    assert stats["feed_lag_sec"] == pytest.approx(4.0)
    assert stats["ws_deltas_ignored"] == 1

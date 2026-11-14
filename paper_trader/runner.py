"""24/7 asyncio paper-trading runner wiring collector -> engine -> simulator -> JSONL logger."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import signal as signal_mod
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from collector.clob_ws_stream import ClobOrderBookStream
from collector.gamma_discovery import GammaDiscoveryClient
from collector.news_stream import GeminiNewsClassifier, NewsStreamCollector
from contracts.models import (
    MarketGroup,
    NewsAssessment,
    OrderBookSnapshot,
    PaperFill,
    TradeSignal,
)
from paper_trader.logger import PaperTradeLogger
from paper_trader.simulator import PaperOrderSimulator

logger = logging.getLogger(__name__)

PAPER_ONLY: bool = True


def validate_paper_only_safety() -> None:
    """Enforce mandatory paper-only execution and reject any wallet PRIVATE_KEY."""
    assert PAPER_ONLY is True, "PAPER_ONLY must always remain True"
    if PAPER_ONLY is not True:
        raise RuntimeError("PAPER_ONLY must always remain True")
    if os.getenv("PRIVATE_KEY", "").strip():
        raise RuntimeError(
            "Security violation: PRIVATE_KEY must not be set during paper-trading mode."
        )


@runtime_checkable
class EngineProtocol(Protocol):
    """Protocol decoupling /paper_trader from Recinto 2 (/engine)."""

    def evaluate_group(
        self, group: MarketGroup, books: dict[str, OrderBookSnapshot]
    ) -> TradeSignal | None:
        ...

    def evaluate_news(
        self, assessment: NewsAssessment, book: OrderBookSnapshot | None
    ) -> TradeSignal | None:
        ...

    def approve_signal(
        self, signal: TradeSignal, books: dict[str, OrderBookSnapshot]
    ) -> tuple[bool, str]:
        ...


class DynamicEngineAdapter:
    """Lazily resolves /engine modules when present, failing safe when absent."""

    def __init__(self) -> None:
        self._arb_eval: Callable[..., Any] | None = self._load_callable(
            "engine.combinatorial_arb",
            "evaluate_group",
            ("CombinatorialArbEngine", "CombinatorialArbitrageEngine", "CombinatorialArb"),
        )
        self._news_eval: Callable[..., Any] | None = self._load_callable(
            "engine.news_latency",
            "evaluate_news",
            ("NewsLatencyEvaluator", "NewsLatencyEngine", "NewsEvaluator"),
        )
        self._risk_approve: Callable[..., Any] | None = self._load_callable(
            "engine.risk_manager",
            "approve_signal",
            ("RiskManager", "DeterministicRiskManager"),
        )

    @staticmethod
    def _load_callable(
        module_name: str,
        method_name: str,
        preferred_classes: tuple[str, ...],
    ) -> Callable[..., Any] | None:
        try:
            mod = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            return None

        fn = getattr(mod, method_name, None)
        if callable(fn):
            return fn

        for cls_name in preferred_classes:
            cls = getattr(mod, cls_name, None)
            if isinstance(cls, type):
                try:
                    instance = cls()
                    bound = getattr(instance, method_name, None)
                    if callable(bound):
                        return bound
                except TypeError:
                    continue

        for attr in vars(mod).values():
            if isinstance(attr, type) and attr.__module__ == mod.__name__:
                bound_check = getattr(attr, method_name, None)
                if callable(bound_check):
                    try:
                        instance = attr()
                        bound = getattr(instance, method_name, None)
                        if callable(bound):
                            return bound
                    except TypeError:
                        continue

        return None

    def evaluate_group(
        self, group: MarketGroup, books: dict[str, OrderBookSnapshot]
    ) -> TradeSignal | None:
        """Delegate group arbitrage evaluation to engine.combinatorial_arb if loaded."""
        if self._arb_eval is None:
            return None
        return self._arb_eval(group, books)

    def evaluate_news(
        self, assessment: NewsAssessment, book: OrderBookSnapshot | None
    ) -> TradeSignal | None:
        """Delegate news-latency evaluation to engine.news_latency if loaded."""
        if self._news_eval is None:
            return None
        return self._news_eval(assessment, book)

    def approve_signal(
        self, signal: TradeSignal, books: dict[str, OrderBookSnapshot]
    ) -> tuple[bool, str]:
        """Delegate deterministic risk checks to engine.risk_manager if loaded."""
        if self._risk_approve is None:
            return False, "engine_not_loaded"
        result = self._risk_approve(signal, books)
        if isinstance(result, tuple) and len(result) >= 2:
            return bool(result[0]), str(result[1])
        if isinstance(result, bool):
            return result, ("approved" if result else "rejected")
        if hasattr(result, "approved"):
            reason = getattr(result, "reason", "approved" if result.approved else "rejected")
            return bool(result.approved), str(reason)
        return False, "invalid_risk_response"


class PaperTradingRunner:
    """Orchestrates Polymarket discovery, L2 WS stream, news stream, engine, and simulator."""

    def __init__(
        self,
        discovery_client: GammaDiscoveryClient | Any | None = None,
        book_stream: ClobOrderBookStream | Any | None = None,
        news_collector: NewsStreamCollector | Any | None = None,
        engine: EngineProtocol | None = None,
        simulator: PaperOrderSimulator | None = None,
        trade_logger: PaperTradeLogger | None = None,
        market_groups: list[MarketGroup] | None = None,
        books: dict[str, OrderBookSnapshot] | None = None,
        heartbeat_interval_sec: float = 60.0,
        min_liquidity_usdc: float = 0.0,
        news_feed_urls: list[str] | None = None,
        log_path: str | Path = "logs/paper_trades.jsonl",
    ) -> None:
        validate_paper_only_safety()

        self._book_stream_injected = book_stream is not None
        self._news_collector_injected = news_collector is not None

        self.discovery_client = (
            discovery_client
            if discovery_client is not None
            else GammaDiscoveryClient(min_liquidity_usdc=min_liquidity_usdc)
        )
        self.book_stream = book_stream
        self.news_collector = news_collector
        self._news_feed_urls = list(news_feed_urls) if news_feed_urls else []
        self.engine: EngineProtocol = (
            engine if engine is not None else DynamicEngineAdapter()
        )
        self.simulator = simulator if simulator is not None else PaperOrderSimulator()
        self.trade_logger = (
            trade_logger if trade_logger is not None else PaperTradeLogger(log_path=log_path)
        )
        self.heartbeat_interval_sec = heartbeat_interval_sec

        self._books: dict[str, OrderBookSnapshot] = dict(books) if books else {}
        self._market_groups: list[MarketGroup] = []
        self._groups_by_token: dict[str, list[MarketGroup]] = {}
        self._eval_lock = asyncio.Lock()
        self._stats: dict[str, int] = {
            "book_updates": 0,
            "news_assessments": 0,
            "signals_generated": 0,
            "signals_approved": 0,
            "signals_rejected_risk": 0,
            "fills_executed": 0,
            "fills_rejected": 0,
        }

        if market_groups:
            self.set_market_groups(market_groups)

    def set_market_groups(self, groups: list[MarketGroup]) -> list[str]:
        """Store discovered market groups and index them by token_id."""
        self._market_groups = list(groups)
        groups_by_token: dict[str, list[MarketGroup]] = {}
        seen_tokens: set[str] = set()
        ordered_tokens: list[str] = []

        for group in self._market_groups:
            for outcome in group.outcomes:
                token_id = outcome.token_id
                if token_id not in seen_tokens:
                    seen_tokens.add(token_id)
                    ordered_tokens.append(token_id)
                bucket = groups_by_token.setdefault(token_id, [])
                if group not in bucket:
                    bucket.append(group)

        self._groups_by_token = groups_by_token
        return ordered_tokens

    def get_heartbeat_stats(self) -> dict[str, Any]:
        """Return a snapshot of runtime paper-trading counters."""
        return {
            **self._stats,
            "paper_only": PAPER_ONLY,
            "tracked_groups": len(self._market_groups),
            "tracked_books": len(self._books),
        }

    def _sync_books_from_stream(self) -> None:
        if self.book_stream is not None and hasattr(self.book_stream, "get_all_books"):
            stream_books = self.book_stream.get_all_books()
            if isinstance(stream_books, dict):
                self._books.update(stream_books)

    async def handle_book_update(self, snapshot: OrderBookSnapshot) -> list[PaperFill]:
        """Evaluate affected market groups upon an L2 order-book update."""
        async with self._eval_lock:
            self._sync_books_from_stream()
            self._books[snapshot.token_id] = snapshot
            self._stats["book_updates"] += 1

            if not self._groups_by_token and self._market_groups:
                self.set_market_groups(self._market_groups)

            affected_groups = self._groups_by_token.get(snapshot.token_id, [])
            if not affected_groups:
                return []

            books_view = dict(self._books)
            fills: list[PaperFill] = []

            for group in affected_groups:
                signal = self.engine.evaluate_group(group, books_view)
                if signal is None:
                    continue

                self._stats["signals_generated"] += 1
                approved, reason = self.engine.approve_signal(signal, books_view)
                if not approved:
                    self._stats["signals_rejected_risk"] += 1
                    self.trade_logger.log_signal_rejected(signal, reason)
                    continue

                self._stats["signals_approved"] += 1
                fill = self.simulator.simulate_signal(signal, books_view)
                self.trade_logger.log_paper_fill(fill, signal=signal)
                if fill.status == "FILLED":
                    self._stats["fills_executed"] += 1
                else:
                    self._stats["fills_rejected"] += 1
                fills.append(fill)

            return fills

    async def handle_news_assessment(self, assessment: NewsAssessment) -> PaperFill | None:
        """Evaluate a classified news event against the current target order book."""
        async with self._eval_lock:
            self._sync_books_from_stream()
            self._stats["news_assessments"] += 1

            book = self._books.get(assessment.target_token_id)
            if (
                book is None
                and self.book_stream is not None
                and hasattr(self.book_stream, "get_book")
            ):
                maybe_book = self.book_stream.get_book(assessment.target_token_id)
                if isinstance(maybe_book, OrderBookSnapshot):
                    book = maybe_book
                    self._books[assessment.target_token_id] = book

            books_view = dict(self._books)
            signal = self.engine.evaluate_news(assessment, book)
            if signal is None:
                return None

            self._stats["signals_generated"] += 1
            approved, reason = self.engine.approve_signal(signal, books_view)
            if not approved:
                self._stats["signals_rejected_risk"] += 1
                self.trade_logger.log_signal_rejected(signal, reason)
                return None

            self._stats["signals_approved"] += 1
            fill = self.simulator.simulate_signal(signal, books_view)
            self.trade_logger.log_paper_fill(
                fill,
                signal=signal,
                news_published_at=assessment.published_at,
            )
            if fill.status == "FILLED":
                self._stats["fills_executed"] += 1
            else:
                self._stats["fills_rejected"] += 1
            return fill

    async def _heartbeat_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=max(0.05, self.heartbeat_interval_sec)
                )
                break
            except asyncio.TimeoutError:
                self.trade_logger.log_heartbeat(self.get_heartbeat_stats())

    async def run(
        self,
        stop_event: asyncio.Event | None = None,
        discovery_limit: int = 50,
    ) -> None:
        """Discover active markets, stream books and news, and log paper executions."""
        validate_paper_only_safety()
        stop = stop_event if stop_event is not None else asyncio.Event()

        groups = await self.discovery_client.discover_market_groups(limit=discovery_limit)
        token_ids = self.set_market_groups(groups)

        if self.book_stream is None:
            self.book_stream = ClobOrderBookStream(
                token_ids=token_ids,
                on_book_update=self.handle_book_update,
            )
        elif getattr(self.book_stream, "on_book_update", None) is None:
            self.book_stream.on_book_update = self.handle_book_update

        if self.news_collector is None:
            self.news_collector = NewsStreamCollector(
                feed_urls=self._news_feed_urls,
                classifier=GeminiNewsClassifier(),
                on_assessment=self.handle_news_assessment,
            )
        elif getattr(self.news_collector, "on_assessment", None) is None:
            self.news_collector.on_assessment = self.handle_news_assessment

        tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(self._heartbeat_loop(stop), name="paper_heartbeat")
        ]

        stream_tokens = getattr(self.book_stream, "token_ids", token_ids)
        if stream_tokens or self._book_stream_injected:
            tasks.append(
                asyncio.create_task(
                    self.book_stream.run(stop_event=stop), name="clob_book_stream"
                )
            )
        else:
            logger.warning("No CLOB token_ids discovered; book stream idle until stop.")

        feed_urls = getattr(self.news_collector, "feed_urls", [])
        if feed_urls or self._news_collector_injected:
            tasks.append(
                asyncio.create_task(
                    self.news_collector.run(
                        lambda: list(self._market_groups),
                        stop_event=stop,
                    ),
                    name="news_stream_collector",
                )
            )

        def _on_task_done(task: asyncio.Task[Any]) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Background task %s failed unexpectedly: %s",
                    task.get_name(),
                    exc,
                    exc_info=exc,
                )

        for task in tasks:
            task.add_done_callback(_on_task_done)

        try:
            await stop.wait()
        finally:
            stop.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.trade_logger.log_heartbeat(self.get_heartbeat_stats())


def _parse_feed_urls_from_env() -> list[str]:
    raw = os.getenv("NEWS_FEED_URLS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


async def _run_cli() -> None:
    validate_paper_only_safety()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal_mod.SIGINT, signal_mod.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    log_path = os.getenv("LOG_FILE_PATH", "logs/paper_trades.jsonl")
    min_liq = float(os.getenv("MIN_LIQUIDITY_USDC", "0.0") or "0.0")
    discovery_limit = int(os.getenv("DISCOVERY_LIMIT", "50") or "50")
    heartbeat_sec = float(os.getenv("HEARTBEAT_INTERVAL_SEC", "60.0") or "60.0")

    runner = PaperTradingRunner(
        min_liquidity_usdc=min_liq,
        news_feed_urls=_parse_feed_urls_from_env(),
        heartbeat_interval_sec=heartbeat_sec,
        log_path=log_path,
    )
    await runner.run(stop_event=stop_event, discovery_limit=discovery_limit)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    validate_paper_only_safety()
    try:
        asyncio.run(_run_cli())
    except KeyboardInterrupt:
        logger.info("PaperTradingRunner stopped via KeyboardInterrupt.")

"""Asynchronous Polymarket CLOB REST + WebSocket L2 order book stream.

Read-only public order book ingestion (Paper Trading phase — no wallet keys or orders).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets

from contracts.models import BookLevel, OrderBookSnapshot

logger = logging.getLogger(__name__)

DEFAULT_CLOB_REST_URL = "https://clob.polymarket.com"
DEFAULT_CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any) -> float | None:
    """Convert value to a finite float, returning None if invalid."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_book_levels(raw_levels: list[dict[str, Any]], descending: bool) -> list[BookLevel]:
    """Validate and sort raw CLOB price levels into BookLevel models."""
    if not raw_levels:
        return []

    levels: list[BookLevel] = []
    for item in raw_levels:
        if not isinstance(item, dict):
            continue
        price = _safe_float(item.get("price"))
        size = _safe_float(item.get("size"))
        if price is None or size is None:
            continue
        if not (0.0 <= price <= 1.0) or size <= 0.0:
            continue
        levels.append(BookLevel(price=price, size=size))

    levels.sort(key=lambda level: level.price, reverse=descending)
    return levels


class ClobOrderBookStream:
    """Maintains L2 order book snapshots from Polymarket CLOB REST and WebSocket."""

    def __init__(
        self,
        token_ids: list[str],
        rest_url: str = DEFAULT_CLOB_REST_URL,
        ws_url: str = DEFAULT_CLOB_WS_URL,
        ping_interval_sec: float = 10.0,
        stale_threshold_sec: float = 30.0,
        max_backoff_sec: float = 60.0,
        on_book_update: Callable[[OrderBookSnapshot], Awaitable[None] | None] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.token_ids = list(dict.fromkeys(str(tid).strip() for tid in token_ids if str(tid).strip()))
        self.rest_url = rest_url.rstrip("/")
        self.ws_url = ws_url
        self.ping_interval_sec = float(ping_interval_sec)
        self.stale_threshold_sec = float(stale_threshold_sec)
        self.max_backoff_sec = float(max_backoff_sec)
        self.on_book_update = on_book_update
        self._http_client = http_client
        self._books: dict[str, OrderBookSnapshot] = {}

    async def fetch_initial_snapshot(self, token_id: str) -> OrderBookSnapshot:
        """Fetch an initial L2 snapshot via GET /book?token_id=<token_id>."""
        if not token_id:
            raise ValueError("token_id must be non-empty")

        url = f"{self.rest_url}/book"
        params = {"token_id": token_id}

        if self._http_client is not None:
            response = await self._http_client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()

        bids = parse_book_levels(payload.get("bids") or [], descending=True) if isinstance(payload, dict) else []
        asks = parse_book_levels(payload.get("asks") or [], descending=False) if isinstance(payload, dict) else []
        snapshot = OrderBookSnapshot(
            token_id=token_id,
            bids=bids,
            asks=asks,
            updated_at=datetime.now(timezone.utc),
            is_stale=False,
        )
        self._books[token_id] = snapshot
        return snapshot

    async def initialize_books(self) -> dict[str, OrderBookSnapshot]:
        """Fetch initial REST snapshots for all configured token_ids."""
        if not self.token_ids:
            return {}

        if self._http_client is not None:
            await asyncio.gather(*(self.fetch_initial_snapshot(tid) for tid in self.token_ids))
        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                self._http_client = client
                try:
                    await asyncio.gather(*(self.fetch_initial_snapshot(tid) for tid in self.token_ids))
                finally:
                    self._http_client = None

        return self.get_all_books()

    def apply_ws_message(
        self,
        message: str | dict[str, Any] | list[Any],
        now: datetime | None = None,
    ) -> list[OrderBookSnapshot]:
        """Apply a CLOB WebSocket message (book snapshot or price_change delta) to local books."""
        ts = _ensure_utc(now) if now is not None else datetime.now(timezone.utc)

        parsed: Any = message
        if isinstance(parsed, bytes):
            parsed = parsed.decode("utf-8", errors="replace")
        if isinstance(parsed, str):
            stripped = parsed.strip()
            if not stripped or stripped.upper() in {"PONG", "PING"}:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON WS message: %r", stripped[:120])
                return []

        if isinstance(parsed, dict):
            events = [parsed]
        elif isinstance(parsed, list):
            events = [item for item in parsed if isinstance(item, dict)]
        else:
            return []

        updated: dict[str, OrderBookSnapshot] = {}

        for event in events:
            event_type = str(event.get("event_type") or event.get("type") or "").strip().lower()
            if event_type == "book":
                token_id = str(event.get("asset_id") or event.get("token_id") or "").strip()
                if not token_id:
                    continue
                bids = parse_book_levels(event.get("bids") or [], descending=True)
                asks = parse_book_levels(event.get("asks") or [], descending=False)
                snapshot = OrderBookSnapshot(
                    token_id=token_id,
                    bids=bids,
                    asks=asks,
                    updated_at=ts,
                    is_stale=False,
                )
                self._books[token_id] = snapshot
                updated[token_id] = snapshot
            elif event_type == "price_change":
                for token_id, snapshot in self._apply_price_change_event(event, ts).items():
                    updated[token_id] = snapshot

        return list(updated.values())

    def _apply_price_change_event(
        self,
        event: dict[str, Any],
        ts: datetime,
    ) -> dict[str, OrderBookSnapshot]:
        """Apply both Polymarket price_change schemas to affected token books."""
        top_token = str(event.get("asset_id") or event.get("token_id") or "").strip()
        raw_changes: list[dict[str, Any]] = []
        if isinstance(event.get("price_changes"), list):
            raw_changes.extend(item for item in event["price_changes"] if isinstance(item, dict))
        if isinstance(event.get("changes"), list):
            raw_changes.extend(item for item in event["changes"] if isinstance(item, dict))

        if not raw_changes:
            return {}

        changes_by_token: dict[str, list[tuple[str, float, float]]] = {}
        for change in raw_changes:
            token_id = str(change.get("asset_id") or change.get("token_id") or top_token).strip()
            if not token_id:
                continue
            side = str(change.get("side") or "").strip().upper()
            if side not in {"BUY", "BID", "SELL", "ASK"}:
                continue
            price = _safe_float(change.get("price"))
            size = _safe_float(change.get("size"))
            if price is None or size is None or not (0.0 <= price <= 1.0):
                continue
            changes_by_token.setdefault(token_id, []).append((side, price, size))

        updated: dict[str, OrderBookSnapshot] = {}
        for token_id, deltas in changes_by_token.items():
            existing = self._books.get(token_id)
            bid_map = {lvl.price: lvl.size for lvl in existing.bids} if existing is not None else {}
            ask_map = {lvl.price: lvl.size for lvl in existing.asks} if existing is not None else {}

            for side, price, size in deltas:
                target_map = bid_map if side in {"BUY", "BID"} else ask_map
                if size <= 0.0:
                    target_map.pop(price, None)
                else:
                    target_map[price] = size

            bids = sorted(
                (BookLevel(price=p, size=s) for p, s in bid_map.items()),
                key=lambda level: level.price,
                reverse=True,
            )
            asks = sorted(
                (BookLevel(price=p, size=s) for p, s in ask_map.items()),
                key=lambda level: level.price,
            )
            snapshot = OrderBookSnapshot(
                token_id=token_id,
                bids=bids,
                asks=asks,
                updated_at=ts,
                is_stale=False,
            )
            self._books[token_id] = snapshot
            updated[token_id] = snapshot

        return updated

    def get_book(self, token_id: str, now: datetime | None = None) -> OrderBookSnapshot | None:
        """Return a snapshot copy for token_id, marking is_stale=True if past threshold."""
        snapshot = self._books.get(token_id)
        if snapshot is None:
            return None

        ref_now = _ensure_utc(now) if now is not None else datetime.now(timezone.utc)
        elapsed_sec = (ref_now - _ensure_utc(snapshot.updated_at)).total_seconds()
        if snapshot.is_stale or elapsed_sec > self.stale_threshold_sec:
            return snapshot.model_copy(update={"is_stale": True})
        return snapshot.model_copy()

    def get_all_books(self, now: datetime | None = None) -> dict[str, OrderBookSnapshot]:
        """Return all stored book snapshots evaluated for staleness at `now`."""
        ref_now = _ensure_utc(now) if now is not None else datetime.now(timezone.utc)
        result: dict[str, OrderBookSnapshot] = {}
        for token_id in list(self._books):
            snapshot = self.get_book(token_id, now=ref_now)
            if snapshot is not None:
                result[token_id] = snapshot
        return result

    def mark_all_stale(self) -> None:
        """Mark all stored snapshots as stale."""
        for token_id, snapshot in list(self._books.items()):
            self._books[token_id] = snapshot.model_copy(update={"is_stale": True})

    async def _ws_send(self, ws: Any, payload: str) -> None:
        """Send a payload to a WebSocket object supporting sync or async send."""
        res = ws.send(payload)
        if inspect.isawaitable(res):
            await res

    async def _ping_loop(self, ws: Any, stop_event: asyncio.Event | None = None) -> None:
        """Send Polymarket text PING heartbeat every ping_interval_sec."""
        if self.ping_interval_sec <= 0.0:
            return

        while stop_event is None or not stop_event.is_set():
            if stop_event is not None:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.ping_interval_sec)
                    break
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(self.ping_interval_sec)

            if stop_event is not None and stop_event.is_set():
                break
            await self._ws_send(ws, "PING")

    @contextlib.asynccontextmanager
    async def _open_ws(self, connector: Callable[..., Any]) -> AsyncIterator[Any]:
        """Open a WebSocket supporting both async context managers and direct returns."""
        conn = connector(self.ws_url)
        if hasattr(conn, "__aenter__"):
            async with conn as ws:
                yield ws
            return

        ws = await conn if inspect.isawaitable(conn) else conn
        try:
            yield ws
        finally:
            close_fn = getattr(ws, "close", None)
            if callable(close_fn):
                res = close_fn()
                if inspect.isawaitable(res):
                    await res

    async def _iter_messages(
        self,
        ws: Any,
        stop_event: asyncio.Event | None,
        ping_task: asyncio.Task[None],
    ) -> AsyncIterator[Any]:
        """Yield messages from ws while racing against ping failure and stop_event."""
        use_recv = hasattr(ws, "recv") and callable(ws.recv)
        aiter_ws = None if use_recv else aiter(ws)

        async def _read_next() -> Any:
            if use_recv:
                res = ws.recv()
                return await res if inspect.isawaitable(res) else res
            return await anext(aiter_ws)  # type: ignore[arg-type]

        while stop_event is None or not stop_event.is_set():
            next_task: asyncio.Task[Any] = asyncio.create_task(_read_next())
            tasks: set[asyncio.Task[Any]] = {next_task, ping_task}
            stop_task: asyncio.Task[bool] | None = None
            if stop_event is not None:
                stop_task = asyncio.create_task(stop_event.wait())
                tasks.add(stop_task)

            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                if stop_task is not None and not stop_task.done():
                    stop_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stop_task
                if not next_task.done():
                    next_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await next_task

            if stop_event is not None and stop_event.is_set():
                return

            if ping_task in done:
                exc = ping_task.exception()
                if exc is not None:
                    raise exc
                return

            if next_task in done:
                try:
                    raw_msg = next_task.result()
                except StopAsyncIteration:
                    return
                yield raw_msg

    async def _emit_update(self, snapshot: OrderBookSnapshot) -> None:
        """Invoke on_book_update callback safely (sync or async)."""
        if self.on_book_update is None:
            return
        try:
            result = self.on_book_update(snapshot)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("on_book_update callback raised an error for token_id=%s", snapshot.token_id)

    async def _sleep_backoff(self, delay: float, stop_event: asyncio.Event | None) -> None:
        """Sleep backoff delay via asyncio.sleep while remaining interruptible by stop_event."""
        if delay <= 0.0:
            return
        if stop_event is None:
            await asyncio.sleep(delay)
            return

        sleep_task = asyncio.create_task(asyncio.sleep(delay))
        stop_task = asyncio.create_task(stop_event.wait())
        try:
            await asyncio.wait({sleep_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleep_task, stop_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def run(
        self,
        stop_event: asyncio.Event | None = None,
        ws_connect: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize snapshots and maintain the CLOB WebSocket subscription loop."""
        if stop_event is not None and stop_event.is_set():
            return

        if self.token_ids:
            await self.initialize_books()

        connector = ws_connect or websockets.connect
        backoff = 1.0

        while stop_event is None or not stop_event.is_set():
            try:
                async with self._open_ws(connector) as ws:
                    sub_payload = json.dumps({"assets_ids": self.token_ids, "type": "market"})
                    await self._ws_send(ws, sub_payload)

                    ping_task = asyncio.create_task(self._ping_loop(ws, stop_event=stop_event))
                    try:
                        async for raw_msg in self._iter_messages(ws, stop_event=stop_event, ping_task=ping_task):
                            backoff = 1.0
                            updated_snapshots = self.apply_ws_message(raw_msg)
                            for snapshot in updated_snapshots:
                                await self._emit_update(snapshot)
                            if stop_event is not None and stop_event.is_set():
                                break
                    finally:
                        ping_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await ping_task

                if stop_event is not None and stop_event.is_set():
                    break
                self.mark_all_stale()
            except asyncio.CancelledError:
                raise
            except Exception:
                if stop_event is not None and stop_event.is_set():
                    break
                logger.warning("CLOB WebSocket connection lost; marking books stale", exc_info=True)
                self.mark_all_stale()

            if stop_event is not None and stop_event.is_set():
                break

            sleep_for = min(backoff, self.max_backoff_sec)
            await self._sleep_backoff(sleep_for, stop_event=stop_event)
            backoff = min(backoff * 2.0, self.max_backoff_sec)

"""Record what the bot receives from the network, and replay it later offline.

Recording wraps the two transports the bot reads from, so the file holds exactly what the
bot saw, in the order it saw it:

    http          every GET response (Gamma discovery, CLOB /book snapshots)
    ws_open       a WebSocket connection was opened
    ws_send       what the bot sent on it (subscription, PING)
    ws_frame      every frame received
    ws_close      the connection ended, with the error if any
    assessment    a news assessment produced by the classifier

Replay serves the same responses and frames back through the same code paths, with no
network access. Exchange timestamps are shifted to "now" while keeping the lag each
message had when it was recorded, so freshness checks behave as they did live.

Read-only by construction: nothing here signs, sends or prices a real order.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from contracts.models import NewsAssessment

logger = logging.getLogger(__name__)

SESSION_FORMAT_VERSION = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- recording


class SessionRecorder:
    """Append-only JSONL writer for one recorded session."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.counts: dict[str, int] = defaultdict(int)
        self.write({"kind": "meta", "version": SESSION_FORMAT_VERSION})

    def write(self, record: dict[str, Any]) -> None:
        entry = {"t": _now().isoformat(), **record}
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.counts[str(record.get("kind"))] += 1


class RecordingTransport(httpx.AsyncBaseTransport):
    """httpx transport that records every response before handing it to the caller."""

    def __init__(self, inner: httpx.AsyncBaseTransport, recorder: SessionRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        body = await response.aread()
        await response.aclose()
        self._recorder.write(
            {
                "kind": "http",
                "method": request.method,
                "url": str(request.url),
                "status": response.status_code,
                "body": body.decode("utf-8", errors="replace"),
            }
        )
        # The body is already decoded: drop the headers that describe the wire encoding.
        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
        ]
        return httpx.Response(status_code=response.status_code, headers=headers, content=body, request=request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _RecordingSocket:
    def __init__(self, ws: Any, recorder: SessionRecorder, conn: int) -> None:
        self._ws = ws
        self._recorder = recorder
        self._conn = conn

    async def send(self, data: Any) -> None:
        self._recorder.write({"kind": "ws_send", "conn": self._conn, "data": data if isinstance(data, str) else repr(data)})
        result = self._ws.send(data)
        if inspect.isawaitable(result):
            await result

    async def recv(self) -> Any:
        try:
            result = self._ws.recv()
            data = await result if inspect.isawaitable(result) else result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._recorder.write({"kind": "ws_close", "conn": self._conn, "error": repr(exc)})
            raise
        text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else data
        self._recorder.write({"kind": "ws_frame", "conn": self._conn, "data": text})
        return data

    async def close(self) -> None:
        close_fn = getattr(self._ws, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result


def recording_connector(inner: Callable[..., Any], recorder: SessionRecorder) -> Callable[..., Any]:
    """Wrap a WebSocket connector (e.g. `websockets.connect`) so every frame is recorded."""
    counter = iter(range(1_000_000))

    @asynccontextmanager
    async def connect(url: str) -> AsyncIterator[_RecordingSocket]:
        conn = next(counter)
        recorder.write({"kind": "ws_open", "conn": conn, "url": url})
        handle = inner(url)
        if hasattr(handle, "__aenter__"):
            async with handle as ws:
                yield _RecordingSocket(ws, recorder, conn)
            return
        ws = await handle if inspect.isawaitable(handle) else handle
        try:
            yield _RecordingSocket(ws, recorder, conn)
        finally:
            close_fn = getattr(ws, "close", None)
            if callable(close_fn):
                result = close_fn()
                if inspect.isawaitable(result):
                    await result

    return connect


def record_assessments(
    handler: Callable[[NewsAssessment], Awaitable[Any]], recorder: SessionRecorder
) -> Callable[[NewsAssessment], Awaitable[Any]]:
    """Wrap the runner's news handler so each assessment is recorded before it is handled."""

    async def recording_handler(assessment: NewsAssessment) -> Any:
        recorder.write({"kind": "assessment", "payload": assessment.model_dump(mode="json")})
        return await handler(assessment)

    return recording_handler


# --------------------------------------------------------------------------- replay


def _request_key(method: str, url: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    parsed = httpx.URL(url)
    return method.upper(), parsed.path, tuple(sorted(parsed.params.multi_items()))


class SessionReplay:
    """Serve a recorded session back to the bot through fake transports."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._http: dict[tuple[str, str, tuple[tuple[str, str], ...]], deque[dict[str, Any]]] = defaultdict(deque)
        self._timeline: list[dict[str, Any]] = []
        self._cursor = 0
        self._connections_opened = 0
        self.done = asyncio.Event()
        self.unmatched_requests: list[str] = []
        self.frames_replayed = 0
        self.assessments_replayed = 0
        self.news_handler: Callable[[NewsAssessment], Awaitable[Any]] | None = None
        self.records_read = 0
        # Settings the recording was made with (discovery limit, liquidity filter).
        self.config: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                self.records_read += 1
                kind = record.get("kind")
                if kind == "http":
                    self._http[_request_key(str(record.get("method") or "GET"), str(record.get("url") or ""))].append(record)
                elif kind in {"ws_open", "ws_frame", "ws_close", "assessment"}:
                    self._timeline.append(record)
                elif kind == "config":
                    self.config.update({k: v for k, v in record.items() if k not in {"t", "kind"}})

    # ----- HTTP

    def _handle_request(self, request: httpx.Request) -> httpx.Response:
        queue = self._http.get(_request_key(request.method, str(request.url)))
        if not queue:
            self.unmatched_requests.append(str(request.url))
            return httpx.Response(404, json={"error": "request not present in the recorded session"})
        # Serve responses in recorded order; keep repeating the last one once exhausted.
        record = queue.popleft() if len(queue) > 1 else queue[0]
        return httpx.Response(int(record.get("status") or 200), text=str(record.get("body") or ""))

    def http_transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle_request)

    # ----- WebSocket

    def _shift_frame(self, record: dict[str, Any]) -> str:
        """Move exchange timestamps to now, keeping the lag the frame had when recorded."""
        data = record.get("data")
        if not isinstance(data, str):
            return ""
        received_at = _parse_time(record.get("t"))
        if received_at is None or not data.lstrip().startswith(("{", "[")):
            return data
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return data
        received_ms = received_at.timestamp() * 1000.0
        now_ms = _now().timestamp() * 1000.0

        def shift(event: Any) -> None:
            if not isinstance(event, dict):
                return
            try:
                exchange_ms = float(event.get("timestamp"))
            except (TypeError, ValueError):
                return
            event["timestamp"] = str(int(now_ms - (received_ms - exchange_ms)))

        for event in parsed if isinstance(parsed, list) else [parsed]:
            shift(event)
        return json.dumps(parsed)

    async def _dispatch_assessment(self, record: dict[str, Any]) -> None:
        if self.news_handler is None:
            return
        try:
            assessment = NewsAssessment.model_validate(record.get("payload"))
        except Exception:
            logger.warning("Skipping malformed recorded assessment")
            return
        recorded_at = _parse_time(record.get("t"))
        if recorded_at is not None:
            # Keep the news as old as it was when the bot first saw it.
            age = recorded_at - assessment.published_at.astimezone(timezone.utc)
            assessment = assessment.model_copy(update={"published_at": _now() - max(age, timedelta(0))})
        self.assessments_replayed += 1
        await self.news_handler(assessment)

    async def _next_frame(self, conn: int) -> str:
        """Next frame for connection `conn`; raises ConnectionError when that connection ends."""
        while self._cursor < len(self._timeline):
            record = self._timeline[self._cursor]
            kind, record_conn = record.get("kind"), record.get("conn")
            if kind == "assessment":
                self._cursor += 1
                await self._dispatch_assessment(record)
                continue
            if record_conn != conn:
                if isinstance(record_conn, int) and record_conn > conn:
                    raise ConnectionError("recorded connection ended")
                self._cursor += 1  # leftovers of an earlier connection
                continue
            self._cursor += 1
            if kind == "ws_frame":
                self.frames_replayed += 1
                return self._shift_frame(record)
            if kind == "ws_close":
                raise ConnectionError(f"recorded disconnect: {record.get('error')}")
        self.done.set()
        await asyncio.Event().wait()  # idle until the socket is closed
        raise ConnectionError("unreachable")

    def ws_connector(self) -> Callable[..., Any]:
        replay = self

        class _ReplaySocket:
            def __init__(self, conn: int) -> None:
                self._conn = conn
                self.sent: list[Any] = []

            async def send(self, data: Any) -> None:
                self.sent.append(data)

            async def recv(self) -> str:
                return await replay._next_frame(self._conn)

            async def close(self) -> None:
                return None

        @asynccontextmanager
        async def connect(url: str) -> AsyncIterator[_ReplaySocket]:
            conn = replay._connections_opened
            replay._connections_opened += 1
            yield _ReplaySocket(conn)

        return connect

    @property
    def connections_recorded(self) -> int:
        return sum(1 for record in self._timeline if record.get("kind") == "ws_open")

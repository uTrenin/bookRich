"""The initial book dump must fit in one WebSocket message.

Found on the recording of 2026-11-19 (run 44521849932): right after subscribing, the market
channel sends the book of every followed token in one message. It weighed about 1.1 MB,
over the 1 MiB default of `websockets`, so every connection was closed with code 1009 and
not a single market message reached the bot in 15 minutes.

The end-to-end test uses a real `websockets` server on the loopback interface, because the
limit lives inside the library: a test double would not enforce it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import websockets

import paper_trader.record as record_module
from collector import clob_ws_stream
from collector.clob_ws_stream import (
    DEFAULT_WS_MAX_MESSAGE_BYTES,
    ClobOrderBookStream,
    default_ws_connect,
)

TOKENS = 600
LEVELS = 30  # per side: about 1.4 MB in total, more than the 1.1 MB seen live


def _book_dump(token_ids: list[str]) -> str:
    """One message with the full book of every token, as the market channel sends on subscribe."""
    events = []
    for token_id in token_ids:
        events.append(
            {
                "event_type": "book",
                "asset_id": token_id,
                "market": "0x" + "ab" * 32,
                "timestamp": "1763550000000",
                "hash": "0x" + "cd" * 20,
                "bids": [{"price": f"{0.01 * (i + 1):.2f}", "size": f"{100 + i}.25"} for i in range(LEVELS)],
                "asks": [{"price": f"{0.99 - 0.01 * i:.2f}", "size": f"{100 + i}.75"} for i in range(LEVELS)],
            }
        )
    return json.dumps(events)


def _token_ids() -> list[str]:
    # Real token ids are 76-78 digit integers.
    return [str(10**76 + i) for i in range(TOKENS)]


def test_the_limit_is_bounded_and_above_what_was_seen_live() -> None:
    assert 1_200_000 < DEFAULT_WS_MAX_MESSAGE_BYTES <= 64 * 1024 * 1024
    assert len(_book_dump(_token_ids()).encode()) > 1024 * 1024  # the test message breaks the old limit


def test_default_connector_passes_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_connect(url: str, **kwargs: Any) -> str:
        calls.append((url, kwargs))
        return "handle"

    monkeypatch.setattr(clob_ws_stream.websockets, "connect", fake_connect)

    assert default_ws_connect("wss://example.invalid/ws") == "handle"
    assert calls == [("wss://example.invalid/ws", {"max_size": DEFAULT_WS_MAX_MESSAGE_BYTES})]


def test_the_recorder_uses_the_same_connector() -> None:
    # The live recordings go through paper_trader.record: it must not fall back to the
    # library default either.
    assert record_module.default_ws_connect is default_ws_connect


async def _serve_one_dump(dump: str, received_subscriptions: list[str]) -> tuple[Any, int]:
    async def handler(ws: Any) -> None:
        received_subscriptions.append(await ws.recv())
        await ws.send(dump)
        await ws.wait_closed()

    # max_size on the server only bounds what it receives; what it sends is not limited.
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = next(iter(server.sockets)).getsockname()[1]
    return server, port


@pytest.mark.asyncio
async def test_stream_receives_a_book_dump_larger_than_one_mebibyte() -> None:
    token_ids = _token_ids()
    dump = _book_dump(token_ids)
    subscriptions: list[str] = []
    server, port = await _serve_one_dump(dump, subscriptions)

    got_all_books = asyncio.Event()
    updates: list[str] = []

    async def on_update(snapshot: Any) -> None:
        updates.append(snapshot.token_id)
        if len(updates) >= TOKENS:
            got_all_books.set()

    stream = ClobOrderBookStream(token_ids=token_ids, ws_url=f"ws://127.0.0.1:{port}", on_book_update=on_update)
    # No REST in this test: the books come from the WebSocket dump only.
    stream.initialize_books = _no_rest  # type: ignore[method-assign]
    stream._resync_after_grace = _no_resync  # type: ignore[method-assign]

    stop = asyncio.Event()
    run_task = asyncio.create_task(stream.run(stop_event=stop))  # default connector
    try:
        await asyncio.wait_for(got_all_books.wait(), timeout=20.0)
    finally:
        stop.set()
        await asyncio.wait_for(run_task, timeout=10.0)
        server.close()
        await server.wait_closed()

    assert json.loads(subscriptions[0])["assets_ids"] == token_ids
    assert sorted(set(updates)) == sorted(token_ids)
    book = stream.get_book(token_ids[-1])
    assert book is not None and not book.is_stale
    assert len(book.asks) == LEVELS and book.asks[0].price == pytest.approx(0.99 - 0.01 * (LEVELS - 1))


@pytest.mark.asyncio
async def test_the_library_default_would_have_refused_the_same_dump() -> None:
    """Guard on the test itself: with the old connector the same message is rejected."""
    dump = _book_dump(_token_ids())
    subscriptions: list[str] = []
    server, port = await _serve_one_dump(dump, subscriptions)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:  # 1 MiB default
            await ws.send("{}")
            with pytest.raises(websockets.exceptions.ConnectionClosedError) as caught:
                await asyncio.wait_for(ws.recv(), timeout=10.0)
        assert caught.value.sent is not None and caught.value.sent.code == 1009
    finally:
        server.close()
        await server.wait_closed()


async def _no_rest() -> dict[str, Any]:
    return {}


async def _no_resync(stop_event: Any) -> None:
    return None

"""Deterministic integration tests for Gamma discovery and CLOB WS order book stream."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from collector.clob_ws_stream import ClobOrderBookStream, parse_book_levels
from collector.gamma_discovery import GammaDiscoveryClient, discover_active_market_groups
from contracts.models import OrderBookSnapshot


class _FakeWebSocket:
    """In-memory async WebSocket test double supporting recv, send, and close."""

    def __init__(self, incoming: list[str | Exception]) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.closed = False
        self._blocker: asyncio.Future[str] | None = None

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if self._incoming:
            item = self._incoming.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        loop = asyncio.get_running_loop()
        self._blocker = loop.create_future()
        return await self._blocker

    async def close(self) -> None:
        self.closed = True
        if self._blocker is not None and not self._blocker.done():
            self._blocker.cancel()


@pytest.mark.asyncio
async def test_gamma_discovery_negrisk_binary_and_filtering() -> None:
    captured_requests: list[httpx.Request] = []
    events_payload = [
        {
            "id": "evt_election",
            "title": "Presidential Election",
            "active": True,
            "closed": False,
            "negRisk": True,
            "markets": [
                {
                    "id": "mkt_alice",
                    "question": "Will Alice win?",
                    "groupItemTitle": "Alice",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 1500.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_alice_yes", "tok_alice_no"]',
                },
                {
                    "id": "mkt_bob",
                    "question": "Will Bob win?",
                    "groupItemTitle": "Bob",
                    "active": True,
                    "closed": False,
                    "liquidity": "800.5",
                    "outcomes": ["No", "Yes"],
                    "clobTokenIds": ["tok_bob_no", "tok_bob_yes"],
                },
                {
                    "id": "mkt_low_liq",
                    "question": "Will Carol win?",
                    "groupItemTitle": "Carol",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 50.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_carol_yes", "tok_carol_no"]',
                },
                {
                    "id": "mkt_closed",
                    "question": "Will Dave win?",
                    "groupItemTitle": "Dave",
                    "active": True,
                    "closed": True,
                    "liquidityNum": 5000.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_dave_yes", "tok_dave_no"]',
                },
            ],
        },
        {
            "id": "evt_complete_negrisk",
            "title": "Complete NegRisk Race",
            "active": True,
            "closed": False,
            "negRisk": True,
            "markets": [
                {
                    "id": "mkt_c1",
                    "question": "Candidate 1 wins?",
                    "groupItemTitle": "C1",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 1000.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_c1_yes", "tok_c1_no"]',
                },
                {
                    "id": "mkt_c2",
                    "question": "Candidate 2 wins?",
                    "groupItemTitle": "C2",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 1200.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_c2_yes", "tok_c2_no"]',
                },
                {
                    "id": "mkt_c3",
                    "question": "Candidate 3 wins?",
                    "groupItemTitle": "C3",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 900.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_c3_yes", "tok_c3_no"]',
                },
            ],
        },
        {
            "id": "evt_non_negrisk_multi",
            "title": "Non-NegRisk Multi Deadlines",
            "active": True,
            "closed": False,
            "negRisk": False,
            "markets": [
                {
                    "id": "mkt_q1",
                    "question": "Event happens by Q1?",
                    "groupItemTitle": "Q1",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 800.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_q1_yes", "tok_q1_no"]',
                },
                {
                    "id": "mkt_q2",
                    "question": "Event happens by Q2?",
                    "groupItemTitle": "Q2",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 800.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_q2_yes", "tok_q2_no"]',
                },
            ],
        },
        {
            "id": "evt_binary",
            "title": "BTC above 100k?",
            "active": True,
            "closed": False,
            "negRisk": False,
            "markets": [
                {
                    "id": "mkt_btc",
                    "question": "BTC above 100k by Dec?",
                    "active": True,
                    "closed": False,
                    "volume": "2500.0",
                    "outcomes": ["Yes", "No"],
                    "clobTokenIds": ["tok_btc_yes", "tok_btc_no"],
                }
            ],
        },
        {
            "id": "evt_closed",
            "title": "Closed Event",
            "active": True,
            "closed": True,
            "markets": [
                {
                    "id": "mkt_c",
                    "question": "Closed?",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 5000.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_c1", "tok_c2"]',
                }
            ],
        },
        {
            "id": "evt_inactive",
            "title": "Inactive Event",
            "active": False,
            "closed": False,
            "markets": [
                {
                    "id": "mkt_i",
                    "question": "Inactive?",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 5000.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_i1", "tok_i2"]',
                }
            ],
        },
        {
            "id": "evt_single_surviving_negrisk",
            "title": "Incomplete Multi",
            "active": True,
            "closed": False,
            "enableNegRisk": True,
            "markets": [
                {
                    "id": "mkt_s1",
                    "question": "Only 1 survives",
                    "groupItemTitle": "One",
                    "active": True,
                    "closed": False,
                    "liquidityNum": 500.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_s1_yes", "tok_s1_no"]',
                },
                {
                    "id": "mkt_s2",
                    "question": "Closed leg",
                    "groupItemTitle": "Two",
                    "active": False,
                    "closed": False,
                    "liquidityNum": 500.0,
                    "outcomes": '["Yes", "No"]',
                    "clobTokenIds": '["tok_s2_yes", "tok_s2_no"]',
                },
            ],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json=events_payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        groups = await discover_active_market_groups(
            limit=50,
            min_liquidity_usdc=200.0,
            http_client=client,
            order_by=None,  # this test is about the parsing rules, in the order the API answers
        )

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.method == "GET"
    assert req.url.path == "/events"
    assert req.url.params["active"] == "true"
    assert req.url.params["closed"] == "false"
    assert req.url.params["limit"] == "50"

    assert len(groups) == 4

    election = groups[0]
    assert election.event_id == "evt_election"
    # Partially filtered negRisk event cannot be treated as exhaustive sum-to-one arb.
    assert election.mutually_exclusive is False
    assert [(o.outcome_label, o.token_id) for o in election.outcomes] == [
        ("Alice", "tok_alice_yes"),
        ("Bob", "tok_bob_yes"),
    ]

    complete_negrisk = groups[1]
    assert complete_negrisk.event_id == "evt_complete_negrisk"
    assert complete_negrisk.mutually_exclusive is True
    assert [(o.outcome_label, o.token_id) for o in complete_negrisk.outcomes] == [
        ("C1", "tok_c1_yes"),
        ("C2", "tok_c2_yes"),
        ("C3", "tok_c3_yes"),
    ]

    non_negrisk_multi = groups[2]
    assert non_negrisk_multi.event_id == "evt_non_negrisk_multi"
    assert non_negrisk_multi.mutually_exclusive is False
    assert [(o.outcome_label, o.token_id) for o in non_negrisk_multi.outcomes] == [
        ("Q1", "tok_q1_yes"),
        ("Q2", "tok_q2_yes"),
    ]

    binary = groups[3]
    assert binary.event_id == "evt_binary"
    assert binary.mutually_exclusive is True
    assert [(o.outcome_label, o.token_id) for o in binary.outcomes] == [
        ("Yes", "tok_btc_yes"),
        ("No", "tok_btc_no"),
    ]


@pytest.mark.asyncio
async def test_gamma_discovery_wrapped_data_fallback_labels_and_malformed_liquidity() -> None:
    payload = {
        "data": [
            {
                "id": "evt_wrapped",
                "title": "Wrapped Event",
                "active": True,
                "closed": False,
                "markets": [
                    {
                        "id": "mkt_bad_liq",
                        "question": "Bad liq",
                        "active": True,
                        "closed": False,
                        "liquidityNum": "not-a-number",
                        "liquidity": None,
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["t1", "t2"]',
                    }
                ],
            },
            {
                "id": "evt_one_token",
                "title": "Only One Token",
                "active": True,
                "closed": False,
                "markets": [
                    {
                        "id": "mkt_one",
                        "question": "Broken",
                        "active": True,
                        "closed": False,
                        "liquidityNum": 1000.0,
                        "outcomes": '["Yes"]',
                        "clobTokenIds": '["only_one"]',
                    }
                ],
            },
            {
                "id": "evt_fallback_question",
                "title": "Multi Without GroupItemTitle",
                "active": True,
                "closed": False,
                "enableNegRisk": True,
                "markets": [
                    {
                        "id": "m1",
                        "question": "Option Alpha?",
                        "groupItemTitle": "",
                        "active": True,
                        "closed": False,
                        "liquidityNum": 500.0,
                        "clobTokenIds": '["tok_m1_yes", "tok_m1_no"]',
                    },
                    {
                        "id": "m2",
                        "question": "Option Beta?",
                        "active": True,
                        "closed": False,
                        "liquidityNum": 500.0,
                        "clobTokenIds": '["tok_m2_yes", "tok_m2_no"]',
                    },
                ],
            },
        ]
    }

    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        # order_by=None: parsing rules only, in the order the API answers.
        strict_client = GammaDiscoveryClient(min_liquidity_usdc=10.0, http_client=client, order_by=None)
        strict_groups = await strict_client.discover_market_groups()
        assert len(strict_groups) == 1
        assert strict_groups[0].event_id == "evt_fallback_question"
        assert [o.outcome_label for o in strict_groups[0].outcomes] == ["Option Alpha?", "Option Beta?"]

        permissive_client = GammaDiscoveryClient(min_liquidity_usdc=0.0, http_client=client, order_by=None)
        permissive_groups = await permissive_client.discover_market_groups()
        assert [g.event_id for g in permissive_groups] == ["evt_wrapped", "evt_fallback_question"]


def test_parse_book_levels_validation_and_sorting() -> None:
    raw = [
        {"price": "0.40", "size": "10"},
        {"price": "0.60", "size": "25"},
        {"price": "0.50", "size": "0"},      # invalid zero size
        {"price": "0.55", "size": "-5"},     # invalid negative size
        {"price": "1.05", "size": "10"},     # invalid price > 1.0
        {"price": "-0.01", "size": "10"},    # invalid price < 0.0
        {"price": "nan", "size": "10"},      # invalid NaN
        {"price": "bad", "size": "10"},      # invalid non-float
        {"price": "0.0", "size": "5"},       # valid boundary price 0.0
        {"price": "1.0", "size": "5"},       # valid boundary price 1.0
    ]

    bids = parse_book_levels(raw, descending=True)
    asks = parse_book_levels(raw, descending=False)

    assert [(lvl.price, lvl.size) for lvl in bids] == [
        (1.0, 5.0),
        (0.60, 25.0),
        (0.40, 10.0),
        (0.0, 5.0),
    ]
    assert [(lvl.price, lvl.size) for lvl in asks] == [
        (0.0, 5.0),
        (0.40, 10.0),
        (0.60, 25.0),
        (1.0, 5.0),
    ]


@pytest.mark.asyncio
async def test_clob_rest_snapshot_and_ws_delta_updates() -> None:
    def rest_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/book"
        token_id = request.url.params["token_id"]
        if token_id == "tok_1":
            return httpx.Response(
                200,
                json={
                    "bids": [{"price": "0.45", "size": "100"}, {"price": "0.48", "size": "50"}],
                    "asks": [{"price": "0.55", "size": "70"}, {"price": "0.52", "size": "30"}],
                },
            )
        return httpx.Response(200, json={"bids": [], "asks": []})

    transport = httpx.MockTransport(rest_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        stream = ClobOrderBookStream(
            token_ids=["tok_1", "tok_2"],
            stale_threshold_sec=30.0,
            http_client=client,
        )
        books = await stream.initialize_books()

    assert set(books.keys()) == {"tok_1", "tok_2"}
    assert [lvl.price for lvl in books["tok_1"].bids] == [0.48, 0.45]
    assert [lvl.price for lvl in books["tok_1"].asks] == [0.52, 0.55]
    assert books["tok_1"].is_stale is False
    assert books["tok_2"].bids == [] and books["tok_2"].asks == []

    t0 = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)

    # PONG & invalid messages are ignored cleanly
    assert stream.apply_ws_message("PONG", now=t0) == []
    assert stream.apply_ws_message("   ", now=t0) == []
    assert stream.apply_ws_message("not json", now=t0) == []

    # Schema 1: top-level asset_id + changes (plus invalid out-of-range change skipped safely)
    delta_msg_1 = json.dumps(
        {
            "event_type": "price_change",
            "asset_id": "tok_1",
            "changes": [
                {"price": "0.48", "size": "0", "side": "BUY"},    # delete 0.48 bid
                {"price": "0.49", "size": "80", "side": "BUY"},   # insert 0.49 bid
                {"price": "0.52", "size": "45", "side": "SELL"},  # update 0.52 ask
                {"price": "0.50", "size": "20", "side": "SELL"},  # insert 0.50 ask
                {"price": "1.25", "size": "10", "side": "SELL"},  # invalid price > 1 ignored
                {"price": "0.51", "size": "10", "side": "HOLD"},  # invalid side ignored
            ],
        }
    )
    updated_1 = stream.apply_ws_message(delta_msg_1, now=t0)
    assert len(updated_1) == 1
    assert [(b.price, b.size) for b in updated_1[0].bids] == [(0.49, 80.0), (0.45, 100.0)]
    assert [(a.price, a.size) for a in updated_1[0].asks] == [(0.50, 20.0), (0.52, 45.0), (0.55, 70.0)]

    # Schema 2: price_changes list across multiple asset_ids
    delta_msg_2 = {
        "event_type": "price_change",
        "price_changes": [
            {"asset_id": "tok_1", "price": "0.50", "size": "0", "side": "SELL"},
            {"asset_id": "tok_2", "price": "0.31", "size": "200", "side": "BUY"},
        ],
    }
    updated_2 = stream.apply_ws_message(delta_msg_2, now=t0 + timedelta(seconds=5))
    assert {s.token_id for s in updated_2} == {"tok_1", "tok_2"}
    tok1_snap = stream.get_book("tok_1", now=t0 + timedelta(seconds=10))
    tok2_snap = stream.get_book("tok_2", now=t0 + timedelta(seconds=10))
    assert tok1_snap is not None and [a.price for a in tok1_snap.asks] == [0.52, 0.55]
    assert tok2_snap is not None and [(b.price, b.size) for b in tok2_snap.bids] == [(0.31, 200.0)]

    # Full book replacement via list payload
    book_event = json.dumps(
        [
            {
                "event_type": "book",
                "asset_id": "tok_1",
                "bids": [{"price": "0.40", "size": "15"}],
                "asks": [{"price": "0.60", "size": "25"}],
            }
        ]
    )
    t1 = t0 + timedelta(seconds=15)
    updated_3 = stream.apply_ws_message(book_event, now=t1)
    assert len(updated_3) == 1
    assert [(b.price, b.size) for b in updated_3[0].bids] == [(0.40, 15.0)]
    assert [(a.price, a.size) for a in updated_3[0].asks] == [(0.60, 25.0)]

    # Staleness threshold expiry check across get_book and get_all_books
    fresh = stream.get_book("tok_1", now=t1 + timedelta(seconds=30))
    assert fresh is not None and fresh.is_stale is False
    expired = stream.get_book("tok_1", now=t1 + timedelta(seconds=30.1))
    assert expired is not None and expired.is_stale is True
    assert stream.get_book("missing_token") is None

    # Staleness follows connection liveness: the last frame arrived at t1, so tok_2 (last
    # changed at t0+5s) is still a valid, unchanged book until the connection goes quiet.
    all_at_t1_plus_20 = stream.get_all_books(now=t1 + timedelta(seconds=20))
    assert all_at_t1_plus_20["tok_1"].is_stale is False
    assert all_at_t1_plus_20["tok_2"].is_stale is False

    all_at_t1_plus_25 = stream.get_all_books(now=t1 + timedelta(seconds=25))
    assert all_at_t1_plus_25["tok_1"].is_stale is False
    assert all_at_t1_plus_25["tok_2"].is_stale is False

    # No frame for more than 30s (not even a PONG): every book is stale.
    all_silent = stream.get_all_books(now=t1 + timedelta(seconds=30.1))
    assert all_silent["tok_1"].is_stale is True
    assert all_silent["tok_2"].is_stale is True

    # A PONG alone keeps the books of a live connection fresh.
    stream.apply_ws_message("PONG", now=t1 + timedelta(seconds=28))
    kept_alive = stream.get_all_books(now=t1 + timedelta(seconds=50))
    assert kept_alive["tok_1"].is_stale is False
    assert kept_alive["tok_2"].is_stale is False

    stream.mark_all_stale()
    assert stream.get_book("tok_1", now=t1) is not None
    assert stream.get_book("tok_1", now=t1).is_stale is True  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_clob_ping_loop_emits_heartbeats() -> None:
    fake_ws = _FakeWebSocket([])
    stop_event = asyncio.Event()
    stream = ClobOrderBookStream(token_ids=["tok_1"], ping_interval_sec=0.01)

    task = asyncio.create_task(stream._ping_loop(fake_ws, stop_event=stop_event))
    await asyncio.sleep(0.055)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert len(fake_ws.sent) >= 2
    assert all(msg == "PING" for msg in fake_ws.sent)


@pytest.mark.asyncio
async def test_clob_run_emits_ping_subscription_and_async_callback() -> None:
    rest_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"bids": [{"price": "0.40", "size": "10"}], "asks": [{"price": "0.60", "size": "10"}]},
        )
    )
    fake_ws = _FakeWebSocket(
        incoming=[
            "PONG",
            json.dumps(
                {
                    "event_type": "book",
                    "asset_id": "tok_1",
                    "bids": [{"price": "0.49", "size": "40"}],
                    "asks": [{"price": "0.51", "size": "35"}],
                }
            ),
        ]
    )

    @asynccontextmanager
    async def fake_ws_connect(url: str) -> AsyncIterator[_FakeWebSocket]:
        assert url == "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        yield fake_ws

    received: list[OrderBookSnapshot] = []
    stop_event = asyncio.Event()

    async def on_update(snapshot: OrderBookSnapshot) -> None:
        received.append(snapshot)

    async with httpx.AsyncClient(transport=rest_transport) as http_client:
        stream = ClobOrderBookStream(
            token_ids=["tok_1"],
            ping_interval_sec=0.01,
            on_book_update=on_update,
            http_client=http_client,
        )
        run_task = asyncio.create_task(stream.run(stop_event=stop_event, ws_connect=fake_ws_connect))

        # Wait until update received AND at least 2 PING frames emitted during live run
        for _ in range(100):
            if len(received) == 1 and fake_ws.sent.count("PING") >= 2:
                break
            await asyncio.sleep(0.01)

        stop_event.set()
        await asyncio.wait_for(run_task, timeout=1.0)

    assert json.loads(fake_ws.sent[0]) == {"assets_ids": ["tok_1"], "type": "market"}
    assert fake_ws.sent.count("PING") >= 2
    assert len(received) == 1
    assert received[0].bids[0].price == 0.49
    assert received[0].is_stale is False
    # Clean stop via stop_event should not mark books stale
    current = stream.get_book("tok_1")
    assert current is not None and current.is_stale is False


@pytest.mark.asyncio
async def test_clob_run_disconnect_marks_stale_and_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest_transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            json={"bids": [{"price": "0.45", "size": "10"}], "asks": [{"price": "0.55", "size": "10"}]},
        )
    )
    recorded_sleeps: list[float] = []
    stop_event = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)
        if len(recorded_sleeps) >= 3:
            stop_event.set()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    stale_observed: list[bool] = []
    attempt = 0

    @asynccontextmanager
    async def fake_ws_connect(url: str) -> AsyncIterator[_FakeWebSocket]:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            yield _FakeWebSocket(
                incoming=[
                    json.dumps(
                        {
                            "event_type": "book",
                            "asset_id": "tok_1",
                            "bids": [{"price": "0.47", "size": "50"}],
                            "asks": [{"price": "0.53", "size": "60"}],
                        }
                    ),
                    ConnectionError("drop #1"),
                ]
            )
            return
        if attempt == 2:
            snap = stream.get_book("tok_1")
            if snap is not None:
                stale_observed.append(snap.is_stale)
            raise ConnectionError("connect fail #2")
        raise ConnectionError(f"connect fail #{attempt}")

    async with httpx.AsyncClient(transport=rest_transport) as http_client:
        stream = ClobOrderBookStream(
            token_ids=["tok_1"],
            ping_interval_sec=10.0,
            max_backoff_sec=60.0,
            http_client=http_client,
        )
        await asyncio.wait_for(
            stream.run(stop_event=stop_event, ws_connect=fake_ws_connect),
            timeout=2.0,
        )

    assert stale_observed == [True]
    assert recorded_sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_clob_run_resets_backoff_and_caps_at_max_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest_transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"bids": [], "asks": []})
    )
    recorded_sleeps: list[float] = []
    stop_event = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)
        if len(recorded_sleeps) >= 5:
            stop_event.set()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    attempt = 0

    @asynccontextmanager
    async def fake_ws_connect(url: str) -> AsyncIterator[_FakeWebSocket]:
        nonlocal attempt
        attempt += 1
        if attempt in (1, 2):
            raise ConnectionError(f"fail {attempt}")
        if attempt == 3:
            yield _FakeWebSocket(
                incoming=[
                    json.dumps(
                        {
                            "event_type": "book",
                            "asset_id": "tok_1",
                            "bids": [{"price": "0.50", "size": "10"}],
                            "asks": [{"price": "0.51", "size": "10"}],
                        }
                    ),
                    ConnectionError("drop after message"),
                ]
            )
            return
        raise ConnectionError(f"fail {attempt}")

    async with httpx.AsyncClient(transport=rest_transport) as http_client:
        stream = ClobOrderBookStream(
            token_ids=["tok_1"],
            ping_interval_sec=10.0,
            max_backoff_sec=3.0,
            http_client=http_client,
        )
        await asyncio.wait_for(
            stream.run(stop_event=stop_event, ws_connect=fake_ws_connect),
            timeout=2.0,
        )

    assert recorded_sleeps == [1.0, 2.0, 1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_clob_run_async_iterator_ws_and_eof_marks_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AsyncIterWebSocket:
        def __init__(self, messages: list[str]) -> None:
            self._messages = list(messages)
            self.sent: list[str] = []

        async def send(self, payload: str) -> None:
            self.sent.append(payload)

        def __aiter__(self) -> AsyncIterator[str]:
            return self

        async def __anext__(self) -> str:
            if not self._messages:
                raise StopAsyncIteration
            return self._messages.pop(0)

    rest_transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"bids": [], "asks": []})
    )
    recorded_sleeps: list[float] = []
    stop_event = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        recorded_sleeps.append(delay)
        stop_event.set()

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def fake_ws_connect(url: str) -> _AsyncIterWebSocket:
        return _AsyncIterWebSocket(
            [
                json.dumps(
                    {
                        "event_type": "book",
                        "asset_id": "tok_1",
                        "bids": [{"price": "0.42", "size": "15"}],
                        "asks": [{"price": "0.58", "size": "20"}],
                    }
                )
            ]
        )

    updates: list[OrderBookSnapshot] = []
    async with httpx.AsyncClient(transport=rest_transport) as http_client:
        stream = ClobOrderBookStream(
            token_ids=["tok_1"],
            ping_interval_sec=10.0,
            on_book_update=updates.append,
            http_client=http_client,
        )
        await asyncio.wait_for(
            stream.run(stop_event=stop_event, ws_connect=fake_ws_connect),
            timeout=2.0,
        )

    assert len(updates) == 1
    assert updates[0].bids[0].price == 0.42
    assert recorded_sleeps == [1.0]
    snap = stream.get_book("tok_1")
    assert snap is not None and snap.is_stale is True

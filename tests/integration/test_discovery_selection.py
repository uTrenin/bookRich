"""Which markets the bot follows, and what counts as a lost update.

Both rules come from the first recording of real data (2026-11-16):
- Gamma returns the oldest events first unless asked otherwise, and none of those could
  be arbitraged; the bot now asks for the most traded and keeps the multi-outcome ones.
- every real `price_change` carries two entries, the token we follow and its complement;
  the complement must not be reported as a lost update.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from collector.clob_ws_stream import ClobOrderBookStream
from collector.gamma_discovery import GammaDiscoveryClient
from paper_trader.record import record_session
from paper_trader.replay import replay_session

from .test_record_replay import _adapter, _FakeExchange, _trades

T0 = datetime(2026, 11, 16, 20, 45, tzinfo=timezone.utc)


def _market(market_id: str, label: str, *, closed: bool = False) -> dict[str, Any]:
    return {
        "id": market_id,
        "question": f"{label}?",
        "groupItemTitle": label,
        "active": True,
        "closed": closed,
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": json.dumps([f"{market_id}-yes", f"{market_id}-no"]),
    }


def _event(event_id: str, volume_24h: float, markets: list[dict[str, Any]], *, neg_risk: bool) -> dict[str, Any]:
    return {
        "id": event_id,
        "title": f"Event {event_id}",
        "active": True,
        "closed": False,
        "negRisk": neg_risk,
        "volume24hr": volume_24h,
        "markets": markets,
    }


# In the order Gamma would return them without an explicit order: oldest first.
EVENTS = [
    _event("ladder", 900.0, [_market("l1", "By March"), _market("l2", "By June")], neg_risk=False),
    _event("binary-hot", 50_000.0, [_market("b1", "Team A wins")], neg_risk=False),
    _event("election-quiet", 300.0, [_market("q1", "Ann"), _market("q2", "Bo"), _market("q3", "Cy")], neg_risk=True),
    _event("election-hot", 8_000.0, [_market("h1", "Dee"), _market("h2", "Eli")], neg_risk=True),
    # One candidate's market is already closed: the group is not exhaustive any more.
    _event(
        "election-partial",
        20_000.0,
        [_market("p1", "Flo"), _market("p2", "Gus"), _market("p3", "Hal", closed=True)],
        neg_risk=True,
    ),
]


def _transport(requests: list[httpx.Request], *, refuse_order_with: int | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if refuse_order_with is not None and "order" in request.url.params:
            return httpx.Response(refuse_order_with, json={"error": "order not supported"})
        return httpx.Response(200, json=EVENTS)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_discovery_asks_for_the_most_traded_events_and_looks_at_more_than_it_keeps() -> None:
    requests: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(requests)) as http_client:
        groups = await GammaDiscoveryClient(http_client=http_client).discover_market_groups(limit=2)

    assert len(requests) == 1
    params = requests[0].url.params
    assert params["order"] == "volume24hr" and params["ascending"] == "false"
    assert params["active"] == "true" and params["closed"] == "false"
    assert params["limit"] == "10"  # 5 candidates for each group kept

    # The two groups where an arbitrage can exist, the more traded one first.
    assert [g.event_id for g in groups] == ["election-hot", "election-quiet"]
    assert all(g.mutually_exclusive for g in groups)


@pytest.mark.asyncio
async def test_discovery_ranks_multi_outcome_groups_first_then_by_volume() -> None:
    async with httpx.AsyncClient(transport=_transport([])) as http_client:
        client = GammaDiscoveryClient(http_client=http_client)
        groups = await client.discover_market_groups(limit=10)

    assert [g.event_id for g in groups] == [
        "election-hot", "election-quiet",  # exhaustive, several markets
        "binary-hot",  # YES and NO of one market: mirrored book, kept for the news strategy
        "election-partial",  # not exhaustive
        "ladder",
    ]
    by_id = {g.event_id: g for g in groups}
    assert client.is_multi_outcome_arbitrage_group(by_id["election-hot"]) is True
    assert by_id["binary-hot"].mutually_exclusive is True
    assert client.is_multi_outcome_arbitrage_group(by_id["binary-hot"]) is False
    assert by_id["election-partial"].mutually_exclusive is False


@pytest.mark.asyncio
async def test_discovery_falls_back_to_the_plain_order_when_ordering_is_refused() -> None:
    requests: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(requests, refuse_order_with=422)) as http_client:
        groups = await GammaDiscoveryClient(http_client=http_client).discover_market_groups(limit=2)

    assert ["order" in r.url.params for r in requests] == [True, False]
    assert requests[1].url.params["limit"] == "10"
    # Same selection, done on our side with the volumes in the answer.
    assert [g.event_id for g in groups] == ["election-hot", "election-quiet"]


@pytest.mark.asyncio
async def test_discovery_does_not_hide_a_server_error() -> None:
    requests: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(requests, refuse_order_with=503)) as http_client:
        with pytest.raises(httpx.HTTPStatusError):
            await GammaDiscoveryClient(http_client=http_client).discover_market_groups(limit=2)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_discovery_without_ordering_keeps_the_api_order() -> None:
    requests: list[httpx.Request] = []
    async with httpx.AsyncClient(transport=_transport(requests)) as http_client:
        groups = await GammaDiscoveryClient(http_client=http_client, order_by=None).discover_market_groups(limit=7)

    assert "order" not in requests[0].url.params and requests[0].url.params["limit"] == "7"
    assert [g.event_id for g in groups] == ["ladder", "binary-hot", "election-quiet", "election-hot", "election-partial"]


def test_candidate_pool_is_capped_and_never_smaller_than_the_limit() -> None:
    client = GammaDiscoveryClient(candidate_pool_factor=0)
    assert client.candidate_pool_factor == 1
    assert GammaDiscoveryClient(order_by="  ").order_by is None


# --------------------------------------------------------------------------- lost updates


def _real_price_change(yes_token: str, no_token: str) -> dict[str, Any]:
    """Shape of a real `price_change`: one entry for each of the two tokens of the market."""
    return {
        "event_type": "price_change",
        "market": "0xabc",
        "timestamp": str(int(T0.timestamp() * 1000)),
        "price_changes": [
            {"asset_id": yes_token, "price": "0.51", "size": "40", "side": "SELL", "best_bid": "0.40", "best_ask": "0.50"},
            {"asset_id": no_token, "price": "0.49", "size": "40", "side": "BUY", "best_bid": "0.50", "best_ask": "0.60"},
        ],
    }


def _book(token_id: str) -> dict[str, Any]:
    return {
        "event_type": "book",
        "asset_id": token_id,
        "asks": [{"price": "0.50", "size": "100"}],
        "bids": [{"price": "0.40", "size": "100"}],
    }


def test_complement_token_of_a_followed_market_is_not_a_lost_update() -> None:
    stream = ClobOrderBookStream(token_ids=["yes"])
    stream.apply_ws_message(_book("yes"), now=T0)

    updated = stream.apply_ws_message(_real_price_change("yes", "no"), now=T0)

    assert [snapshot.token_id for snapshot in updated] == ["yes"]
    assert [(lvl.price, lvl.size) for lvl in updated[0].asks] == [(0.50, 100.0), (0.51, 40.0)]
    assert stream.get_book("no", now=T0) is None  # no one-level book for the token we do not follow
    assert stream.ignored_delta_count == 0


def test_delta_for_a_followed_token_without_snapshot_is_still_counted() -> None:
    stream = ClobOrderBookStream(token_ids=["yes"])
    assert stream.apply_ws_message(_real_price_change("yes", "no"), now=T0) == []
    assert stream.ignored_delta_count == 1


@pytest.mark.asyncio
async def test_a_token_no_longer_followed_stops_counting() -> None:
    stream = ClobOrderBookStream(token_ids=["yes", "old"])
    await stream.update_tokens(["yes"])  # removal only: no snapshot to fetch
    stream.apply_ws_message(_real_price_change("old", "old-no"), now=T0)
    assert stream.ignored_delta_count == 0


# --------------------------------------------------------------------------- recordings


@pytest.mark.asyncio
async def test_recording_stores_how_markets_were_chosen_and_old_recordings_still_replay(tmp_path: Path) -> None:
    pytest.importorskip("engine.combinatorial_arb", reason="needs the real /engine")

    async def record(name: str, **settings: Any) -> tuple[Path, Path]:
        exchange = _FakeExchange()
        session_file, live_log = tmp_path / f"{name}.jsonl", tmp_path / f"{name}-live.jsonl"
        await record_session(
            session_file, minutes=1.0, log_path=live_log, discovery_limit=20, engine=_adapter(),
            inner_transport=httpx.MockTransport(exchange.handle_http), inner_connector=exchange.connector(),
            stop_event=exchange.finished, **settings,
        )
        return session_file, live_log

    def config_of(session_file: Path) -> dict[str, Any]:
        lines = [json.loads(line) for line in session_file.read_text(encoding="utf-8").splitlines()]
        return next(line for line in lines if line["kind"] == "config")

    # A recording made today says how the markets were chosen, and replays the same way.
    new_file, new_live = await record("new")
    assert config_of(new_file)["discovery_order"] == "volume24hr"
    assert config_of(new_file)["discovery_pool_factor"] == 5
    report = await replay_session(new_file, tmp_path / "new-replay.jsonl", engine=_adapter())
    assert report["replay"]["unmatched_requests"] == 0
    assert _trades(tmp_path / "new-replay.jsonl") == _trades(new_live)

    # A recording made before this setting existed has no such entry: plain API order.
    old_file, old_live = await record("old", discovery_order=None, discovery_pool_factor=1)
    lines = old_file.read_text(encoding="utf-8").splitlines()
    stripped = []
    for line in lines:
        entry = json.loads(line)
        if entry["kind"] == "config":
            entry.pop("discovery_order")
            entry.pop("discovery_pool_factor")
        stripped.append(json.dumps(entry))
    old_file.write_text("\n".join(stripped) + "\n", encoding="utf-8")

    report = await replay_session(old_file, tmp_path / "old-replay.jsonl", engine=_adapter())
    assert report["replay"]["unmatched_requests"] == 0 and report["replay"]["completed"] is True
    assert _trades(tmp_path / "old-replay.jsonl") == _trades(old_live)

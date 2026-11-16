"""End-to-end replay (T3-C): a scripted market session through the whole paper bot.

The session in `data/replay_session.json` is hand-built in the format of Polymarket's
public messages (Gamma /events, CLOB /book, WS market channel, an RSS feed). It is NOT a
live capture. It is replayed through the real components with zero network access:

    GammaDiscoveryClient -> ClobOrderBookStream (REST snapshot + WS + reconnect)
    -> PaperTradingRunner -> DynamicEngineAdapter -> /engine -> PaperOrderSimulator
    -> PaperTradeLogger (JSONL)

Only the transports are replaced: HTTP by httpx.MockTransport, the WebSocket by a socket
that replays the recorded frames, the LLM by canned answers from the session file.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator
from xml.sax.saxutils import escape

import httpx
import pytest

from collector.clob_ws_stream import ClobOrderBookStream
from collector.gamma_discovery import GammaDiscoveryClient
from collector.news_stream import GeminiNewsClassifier, NewsStreamCollector
from paper_trader.logger import PaperTradeLogger
from paper_trader.runner import DynamicEngineAdapter, PaperTradingRunner

pytest.importorskip("engine.combinatorial_arb", reason="the replay needs the real /engine")

SESSION_PATH = Path(__file__).parent / "data" / "replay_session.json"
FEED_URL = "https://news.replay.test/rss"

ECB_YES, ECB_NO = "100000000101", "100000000102"
FILM_GAMMA = "200000000203"
FED_YES = "400000000401"


@dataclass
class ReplayResult:
    """Everything the replay produced, for assertions and for a human-readable report."""

    log_entries: list[dict[str, Any]]
    stats: dict[str, Any]
    exposure_by_event: dict[str, float]
    total_exposure_usdc: float
    ws_connections_opened: int
    subscriptions: list[dict[str, Any]]
    llm_prompts: list[str] = field(default_factory=list)
    groups: dict[str, bool] = field(default_factory=dict)  # event_id -> mutually_exclusive


def _now_ms() -> str:
    return str(int(datetime.now(timezone.utc).timestamp() * 1000))


def _prepare_frame(frame: Any) -> str:
    """Drop annotation keys and stamp exchange timestamps with the replay time."""
    if isinstance(frame, str):
        return frame

    def clean(event: dict[str, Any]) -> dict[str, Any]:
        cleaned = {key: value for key, value in event.items() if not key.startswith("_")}
        if "timestamp" in cleaned:
            cleaned["timestamp"] = _now_ms()
        return cleaned

    if isinstance(frame, list):
        return json.dumps([clean(event) for event in frame])
    return json.dumps(clean(frame))


class _ReplaySocket:
    """WebSocket double that replays one recorded connection, then stays idle."""

    def __init__(self, frames: list[Any], on_control: Any) -> None:
        self._frames = list(frames)
        self._on_control = on_control
        self.sent: list[str] = []
        self._idle: asyncio.Future[str] | None = None

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        while self._frames:
            frame = self._frames.pop(0)
            if isinstance(frame, dict) and "_control" in frame:
                if frame["_control"] == "drop":
                    raise ConnectionError("recorded disconnect")
                self._on_control(frame["_control"])
                continue
            return _prepare_frame(frame)
        self._idle = asyncio.get_running_loop().create_future()
        return await self._idle

    async def close(self) -> None:
        if self._idle is not None and not self._idle.done():
            self._idle.cancel()


class _ReplayStream(ClobOrderBookStream):
    """Real stream; only the WebSocket connector is swapped for the replay one."""

    replay_connect: Any = None

    async def run(self, stop_event: asyncio.Event | None = None, ws_connect: Any = None) -> None:
        await super().run(stop_event=stop_event, ws_connect=self.replay_connect)


def _rss_payload(items: list[dict[str, Any]]) -> str:
    now = datetime.now(timezone.utc)
    entries = "".join(
        "<item>"
        f"<guid>{escape(item['guid'])}</guid>"
        f"<title>{escape(item['title'])}</title>"
        f"<description>{escape(item['description'])}</description>"
        f"<pubDate>{format_datetime(now - timedelta(seconds=item['age_sec']))}</pubDate>"
        "</item>"
        for item in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{entries}</channel></rss>'


async def run_replay_session(
    log_path: Path, timeout_sec: float = 10.0, taker_fee_bps: float = 0.0
) -> ReplayResult:
    """Replay the recorded session through the real bot and return what it produced."""
    session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    news_released = asyncio.Event()
    sockets: list[_ReplaySocket] = []
    prompts: list[str] = []

    def http_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/events":
            return httpx.Response(200, json=session["gamma_events"])
        if request.url.path == "/book":
            book = session["rest_books"].get(request.url.params["token_id"])
            return httpx.Response(200, json=book) if book else httpx.Response(404, json={"error": "unknown"})
        if str(request.url) == FEED_URL:
            items = session["rss_items"] if news_released.is_set() else []
            return httpx.Response(200, text=_rss_payload(items))
        return httpx.Response(500, text="unexpected request in replay")

    @asynccontextmanager
    async def replay_connect(url: str) -> AsyncIterator[_ReplaySocket]:
        index = len(sockets)
        frames = session["ws_connections"][index] if index < len(session["ws_connections"]) else []
        socket = _ReplaySocket(frames, on_control=lambda name: news_released.set() if name == "release_news" else None)
        sockets.append(socket)
        yield socket

    async def fake_generate(**kwargs: Any) -> Any:
        prompts.append(kwargs["contents"])
        for headline, answer in session["llm_responses"].items():
            if headline in kwargs["contents"]:
                return SimpleNamespace(text=json.dumps(answer))
        return SimpleNamespace(text='{"target_token_id": null, "estimated_fair_prob": 0.5, "confidence": 0.0}')

    llm_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate)))

    async with httpx.AsyncClient(transport=httpx.MockTransport(http_handler)) as http_client:
        discovery = GammaDiscoveryClient(http_client=http_client)
        groups = await discovery.discover_market_groups(limit=20)
        token_ids = list(dict.fromkeys(o.token_id for g in groups for o in g.outcomes))

        stream = _ReplayStream(token_ids=token_ids, http_client=http_client, max_backoff_sec=0.01)
        stream.replay_connect = replay_connect
        news = NewsStreamCollector(
            feed_urls=[FEED_URL],
            classifier=GeminiNewsClassifier(client=llm_client),
            poll_interval_sec=0.02,
            http_client=http_client,
        )
        # Explicit settings: the replay must not depend on the operator's environment.
        adapter = DynamicEngineAdapter(
            target_shares=25.0,
            min_arb_net_edge=0.015,
            news_max_age_sec=120.0,
            bankroll_usdc=1000.0,
            max_position_usdc=50.0,
            slippage_buffer=0.0,
            fee_rate=taker_fee_bps / 10_000.0,
        )
        runner = PaperTradingRunner(
            discovery_client=discovery,
            book_stream=stream,
            news_collector=news,
            engine=adapter,
            trade_logger=PaperTradeLogger(log_path=log_path),
            heartbeat_interval_sec=3600.0,
        )

        stop_event = asyncio.Event()
        run_task = asyncio.create_task(runner.run(stop_event=stop_event, discovery_limit=20))

        async def _session_consumed() -> None:
            # Both scripted news assessments handled => every recorded frame was processed first.
            while runner.get_heartbeat_stats()["news_assessments"] < len(session["llm_responses"]):
                if run_task.done():
                    return
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(_session_consumed(), timeout=timeout_sec)
        finally:
            stop_event.set()
            await asyncio.wait_for(run_task, timeout=timeout_sec)

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return ReplayResult(
        log_entries=entries,
        stats=runner.get_heartbeat_stats(),
        exposure_by_event=dict(adapter.event_exposure_usdc),
        total_exposure_usdc=adapter.current_exposure_usdc,
        ws_connections_opened=len(sockets),
        subscriptions=[json.loads(socket.sent[0]) for socket in sockets if socket.sent],
        llm_prompts=prompts,
        groups={g.event_id: g.mutually_exclusive for g in groups},
    )


@pytest.mark.asyncio
async def test_replay_session_end_to_end(tmp_path: Path) -> None:
    result = await run_replay_session(tmp_path / "replay.jsonl")

    # --- discovery: only real sum-to-one groups are flagged mutually exclusive
    assert result.groups == {"50101": True, "50202": True, "50303": False, "50404": True}

    # --- stream: one drop, one reconnect, same subscription both times
    assert result.ws_connections_opened == 2
    assert all(sub["type"] == "market" and len(sub["assets_ids"]) == 10 for sub in result.subscriptions)
    assert len(result.subscriptions) == 2

    # --- the JSONL log tells the whole story, in order
    trades = [e for e in result.log_entries if e["event_type"] != "HEARTBEAT_STATS"]
    assert [(e["event_type"], e["payload"]["strategy"], e["payload"]["event_id"]) for e in trades] == [
        ("PAPER_FILL", "COMBINATORIAL_ARB", "50101"),  # ECB YES+NO at 0.97
        ("PAPER_FILL", "COMBINATORIAL_ARB", "50101"),  # still there on the next update
        ("SIGNAL_REJECTED", "COMBINATORIAL_ARB", "50101"),  # third one would exceed the 50 USDC event cap
        ("PAPER_FILL", "COMBINATORIAL_ARB", "50202"),  # film: 0.95 per set with real depth
        ("PAPER_FILL", "NEWS_LATENCY", "50404"),  # Fed cut headline vs a 0.55 ask
    ]
    ecb_1, ecb_2, ecb_rejected, film, fed = (e["payload"] for e in trades)

    # ECB: 25 sets at 0.47 + 0.50
    for fill in (ecb_1, ecb_2):
        assert fill["status"] == "FILLED"
        assert fill["actual_simulated_cost_usdc"] == pytest.approx(24.25)
        assert fill["expected_net_edge"] == pytest.approx(0.03)
        assert fill["slippage_vs_best_ask"] == pytest.approx(0.0)
        assert [leg["token_id"] for leg in fill["filled_legs"]] == [ECB_YES, ECB_NO]
        assert {leg["target_shares"] for leg in fill["filled_legs"]} == {25.0}  # balanced legs
    assert ecb_rejected["reason"].startswith("EVENT_EXPOSURE_EXCEEDED")

    # Film: the 10-share ask at 0.20 was NOT traded (0.998 per set on 25 shares); the 120-share one was.
    assert film["actual_simulated_cost_usdc"] == pytest.approx(25 * 0.95)
    assert film["expected_net_edge"] == pytest.approx(0.05)
    gamma_leg = next(leg for leg in film["filled_legs"] if leg["token_id"] == FILM_GAMMA)
    assert gamma_leg["expected_vwap"] == pytest.approx(0.20)

    # Fed news: fractional Kelly would stake 125 USDC, capped at 50 -> 90.9 shares at 0.55
    assert fed["actual_simulated_cost_usdc"] == pytest.approx(50.0)
    assert fed["expected_net_edge"] == pytest.approx(0.25)
    assert fed["filled_legs"][0]["token_id"] == FED_YES
    assert fed["filled_legs"][0]["target_shares"] == pytest.approx(50.0 / 0.55)
    assert 0.0 <= fed["latency_ms"] < 60_000.0

    # --- what must NOT have happened
    traded_events = {e["payload"]["event_id"] for e in trades}
    assert "50303" not in traded_events  # nested BTC thresholds are not an arbitrage
    assert result.stats["ws_deltas_ignored"] == 1  # the 0.30 ask that arrived before the re-snapshot
    assert len(result.llm_prompts) == 2  # the 15-minute-old item never reached the LLM

    # --- counters and virtual exposure
    assert result.stats["signals_generated"] == 5
    assert result.stats["signals_approved"] == 4
    assert result.stats["fills_executed"] == 4
    assert result.stats["signals_rejected_risk"] == 1
    assert result.stats["news_assessments"] == 2  # the low-confidence rumor was assessed, then discarded
    assert result.stats["paper_only"] is True
    assert result.exposure_by_event == pytest.approx({"50101": 48.5, "50202": 23.75, "50404": 50.0})
    assert result.total_exposure_usdc == pytest.approx(122.25)
    assert result.log_entries[-1]["event_type"] == "HEARTBEAT_STATS"

    # --- virtual P&L at the end of the session (best bids of the last snapshot)
    pnl = result.stats["pnl"]
    fed_shares = 50.0 / 0.55
    market_value = 2 * 25 * (0.51 + 0.48) + 25 * (0.38 + 0.33 + 0.27) + fed_shares * 0.53
    assert pnl["open_lots"] == 4
    assert pnl["cost_basis_usdc"] == pytest.approx(122.25)
    assert pnl["market_value_usdc"] == pytest.approx(market_value)
    assert pnl["unrealized_pnl_usdc"] == pytest.approx(market_value - 122.25)  # about -0.07 USDC
    # The three arbitrage lots are worth 75 USDC at resolution whoever wins: +0.75, +0.75, +1.25.
    assert pnl["locked_arb_pnl_usdc"] == pytest.approx(2.75)
    assert pnl["realized_pnl_usdc"] == 0.0
    assert pnl["unpriced_tokens"] == 0
    assert pnl["equity_usdc"] == pytest.approx(1000.0 + market_value - 122.25)
    assert result.log_entries[-1]["payload"]["pnl"]["locked_arb_pnl_usdc"] == pytest.approx(2.75)


@pytest.mark.asyncio
async def test_replay_session_with_a_two_percent_taker_fee(tmp_path: Path) -> None:
    """Same session, 2% taker fee: thin opportunities disappear, the rest cost more."""
    result = await run_replay_session(tmp_path / "replay_fee.jsonl", taker_fee_bps=200.0)

    trades = [e for e in result.log_entries if e["event_type"] != "HEARTBEAT_STATS"]
    assert [(e["event_type"], e["payload"]["strategy"], e["payload"]["event_id"]) for e in trades] == [
        ("PAPER_FILL", "COMBINATORIAL_ARB", "50202"),
        ("PAPER_FILL", "NEWS_LATENCY", "50404"),
    ]
    film, fed = (e["payload"] for e in trades)

    # ECB: 0.97 per set becomes 0.9894 with the fee, above the 0.985 threshold -> never traded.
    assert "50101" not in result.exposure_by_event

    # Film: 0.95 per set becomes 0.969 -> still an arbitrage, with a thinner edge.
    assert film["expected_net_edge"] == pytest.approx(1.0 - 0.95 * 1.02)
    assert film["signal_total_cost_usdc"] == pytest.approx(23.75)
    assert film["fees_usdc"] == pytest.approx(23.75 * 0.02)
    assert film["actual_simulated_cost_usdc"] == pytest.approx(23.75 * 1.02)

    # Fed news: the 50 USDC cap now includes the fee, so fewer shares are bought.
    assert fed["actual_simulated_cost_usdc"] == pytest.approx(50.0)
    assert fed["fees_usdc"] == pytest.approx(50.0 * 0.02 / 1.02)
    assert fed["filled_legs"][0]["target_shares"] == pytest.approx(50.0 / 1.02 / 0.55)
    assert fed["expected_net_edge"] == pytest.approx(0.80 - 0.55 * 1.02)

    assert result.stats["taker_fee_bps"] == pytest.approx(200.0)
    assert result.stats["signals_generated"] == 2
    assert result.stats["signals_rejected_risk"] == 0
    assert result.exposure_by_event == pytest.approx({"50202": 23.75 * 1.02, "50404": 50.0})

    pnl = result.stats["pnl"]
    assert pnl["fees_paid_usdc"] == pytest.approx(23.75 * 0.02 + 50.0 * 0.02 / 1.02)
    assert pnl["locked_arb_pnl_usdc"] == pytest.approx(25.0 - 23.75 * 1.02)  # the fee eats into the lock

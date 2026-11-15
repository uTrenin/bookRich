"""Record what the bot receives, replay it offline, and get the same decisions."""

from __future__ import annotations

import asyncio
import gzip
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

from collector.clob_ws_stream import ClobOrderBookStream
from contracts.models import NewsAssessment
from paper_trader.record import record_session
from paper_trader.replay import format_replay, main as replay_main, replay_session
from paper_trader.runner import DynamicEngineAdapter
from paper_trader.session_io import RecordingTransport, SessionRecorder, SessionReplay, record_assessments

from .test_replay_e2e import SESSION_PATH, _prepare_frame

pytest.importorskip("engine.combinatorial_arb", reason="needs the real /engine")


def _adapter() -> DynamicEngineAdapter:
    return DynamicEngineAdapter(
        target_shares=25.0, min_arb_net_edge=0.015, news_max_age_sec=120.0, bankroll_usdc=1000.0,
        max_position_usdc=50.0, fee_buffer=0.0, slippage_buffer=0.0, fee_rate=0.0,
    )


class _FakeExchange:
    """Stands in for Polymarket: serves the scripted session over HTTP and WebSocket."""

    def __init__(self) -> None:
        self.session = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        self.connections = 0
        self.finished = asyncio.Event()

    def handle_http(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/events":
            return httpx.Response(200, json=self.session["gamma_events"])
        book = self.session["rest_books"].get(request.url.params.get("token_id", ""))
        return httpx.Response(200, json=book) if book else httpx.Response(404, json={"error": "unknown"})

    def connector(self):
        exchange = self

        class _Socket:
            def __init__(self, frames: list[Any], last: bool) -> None:
                self._frames = [f for f in frames if not (isinstance(f, dict) and f.get("_control") == "release_news")]
                self._last = last

            async def send(self, message: str) -> None:
                return None

            async def recv(self) -> str:
                if self._frames:
                    frame = self._frames.pop(0)
                    if isinstance(frame, dict) and frame.get("_control") == "drop":
                        raise ConnectionError("exchange dropped the connection")
                    return _prepare_frame(frame)
                if self._last:
                    exchange.finished.set()
                await asyncio.Event().wait()
                return ""

            async def close(self) -> None:
                return None

        @asynccontextmanager
        async def connect(url: str) -> AsyncIterator[_Socket]:
            scripted = exchange.session["ws_connections"]
            index = exchange.connections
            exchange.connections += 1
            frames = scripted[index] if index < len(scripted) else []
            yield _Socket(frames, last=index >= len(scripted) - 1)

        return connect


def _trades(log_path: Path) -> list[tuple[str, str, str, float]]:
    out = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["event_type"] in {"PAPER_FILL", "SIGNAL_REJECTED"}:
            payload = entry["payload"]
            cost = payload.get("actual_simulated_cost_usdc", payload.get("total_cost_usdc", 0.0))
            out.append((entry["event_type"], payload["strategy"], payload["event_id"], round(float(cost), 6)))
    return out


async def _record(tmp_path: Path) -> tuple[Path, Path, dict[str, int]]:
    exchange = _FakeExchange()
    session_file, live_log = tmp_path / "session.jsonl", tmp_path / "live.jsonl"
    counts = await record_session(
        session_file,
        minutes=1.0,
        log_path=live_log,
        discovery_limit=20,
        engine=_adapter(),
        inner_transport=httpx.MockTransport(exchange.handle_http),
        inner_connector=exchange.connector(),
        stop_event=exchange.finished,
    )
    return session_file, live_log, counts


@pytest.mark.asyncio
async def test_recorded_session_replays_to_the_same_decisions(tmp_path: Path) -> None:
    session_file, live_log, counts = await _record(tmp_path)

    # What was captured: two discoveries, one book per token, two connections with a drop between.
    assert counts["meta"] == 1 and counts["config"] == 1
    assert counts["http"] == 2 + 10
    assert counts["ws_open"] == 2 and counts["ws_close"] == 1
    assert counts["ws_frame"] == 12
    assert counts["ws_send"] == 2  # one subscription per connection

    live = _trades(live_log)
    assert [(kind, event) for kind, _, event, _ in live] == [
        ("PAPER_FILL", "50101"), ("PAPER_FILL", "50101"), ("SIGNAL_REJECTED", "50101"), ("PAPER_FILL", "50202"),
    ]

    replay_log = tmp_path / "replay.jsonl"
    report = await replay_session(session_file, replay_log, engine=_adapter())

    assert _trades(replay_log) == live  # same decisions, same costs, same order
    assert report["fills_count"] == 3 and report["rejections_count"] == 1
    assert report["replay"] == {
        "session": str(session_file),
        "completed": True,
        "records_read": sum(counts.values()),
        "connections_recorded": 2,
        "frames_replayed": 12,
        "assessments_replayed": 0,
        "unmatched_requests": 0,
    }
    text = format_replay(report)
    assert "12 frame(s) over 2 connection(s)" in text and "WARNING" not in text


@pytest.mark.asyncio
async def test_old_recording_is_not_treated_as_stale(tmp_path: Path) -> None:
    session_file, live_log, _ = await _record(tmp_path)

    # Age the whole recording by two hours, as if it had been made earlier in the day.
    shift = timedelta(hours=2)
    aged_lines = []
    for line in session_file.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["t"] = (datetime.fromisoformat(record["t"]) - shift).isoformat()
        if record.get("kind") == "ws_frame" and record["data"].lstrip().startswith(("{", "[")):
            data = json.loads(record["data"])
            for event in data if isinstance(data, list) else [data]:
                if "timestamp" in event:
                    event["timestamp"] = str(int(event["timestamp"]) - int(shift.total_seconds() * 1000))
            record["data"] = json.dumps(data)
        aged_lines.append(json.dumps(record))
    aged_file = tmp_path / "aged.jsonl"
    aged_file.write_text("\n".join(aged_lines) + "\n", encoding="utf-8")

    replay_log = tmp_path / "aged_replay.jsonl"
    report = await replay_session(aged_file, replay_log, engine=_adapter())
    # Without shifting timestamps the feed would look two hours behind and nothing would trade.
    assert _trades(replay_log) == _trades(live_log)
    assert report["latest_heartbeat"]["feed_lag_sec"] < 5.0


@pytest.mark.asyncio
async def test_same_recording_under_a_different_fee(tmp_path: Path) -> None:
    session_file, _, _ = await _record(tmp_path)
    expensive = DynamicEngineAdapter(
        target_shares=25.0, min_arb_net_edge=0.015, bankroll_usdc=1000.0, max_position_usdc=50.0,
        fee_buffer=0.0, slippage_buffer=0.0, fee_rate=0.02,
    )
    replay_log = tmp_path / "fee.jsonl"
    await replay_session(session_file, replay_log, engine=expensive)
    # At 2% the ECB arbitrage (0.97 -> 0.9894) disappears; the film one survives.
    assert [(kind, event) for kind, _, event, _ in _trades(replay_log)] == [("PAPER_FILL", "50202")]


@pytest.mark.asyncio
async def test_replay_with_other_settings_reports_unmatched_requests(tmp_path: Path) -> None:
    session_file, _, _ = await _record(tmp_path)
    # /events?limit=7 was never recorded: fail with a message that says what to do.
    with pytest.raises(ValueError, match="no recorded answer.*discovery_limit=20"):
        await replay_session(session_file, tmp_path / "mismatch.jsonl", engine=_adapter(), discovery_limit=7)

    # A request missing later in the run is counted and flagged in the report instead.
    replay = SessionReplay(session_file)
    async with httpx.AsyncClient(transport=replay.http_transport()) as client:
        missing = await client.get("https://clob.polymarket.com/book", params={"token_id": "never-recorded"})
    assert missing.status_code == 404 and len(replay.unmatched_requests) == 1
    report = {**(await replay_session(session_file, tmp_path / "ok.jsonl", engine=_adapter()))}
    report["replay"] = {**report["replay"], "unmatched_requests": 1}
    assert "had no recorded answer" in format_replay(report)


@pytest.mark.asyncio
async def test_recording_transport_keeps_the_body_and_drops_wire_encoding(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-encoding": "gzip"}, content=gzip.compress(b'{"ok": true}'))

    recorder = SessionRecorder(tmp_path / "http.jsonl")
    async with httpx.AsyncClient(transport=RecordingTransport(httpx.MockTransport(handler), recorder)) as client:
        response = await client.get("https://example.test/book", params={"token_id": "abc"})
    assert response.json() == {"ok": True}
    assert "content-encoding" not in response.headers

    records = [json.loads(line) for line in (tmp_path / "http.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["kind"] == "meta"
    assert records[1]["kind"] == "http" and records[1]["status"] == 200
    assert records[1]["url"] == "https://example.test/book?token_id=abc"
    assert json.loads(records[1]["body"]) == {"ok": True}


@pytest.mark.asyncio
async def test_recorded_news_assessment_is_replayed_at_its_original_age(tmp_path: Path) -> None:
    recorded_at = datetime(2026, 11, 14, 9, 0, 0, tzinfo=timezone.utc)
    assessment = NewsAssessment(
        news_id="n1", source="wire", headline="Fed cuts", published_at=recorded_at - timedelta(seconds=20),
        target_token_id="tok", estimated_fair_prob=0.8, confidence=0.9,
    )
    session_file = tmp_path / "news.jsonl"
    lines = [
        {"t": recorded_at.isoformat(), "kind": "ws_open", "conn": 0},
        {"t": recorded_at.isoformat(), "kind": "assessment", "payload": assessment.model_dump(mode="json")},
        {"t": recorded_at.isoformat(), "kind": "ws_frame", "conn": 0, "data": "PONG"},
    ]
    session_file.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    replay = SessionReplay(session_file)
    received: list[NewsAssessment] = []

    async def handler(item: NewsAssessment) -> None:
        received.append(item)

    replay.news_handler = handler
    async with replay.ws_connector()("wss://ignored") as socket:
        assert await socket.recv() == "PONG"  # the assessment is dispatched first, in recorded order

    assert len(received) == 1 and received[0].headline == "Fed cuts"
    age = (datetime.now(timezone.utc) - received[0].published_at).total_seconds()
    assert 19.0 <= age <= 25.0  # still a 20-second-old headline, not a day-old one
    assert replay.assessments_replayed == 1

    # The recording side: the wrapper writes the assessment, then calls the real handler.
    recorder = SessionRecorder(tmp_path / "rec.jsonl")
    await record_assessments(handler, recorder)(assessment)
    assert recorder.counts["assessment"] == 1 and len(received) == 2


@pytest.mark.asyncio
async def test_stream_uses_the_connector_given_at_construction() -> None:
    opened: list[str] = []
    stop_event = asyncio.Event()

    @asynccontextmanager
    async def connector(url: str) -> AsyncIterator[Any]:
        opened.append(url)
        stop_event.set()

        class _Idle:
            async def send(self, message: str) -> None:
                return None

            async def recv(self) -> str:
                await asyncio.Event().wait()
                return ""

        yield _Idle()

    stream = ClobOrderBookStream(token_ids=[], ws_connect=connector)
    await asyncio.wait_for(stream.run(stop_event=stop_event), timeout=2.0)
    assert opened == [stream.ws_url]


def test_replay_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    session_file, _, _ = asyncio.run(_record(tmp_path))

    assert replay_main(["--session", str(tmp_path / "missing.jsonl")]) == 2
    existing = tmp_path / "existing.jsonl"
    existing.write_text("", encoding="utf-8")
    assert replay_main(["--session", str(session_file), "--log", str(existing)]) == 2
    capsys.readouterr()

    out_log = tmp_path / "cli.jsonl"
    assert replay_main(["--session", str(session_file), "--log", str(out_log)]) == 0
    printed = capsys.readouterr().out
    assert "Polymarket Paper-Trading Report" in printed
    assert "12 frame(s) over 2 connection(s)" in printed
    assert out_log.exists()

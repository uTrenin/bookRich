"""Tests for 7-day operational readiness enhancements across Recinto 1."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from collector.clob_ws_stream import ClobOrderBookStream
from collector.gamma_discovery import GammaDiscoveryClient
from contracts.models import (
    BookLevel,
    MarketGroup,
    MarketOutcome,
    OrderBookSnapshot,
    OrderLeg,
    PaperFill,
    TradeSignal,
)
from paper_trader.report import analyze_paper_log, format_report, main as report_main
from paper_trader.runner import DynamicEngineAdapter, PaperTradingRunner, _run_cli
import paper_trader.runner as runner_mod


def _make_group(event_id: str, yes_token: str, no_token: str) -> MarketGroup:
    return MarketGroup(
        event_id=event_id,
        title=f"Event {event_id}",
        mutually_exclusive=True,
        outcomes=[
            MarketOutcome(
                market_id=f"m_{event_id}",
                question=f"Question {event_id}?",
                outcome_label="Yes",
                token_id=yes_token,
            ),
            MarketOutcome(
                market_id=f"m_{event_id}",
                question=f"Question {event_id}?",
                outcome_label="No",
                token_id=no_token,
            ),
        ],
    )


def _make_book(token_id: str, updated_at: datetime, is_stale: bool = False) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        bids=[BookLevel(price=0.45, size=100.0)],
        asks=[BookLevel(price=0.47, size=100.0)],
        updated_at=updated_at,
        is_stale=is_stale,
    )


@pytest.mark.asyncio
async def test_gamma_discovery_pagination_deduplicates_events() -> None:
    pages_seen: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params["limit"])
        offset = int(request.url.params.get("offset", "0"))
        pages_seen.append((limit, offset))
        if offset == 0:
            return httpx.Response(200, json=[{"id": "e1"}, {"id": "e2"}])
        if offset == 2:
            # Includes duplicate e2 and one new event e3, plus e4
            return httpx.Response(200, json=[{"id": "e2"}, {"id": "e3"}])
        return httpx.Response(200, json=[{"id": "e4"}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = GammaDiscoveryClient(http_client=http_client)
        events = await client.fetch_active_events(limit=4, page_size=2)

    assert [e["id"] for e in events] == ["e1", "e2", "e3", "e4"]
    assert pages_seen == [(2, 0), (2, 2), (1, 4)]


@pytest.mark.asyncio
async def test_ws_receive_timeout_marks_books_stale_and_reconnects() -> None:
    stale_states_on_update: list[bool] = []
    stop_event = asyncio.Event()

    def rest_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "bids": [{"price": "0.45", "size": "50"}],
                "asks": [{"price": "0.47", "size": "50"}],
            },
        )

    class _SilentThenLiveWS:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt
            self.sent: list[str] = []
            self._live_delivered = False

        async def __aenter__(self) -> "_SilentThenLiveWS":
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def send(self, payload: str) -> None:
            self.sent.append(payload)

        async def recv(self) -> str:
            if self.attempt == 1:
                await asyncio.sleep(5.0)
            if not self._live_delivered:
                self._live_delivered = True
                return json.dumps(
                    {
                        "event_type": "book",
                        "asset_id": "tok_z",
                        "bids": [{"price": "0.46", "size": "50"}],
                        "asks": [{"price": "0.48", "size": "50"}],
                    }
                )
            stop_event.set()
            await asyncio.sleep(0.05)
            return "PONG"

    attempts = 0
    seen_stale_snapshot = False

    def ws_connect(_url: str) -> _SilentThenLiveWS:
        nonlocal attempts, seen_stale_snapshot
        attempts += 1
        if attempts == 2:
            snap = stream.get_book("tok_z", now=datetime.now(timezone.utc))
            seen_stale_snapshot = bool(snap and snap.is_stale)
        return _SilentThenLiveWS(attempts)

    async def on_update(snap: OrderBookSnapshot) -> None:
        stale_states_on_update.append(snap.is_stale)

    async with httpx.AsyncClient(transport=httpx.MockTransport(rest_handler)) as http_client:
        stream = ClobOrderBookStream(
            token_ids=["tok_z"],
            ping_interval_sec=5.0,
            stale_threshold_sec=10.0,
            max_backoff_sec=0.01,
            receive_timeout_sec=0.05,
            resync_grace_sec=0.0,
            on_book_update=on_update,
            http_client=http_client,
        )
        await stream.run(stop_event=stop_event, ws_connect=ws_connect)

    assert attempts >= 2
    assert seen_stale_snapshot is True
    assert stale_states_on_update[-1] is False


@pytest.mark.asyncio
async def test_ws_update_tokens_snapshots_new_and_prunes_closed() -> None:
    fetched: list[str] = []

    def rest_handler(request: httpx.Request) -> httpx.Response:
        tid = request.url.params["token_id"]
        fetched.append(tid)
        return httpx.Response(
            200,
            json={
                "bids": [{"price": "0.40", "size": "20"}],
                "asks": [{"price": "0.42", "size": "20"}],
            },
        )

    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, payload: str) -> None:
            self.sent.append(payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(rest_handler)) as http_client:
        stream = ClobOrderBookStream(token_ids=["tok_old", "tok_keep"], http_client=http_client)
        await stream.initialize_books()
        fake_ws = _FakeWS()
        stream._active_ws = fake_ws

        added = await stream.update_tokens(["tok_keep", "tok_new"])

    assert added == ["tok_new"]
    assert stream.token_ids == ["tok_keep", "tok_new"]
    assert stream.get_book("tok_old") is None
    assert stream.get_book("tok_new") is not None
    assert fetched.count("tok_new") == 1
    assert json.loads(fake_ws.sent[-1]) == {"assets_ids": ["tok_keep", "tok_new"], "type": "market"}


def test_engine_adapter_env_buffers_max_book_age_and_release_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEE_BUFFER", "0.01")
    monkeypatch.setenv("SLIPPAGE_BUFFER", "0.005")
    monkeypatch.setenv("MAX_BOOK_AGE_SEC", "15.0")

    adapter = DynamicEngineAdapter()
    assert adapter.fee_buffer == pytest.approx(0.01)
    assert adapter.slippage_buffer == pytest.approx(0.005)
    assert adapter.max_book_age_sec == pytest.approx(15.0)

    now = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
    old_books = {
        "tok_a": _make_book("tok_a", updated_at=now - timedelta(seconds=30.0)),
    }
    sig = TradeSignal(
        signal_id="sig_age_1",
        strategy="COMBINATORIAL_ARB",
        event_id="evt_1",
        legs=[
            OrderLeg(
                token_id="tok_a",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.47,
                max_acceptable_price=0.50,
            )
        ],
        expected_net_edge=0.05,
        total_cost_usdc=4.7,
        created_at=now,
        rationale="test",
    )
    approved, reason = adapter.approve_signal(sig, old_books, now=now)
    assert approved is False
    assert "STALE_BOOK" in reason

    fill = PaperFill(
        signal_id="sig_age_1",
        strategy="COMBINATORIAL_ARB",
        executed_at=now,
        filled_legs=sig.legs,
        actual_simulated_cost_usdc=20.0,
        slippage_vs_best_ask=0.0,
        status="FILLED",
        note="ok",
    )
    adapter.record_fill(sig, fill)
    assert adapter.current_exposure_usdc == pytest.approx(20.0)
    assert adapter.release_event_exposure("evt_1") == pytest.approx(20.0)
    assert adapter.current_exposure_usdc == pytest.approx(0.0)
    assert adapter.release_event_exposure("evt_1") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_runner_daily_summary_rollover_and_market_refresh_settlement(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "paper.jsonl"
    day1 = datetime(2026, 11, 15, 10, 0, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 11, 16, 0, 1, 0, tzinfo=timezone.utc)

    class _DynamicDiscovery:
        def __init__(self) -> None:
            self.calls = 0

        async def discover_market_groups(self, limit: int = 50) -> list[MarketGroup]:
            self.calls += 1
            if self.calls == 1:
                return []  # Simulate transient empty response -> must NOT wipe exposures
            return [_make_group("evt_open", "y_open", "n_open")]

    class _FakeStream:
        def __init__(self) -> None:
            self.updated_to: list[list[str]] = []

        async def update_tokens(self, token_ids: list[str]) -> list[str]:
            self.updated_to.append(list(token_ids))
            return []

    adapter = DynamicEngineAdapter()
    sig = TradeSignal(
        signal_id="s1",
        strategy="COMBINATORIAL_ARB",
        event_id="evt_closed",
        legs=[
            OrderLeg(
                token_id="y_closed",
                side="BUY",
                target_shares=10.0,
                expected_vwap=0.45,
                max_acceptable_price=0.48,
            )
        ],
        expected_net_edge=0.03,
        total_cost_usdc=4.5,
        created_at=day1,
        rationale="test",
    )
    fill = PaperFill(
        signal_id="s1",
        strategy="COMBINATORIAL_ARB",
        executed_at=day1,
        filled_legs=sig.legs,
        actual_simulated_cost_usdc=25.0,
        slippage_vs_best_ask=0.0,
        status="FILLED",
        note="ok",
    )
    adapter.record_fill(sig, fill)

    discovery = _DynamicDiscovery()
    stream = _FakeStream()
    runner = PaperTradingRunner(
        discovery_client=discovery,
        book_stream=stream,
        engine=adapter,
        market_groups=[_make_group("evt_closed", "y_closed", "n_closed")],
        log_path=log_path,
        daily_summary_enabled=True,
    )
    runner._last_summary_utc_date = "2026-11-15"
    runner.trade_logger.log_paper_fill(fill, signal=sig)

    # First rollover emits DAILY_SUMMARY for 2026-11-15; second call same UTC day is no-op
    summary = runner._maybe_emit_daily_summary(now=day2)
    assert summary is not None
    assert summary["date"] == "2026-11-15"
    assert summary["fills_count"] == 1
    assert runner._maybe_emit_daily_summary(now=day2) is None

    # Transient empty refresh keeps existing exposure intact
    toks_empty, settled_empty = await runner.refresh_markets_once(discovery_limit=20)
    assert toks_empty == []
    assert settled_empty == []
    assert adapter.current_exposure_usdc == pytest.approx(25.0)

    # Valid refresh without evt_closed settles evt_closed and updates tokens
    toks_live, settled_live = await runner.refresh_markets_once(discovery_limit=20)
    assert toks_live == ["y_open", "n_open"]
    assert settled_live == ["evt_closed"]
    assert adapter.current_exposure_usdc == pytest.approx(0.0)
    assert stream.updated_to == [["y_open", "n_open"]]

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    event_types = [e["event_type"] for e in entries]
    assert "DAILY_SUMMARY" in event_types
    assert "PAPER_SETTLED" in event_types


@pytest.mark.asyncio
async def test_run_cli_wires_polymarket_env_urls_and_timers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured["init"] = kwargs

        async def run(self, stop_event: asyncio.Event | None = None, discovery_limit: int = 50) -> None:
            captured["run"] = {"discovery_limit": discovery_limit, "has_stop": stop_event is not None}

    monkeypatch.setattr(runner_mod, "PaperTradingRunner", _FakeRunner)
    monkeypatch.setenv("LOG_FILE_PATH", str(tmp_path / "cli.jsonl"))
    monkeypatch.setenv("POLYMARKET_GAMMA_URL", "https://gamma.example.test")
    monkeypatch.setenv("POLYMARKET_CLOB_URL", "https://clob.example.test")
    monkeypatch.setenv("POLYMARKET_WS_URL", "wss://ws.example.test/market")
    monkeypatch.setenv("WS_RECEIVE_TIMEOUT_SEC", "45.0")
    monkeypatch.setenv("MARKET_REFRESH_INTERVAL_SEC", "900.0")
    monkeypatch.setenv("DAILY_SUMMARY_ENABLED", "false")
    monkeypatch.setenv("DISCOVERY_LIMIT", "75")

    await _run_cli()

    assert captured["init"]["gamma_url"] == "https://gamma.example.test"
    assert captured["init"]["clob_rest_url"] == "https://clob.example.test"
    assert captured["init"]["clob_ws_url"] == "wss://ws.example.test/market"
    assert captured["init"]["ws_receive_timeout_sec"] == pytest.approx(45.0)
    assert captured["init"]["market_refresh_interval_sec"] == pytest.approx(900.0)
    assert captured["init"]["daily_summary_enabled"] is False
    assert captured["run"]["discovery_limit"] == 75


def test_day7_report_analyzer_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = analyze_paper_log(tmp_path / "missing.jsonl")
    assert missing["fills_count"] == 0
    assert missing["execution_rate"] == 0.0

    log_path = tmp_path / "week.jsonl"
    rows = [
        {
            "ts": "2026-11-15T00:00:00+00:00",
            "event_type": "PAPER_FILL",
            "payload": {
                "strategy": "COMBINATORIAL_ARB",
                "actual_simulated_cost_usdc": 24.0,
                "slippage_vs_best_ask": 0.01,
                "expected_net_edge": 0.03,
            },
        },
        {
            "ts": "2026-11-15T12:00:00+00:00",
            "event_type": "PAPER_FILL",
            "payload": {
                "strategy": "NEWS_LATENCY",
                "actual_simulated_cost_usdc": 30.0,
                "slippage_vs_best_ask": 0.03,
                "expected_net_edge": 0.07,
                "latency_ms": 1200.0,
            },
        },
        {
            "ts": "2026-11-16T00:00:00+00:00",
            "event_type": "SIGNAL_REJECTED",
            "payload": {
                "reason": "EVENT_EXPOSURE_EXCEEDED: cap reached",
                "latency_ms": 800.0,
            },
        },
        {
            "ts": "2026-11-16T00:05:00+00:00",
            "event_type": "PAPER_SETTLED",
            "payload": {"event_id": "e1", "released_exposure_usdc": 24.0},
        },
        {
            "ts": "2026-11-16T00:05:00+00:00",
            "event_type": "DAILY_SUMMARY",
            "payload": {"date": "2026-11-15"},
        },
    ]
    log_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n{corrupt_json_line\n",
        encoding="utf-8",
    )

    rep = analyze_paper_log(log_path)
    assert rep["total_entries"] == 5
    assert rep["malformed_lines"] == 1
    assert rep["span_hours"] == pytest.approx(24.0 + 5.0 / 60.0)
    assert rep["fills_count"] == 2
    assert rep["rejections_count"] == 1
    assert rep["execution_rate"] == pytest.approx(2.0 / 3.0)
    assert rep["total_virtual_cost_usdc"] == pytest.approx(54.0)
    assert rep["total_released_exposure_usdc"] == pytest.approx(24.0)
    assert rep["fills_by_strategy"] == {"COMBINATORIAL_ARB": 1, "NEWS_LATENCY": 1}
    assert rep["rejections_by_reason"] == {"EVENT_EXPOSURE_EXCEEDED": 1}
    assert rep["slippage_vs_best_ask"]["p50"] == pytest.approx(0.02)
    assert rep["news_latency_ms"]["samples"] == 2
    assert rep["news_latency_ms"]["avg"] == pytest.approx(1000.0)
    assert "Polymarket Paper-Trading Report" in format_report(rep)

    assert report_main(["--log", str(log_path), "--json"]) == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["fills_count"] == 2
    assert cli_out["daily_summaries_count"] == 1

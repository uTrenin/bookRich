"""Replay a recorded session through the paper bot, offline, and report what it did.

    python -m paper_trader.replay --session logs/session_20261115.jsonl

The bot that runs here is the real one (discovery, stream, engine, simulator, logger);
only the network is replaced by the recording. Trading parameters come from the
environment exactly as in a live run (TAKER_FEE_BPS, MIN_ARB_NET_EDGE, ...), so the same
session can be replayed under different settings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from collector.clob_ws_stream import ClobOrderBookStream
from collector.gamma_discovery import GammaDiscoveryClient
from paper_trader.logger import PaperTradeLogger
from paper_trader.report import analyze_paper_log, format_report
from paper_trader.runner import EngineProtocol, PaperTradingRunner, validate_paper_only_safety
from paper_trader.session_io import SessionReplay


async def build_runner(
    *,
    http_client: httpx.AsyncClient,
    ws_connect: Callable[..., Any],
    log_path: str | Path,
    discovery_limit: int = 50,
    min_liquidity_usdc: float = 0.0,
    engine: EngineProtocol | None = None,
    max_backoff_sec: float = 60.0,
    receive_timeout_sec: float | None = None,
    heartbeat_interval_sec: float = 60.0,
    news_feed_urls: list[str] | None = None,
) -> PaperTradingRunner:
    """Wire a runner on explicit transports.

    Recording and replay both go through this function, so they issue the same requests
    in the same order: one discovery here to learn the tokens, then the runner's own.
    """
    discovery = GammaDiscoveryClient(min_liquidity_usdc=min_liquidity_usdc, http_client=http_client)
    groups = await discovery.discover_market_groups(limit=discovery_limit)
    token_ids = list(dict.fromkeys(outcome.token_id for group in groups for outcome in group.outcomes))
    stream = ClobOrderBookStream(
        token_ids=token_ids,
        http_client=http_client,
        ws_connect=ws_connect,
        max_backoff_sec=max_backoff_sec,
        receive_timeout_sec=receive_timeout_sec,
    )
    return PaperTradingRunner(
        discovery_client=discovery,
        book_stream=stream,
        engine=engine,
        trade_logger=PaperTradeLogger(log_path=log_path),
        heartbeat_interval_sec=heartbeat_interval_sec,
        news_feed_urls=news_feed_urls,
        # A periodic refresh would issue requests the recording cannot line up with.
        market_refresh_interval_sec=0.0,
    )


async def replay_session(
    session_path: str | Path,
    log_path: str | Path,
    *,
    discovery_limit: int | None = None,
    min_liquidity_usdc: float | None = None,
    engine: EngineProtocol | None = None,
    timeout_sec: float = 300.0,
) -> dict[str, Any]:
    """Run the bot on a recorded session and return the log analysis plus replay counters."""
    validate_paper_only_safety()
    replay = SessionReplay(session_path)
    config = replay.config
    limit = int(discovery_limit if discovery_limit is not None else config.get("discovery_limit", 50))
    min_liquidity = float(
        min_liquidity_usdc if min_liquidity_usdc is not None else config.get("min_liquidity_usdc", 0.0)
    )

    async with httpx.AsyncClient(transport=replay.http_transport()) as client:
        try:
            runner = await build_runner(
                http_client=client,
                ws_connect=replay.ws_connector(),
                log_path=log_path,
                discovery_limit=limit,
                min_liquidity_usdc=min_liquidity,
                engine=engine,
                max_backoff_sec=0.01,
                heartbeat_interval_sec=3600.0,
            )
        except httpx.HTTPStatusError as exc:
            # Market discovery asked for something the recording does not contain.
            raise ValueError(
                f"The recording has no recorded answer for {exc.request.url}. It was made with "
                f"discovery_limit={config.get('discovery_limit', 'unknown')}: replay it with the same settings."
            ) from exc
        replay.news_handler = runner.handle_news_assessment
        stop_event = asyncio.Event()
        run_task = asyncio.create_task(runner.run(stop_event=stop_event, discovery_limit=limit))
        done_task = asyncio.create_task(replay.done.wait())
        try:
            await asyncio.wait({run_task, done_task}, timeout=timeout_sec, return_when=asyncio.FIRST_COMPLETED)
        finally:
            done_task.cancel()
            stop_event.set()
            await asyncio.wait_for(run_task, timeout=30.0)

    report = analyze_paper_log(log_path)
    report["replay"] = {
        "session": str(session_path),
        "completed": replay.done.is_set(),
        "records_read": replay.records_read,
        "connections_recorded": replay.connections_recorded,
        "frames_replayed": replay.frames_replayed,
        "assessments_replayed": replay.assessments_replayed,
        "unmatched_requests": len(replay.unmatched_requests),
    }
    return report


def format_replay(report: dict[str, Any]) -> str:
    info = report["replay"]
    lines = [
        format_report(report),
        f"  Replay of {info['session']}: {info['frames_replayed']} frame(s) over "
        f"{info['connections_recorded']} connection(s), {info['assessments_replayed']} news assessment(s)",
    ]
    if not info["completed"]:
        lines.append("  WARNING: the replay stopped before the end of the recording.")
    if info["unmatched_requests"]:
        lines.append(
            f"  WARNING: {info['unmatched_requests']} request(s) had no recorded answer "
            "(different DISCOVERY_LIMIT or MIN_LIQUIDITY_USDC than the recording?)."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded Polymarket session through the paper bot.")
    parser.add_argument("--session", required=True, help="Recording made with `python -m paper_trader.record`.")
    parser.add_argument(
        "--log",
        default=None,
        help="Where to write the replay's trade log (default: logs/replay_<UTC time>.jsonl, always a new file).",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args(argv)

    if not Path(args.session).exists():
        print(f"Session file not found: {args.session}", file=sys.stderr)
        return 2
    log_path = Path(args.log) if args.log else Path("logs") / f"replay_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
    if log_path.exists():
        print(f"Refusing to append to an existing trade log: {log_path}", file=sys.stderr)
        return 2

    try:
        report = asyncio.run(replay_session(args.session, log_path))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else format_replay(report))
    print(f"  Trade log written to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the paper bot for a while and record everything it receives from the network.

    python -m paper_trader.record --minutes 30 --out logs/session_20261115.jsonl

The bot trades on paper as usual while recording. The file can then be replayed offline,
as many times and under as many settings as needed, with `python -m paper_trader.replay`.
It must run on a machine that can reach Polymarket's public endpoints.

Read-only: public GET requests and WebSocket reads only. No order is ever sent.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import websockets

from paper_trader.replay import build_runner
from paper_trader.runner import EngineProtocol, validate_paper_only_safety
from paper_trader.session_io import RecordingTransport, SessionRecorder, record_assessments, recording_connector

logger = logging.getLogger(__name__)


async def record_session(
    out_path: str | Path,
    *,
    minutes: float,
    log_path: str | Path,
    discovery_limit: int = 50,
    min_liquidity_usdc: float = 0.0,
    news_feed_urls: list[str] | None = None,
    engine: EngineProtocol | None = None,
    inner_transport: httpx.AsyncBaseTransport | None = None,
    inner_connector: Callable[..., Any] | None = None,
    stop_event: asyncio.Event | None = None,
) -> dict[str, int]:
    """Record a live session for `minutes` (or until `stop_event`); returns record counts by kind."""
    validate_paper_only_safety()
    recorder = SessionRecorder(out_path)
    recorder.write(
        {"kind": "config", "discovery_limit": int(discovery_limit), "min_liquidity_usdc": float(min_liquidity_usdc)}
    )
    transport = RecordingTransport(inner_transport or httpx.AsyncHTTPTransport(), recorder)
    stop = stop_event if stop_event is not None else asyncio.Event()

    async with httpx.AsyncClient(transport=transport, timeout=10.0) as client:
        runner = await build_runner(
            http_client=client,
            ws_connect=recording_connector(inner_connector or websockets.connect, recorder),
            log_path=log_path,
            discovery_limit=discovery_limit,
            min_liquidity_usdc=min_liquidity_usdc,
            engine=engine,
            receive_timeout_sec=30.0,
            news_feed_urls=news_feed_urls,
        )
        # The classifier's answers are recorded too, so a replay does not need the LLM.
        runner.handle_news_assessment = record_assessments(runner.handle_news_assessment, recorder)  # type: ignore[method-assign]

        run_task = asyncio.create_task(runner.run(stop_event=stop, discovery_limit=discovery_limit))
        stop_task = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait(
                {run_task, stop_task}, timeout=max(0.0, minutes) * 60.0, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            stop_task.cancel()
            stop.set()
            await asyncio.wait_for(run_task, timeout=30.0)

    return dict(recorder.counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a live Polymarket session for later offline replay.")
    parser.add_argument("--minutes", type=float, default=30.0, help="How long to record (default: 30).")
    parser.add_argument(
        "--out", default=None, help="Recording file (default: logs/session_<UTC time>.jsonl)."
    )
    parser.add_argument(
        "--log", default=None, help="Trade log of the live paper run (default: logs/record_<UTC time>.jsonl)."
    )
    args = parser.parse_args(argv)

    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out_path = Path(args.out) if args.out else Path("logs") / f"session_{stamp}.jsonl"
    log_path = Path(args.log) if args.log else Path("logs") / f"record_{stamp}.jsonl"
    if out_path.exists():
        print(f"Refusing to append to an existing recording: {out_path}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    feeds = [item.strip() for item in os.getenv("NEWS_FEED_URLS", "").split(",") if item.strip()]
    try:
        counts = asyncio.run(
            record_session(
                out_path,
                minutes=args.minutes,
                log_path=log_path,
                discovery_limit=int(os.getenv("DISCOVERY_LIMIT", "50") or "50"),
                min_liquidity_usdc=float(os.getenv("MIN_LIQUIDITY_USDC", "0.0") or "0.0"),
                news_feed_urls=feeds,
            )
        )
    except KeyboardInterrupt:
        print(f"Interrupted. Whatever was recorded so far is in {out_path}")
        return 130

    print(f"Recorded to {out_path}: " + ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())))
    print(f"Replay it with: python -m paper_trader.replay --session {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

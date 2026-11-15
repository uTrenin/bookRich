"""Append-only JSONL simulation trade logger and daily summary generator."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from contracts.models import PaperFill, TradeSignal


def _to_utc_datetime(dt: datetime | None) -> datetime:
    """Normalize a datetime (or current time when None) to timezone-aware UTC."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json_default(obj: Any) -> Any:
    """Serialize datetimes, Pydantic models, and Paths for JSONL persistence."""
    if isinstance(obj, datetime):
        return _to_utc_datetime(obj).isoformat()
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class PaperTradeLogger:
    """Thread-safe append-only JSONL logger for 7-day Polymarket paper trading."""

    def __init__(self, log_path: str | Path = "logs/paper_trades.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        ts: datetime | None = None,
    ) -> dict[str, Any]:
        """Append a structured JSONL event entry conforming to docs/plan.md."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        utc_ts = _to_utc_datetime(ts)
        raw_entry = {
            "ts": utc_ts.isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        serialized = json.dumps(raw_entry, ensure_ascii=False, default=_json_default)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
        return json.loads(serialized)

    def log_paper_fill(
        self,
        fill: PaperFill,
        signal: TradeSignal | None = None,
        news_published_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Log a simulated PaperFill (PAPER_FILL when FILLED, else SIGNAL_REJECTED)."""
        payload: dict[str, Any] = fill.model_dump(mode="json")

        if signal is not None:
            payload["event_id"] = signal.event_id
            payload["expected_net_edge"] = signal.expected_net_edge
            payload["signal_total_cost_usdc"] = signal.total_cost_usdc
            payload["rationale"] = signal.rationale

        if news_published_at is not None:
            pub_utc = _to_utc_datetime(news_published_at)
            exec_utc = _to_utc_datetime(fill.executed_at)
            latency_ms = max(0.0, (exec_utc - pub_utc).total_seconds() * 1000.0)
            payload["news_published_at"] = pub_utc.isoformat()
            payload["latency_ms"] = latency_ms

        event_type = "PAPER_FILL" if fill.status == "FILLED" else "SIGNAL_REJECTED"
        return self.log_event(event_type=event_type, payload=payload, ts=fill.executed_at)

    def log_signal_rejected(
        self,
        signal: TradeSignal,
        reason: str,
        ts: datetime | None = None,
    ) -> dict[str, Any]:
        """Log a signal rejected by the risk manager or pre-trade checks."""
        payload: dict[str, Any] = {
            "signal_id": signal.signal_id,
            "strategy": signal.strategy,
            "event_id": signal.event_id,
            "expected_net_edge": signal.expected_net_edge,
            "total_cost_usdc": signal.total_cost_usdc,
            "rationale": signal.rationale,
            "reason": reason,
            "status": "REJECTED",
        }
        return self.log_event(event_type="SIGNAL_REJECTED", payload=payload, ts=ts)

    def log_heartbeat(
        self,
        stats: dict[str, Any],
        ts: datetime | None = None,
    ) -> dict[str, Any]:
        """Log periodic runner health and throughput metrics."""
        return self.log_event(event_type="HEARTBEAT_STATS", payload=dict(stats), ts=ts)

    def generate_daily_summary(
        self,
        date_str: str | None = None,
        write_to_log: bool = True,
    ) -> dict[str, Any]:
        """Aggregate fills, rejections, slippage, virtual cost, and news latency for a UTC date."""
        target_date = date_str or datetime.now(timezone.utc).date().isoformat()

        fills_count = 0
        rejections_count = 0
        heartbeats_count = 0
        total_virtual_cost_usdc = 0.0
        slippage_values: list[float] = []
        latency_values: list[float] = []
        expected_edge_values: list[float] = []
        fills_by_strategy: dict[str, int] = {}

        if self.log_path.exists():
            with self._lock:
                lines = self.log_path.read_text(encoding="utf-8").splitlines()

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                if not isinstance(entry, dict):
                    continue
                ts_raw = entry.get("ts")
                if not isinstance(ts_raw, str):
                    continue
                if not self._matches_utc_date(ts_raw, target_date):
                    continue

                event_type = entry.get("event_type")
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    payload = {}

                if event_type == "DAILY_SUMMARY":
                    continue
                if event_type == "PAPER_FILL":
                    fills_count += 1
                    total_virtual_cost_usdc += float(
                        payload.get("actual_simulated_cost_usdc", 0.0) or 0.0
                    )
                    slippage_values.append(float(payload.get("slippage_vs_best_ask", 0.0) or 0.0))
                    strategy = str(payload.get("strategy", "UNKNOWN"))
                    fills_by_strategy[strategy] = fills_by_strategy.get(strategy, 0) + 1
                    if payload.get("expected_net_edge") is not None:
                        expected_edge_values.append(float(payload["expected_net_edge"]))
                    if payload.get("latency_ms") is not None:
                        latency_values.append(float(payload["latency_ms"]))
                elif event_type == "SIGNAL_REJECTED":
                    rejections_count += 1
                    if payload.get("latency_ms") is not None:
                        latency_values.append(float(payload["latency_ms"]))
                elif event_type == "HEARTBEAT_STATS":
                    heartbeats_count += 1

        avg_slippage = (
            sum(slippage_values) / len(slippage_values) if slippage_values else 0.0
        )
        avg_news_latency_ms = (
            sum(latency_values) / len(latency_values) if latency_values else 0.0
        )
        avg_expected_net_edge = (
            sum(expected_edge_values) / len(expected_edge_values)
            if expected_edge_values
            else 0.0
        )

        summary: dict[str, Any] = {
            "date": target_date,
            "fills_count": fills_count,
            "rejections_count": rejections_count,
            "heartbeats_count": heartbeats_count,
            "total_virtual_cost_usdc": total_virtual_cost_usdc,
            "avg_slippage_vs_best_ask": avg_slippage,
            "avg_news_latency_ms": avg_news_latency_ms,
            "news_latency_samples": len(latency_values),
            "avg_expected_net_edge": avg_expected_net_edge,
            "fills_by_strategy": fills_by_strategy,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if write_to_log:
            self.log_event("DAILY_SUMMARY", summary)

        return summary

    @staticmethod
    def _matches_utc_date(ts_raw: str, target_date: str) -> bool:
        """Check whether an ISO8601 timestamp string falls on target_date in UTC."""
        try:
            parsed = datetime.fromisoformat(ts_raw)
        except ValueError:
            return ts_raw.startswith(target_date)
        return _to_utc_datetime(parsed).date().isoformat() == target_date

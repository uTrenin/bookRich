"""Read-only JSONL analyzer for 7-day Polymarket paper-trading evaluation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_utc_datetime(ts_raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts_raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute percentile (0..100) from a non-empty sorted list using linear interpolation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _summarize_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "avg": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "avg": sum(ordered) / len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 50.0),
        "p95": _percentile(ordered, 95.0),
        "max": ordered[-1],
    }


def analyze_paper_log(log_path: str | Path = "logs/paper_trades.jsonl") -> dict[str, Any]:
    """Aggregate execution, slippage, latency, and risk metrics from a JSONL simulation log."""
    path = Path(log_path)
    fills_count = 0
    rejections_count = 0
    settled_count = 0
    heartbeats_count = 0
    daily_summaries_count = 0
    malformed_lines = 0
    total_entries = 0

    total_virtual_cost_usdc = 0.0
    total_released_exposure_usdc = 0.0
    fills_by_strategy: dict[str, int] = {}
    rejections_by_reason: dict[str, int] = {}
    slippage_values: list[float] = []
    latency_values: list[float] = []
    edge_values: list[float] = []
    first_dt: datetime | None = None
    last_dt: datetime | None = None
    latest_heartbeat: dict[str, Any] | None = None

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue

                if not isinstance(entry, dict):
                    malformed_lines += 1
                    continue

                total_entries += 1
                ts_raw = entry.get("ts")
                if isinstance(ts_raw, str):
                    parsed_dt = _to_utc_datetime(ts_raw)
                    if parsed_dt is not None:
                        if first_dt is None or parsed_dt < first_dt:
                            first_dt = parsed_dt
                        if last_dt is None or parsed_dt > last_dt:
                            last_dt = parsed_dt

                event_type = str(entry.get("event_type") or "")
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    payload = {}

                if event_type == "PAPER_FILL":
                    fills_count += 1
                    strategy = str(payload.get("strategy") or "UNKNOWN")
                    fills_by_strategy[strategy] = fills_by_strategy.get(strategy, 0) + 1

                    cost = _finite_float(payload.get("actual_simulated_cost_usdc"))
                    if cost is not None:
                        total_virtual_cost_usdc += cost
                    slippage = _finite_float(payload.get("slippage_vs_best_ask"))
                    if slippage is not None:
                        slippage_values.append(slippage)
                    edge = _finite_float(payload.get("expected_net_edge"))
                    if edge is not None:
                        edge_values.append(edge)
                    latency = _finite_float(payload.get("latency_ms"))
                    if latency is not None:
                        latency_values.append(latency)

                elif event_type == "SIGNAL_REJECTED":
                    rejections_count += 1
                    raw_reason = str(payload.get("reason") or payload.get("note") or "UNKNOWN")
                    reason_code = raw_reason.split(":", 1)[0].strip() or "UNKNOWN"
                    rejections_by_reason[reason_code] = (
                        rejections_by_reason.get(reason_code, 0) + 1
                    )
                    latency = _finite_float(payload.get("latency_ms"))
                    if latency is not None:
                        latency_values.append(latency)

                elif event_type == "PAPER_SETTLED":
                    settled_count += 1
                    released = _finite_float(payload.get("released_exposure_usdc"))
                    if released is not None:
                        total_released_exposure_usdc += released

                elif event_type == "HEARTBEAT_STATS":
                    heartbeats_count += 1
                    latest_heartbeat = payload

                elif event_type == "DAILY_SUMMARY":
                    daily_summaries_count += 1

    span_hours = 0.0
    if first_dt is not None and last_dt is not None:
        span_hours = max(0.0, (last_dt - first_dt).total_seconds() / 3600.0)

    total_evaluated = fills_count + rejections_count
    execution_rate = (fills_count / total_evaluated) if total_evaluated > 0 else 0.0

    return {
        "log_path": str(path),
        "total_entries": total_entries,
        "malformed_lines": malformed_lines,
        "first_ts": first_dt.isoformat() if first_dt is not None else None,
        "last_ts": last_dt.isoformat() if last_dt is not None else None,
        "span_hours": span_hours,
        "fills_count": fills_count,
        "rejections_count": rejections_count,
        "execution_rate": execution_rate,
        "settled_events_count": settled_count,
        "heartbeats_count": heartbeats_count,
        "daily_summaries_count": daily_summaries_count,
        "total_virtual_cost_usdc": total_virtual_cost_usdc,
        "total_released_exposure_usdc": total_released_exposure_usdc,
        "fills_by_strategy": fills_by_strategy,
        "rejections_by_reason": rejections_by_reason,
        "expected_net_edge": _summarize_distribution(edge_values),
        "slippage_vs_best_ask": _summarize_distribution(slippage_values),
        "news_latency_ms": _summarize_distribution(latency_values),
        "latest_heartbeat": latest_heartbeat,
    }


def format_report(report: dict[str, Any]) -> str:
    """Format an analyze_paper_log() dictionary into a human-readable summary."""
    edge = report["expected_net_edge"]
    slip = report["slippage_vs_best_ask"]
    lat = report["news_latency_ms"]
    lines = [
        f"Polymarket Paper-Trading Report ({report['log_path']})",
        f"  Entries: {report['total_entries']} (malformed: {report['malformed_lines']}) | Span: {report['span_hours']:.2f}h",
        f"  Fills: {report['fills_count']} | Rejections: {report['rejections_count']} | Execution Rate: {report['execution_rate'] * 100.0:.1f}%",
        f"  Settled Events: {report['settled_events_count']} (released {report['total_released_exposure_usdc']:.2f} USDC) | Daily Summaries: {report['daily_summaries_count']}",
        f"  Total Simulated Cost: {report['total_virtual_cost_usdc']:.2f} USDC",
        f"  Fills by Strategy: {report['fills_by_strategy'] or '{}'}",
        f"  Rejections by Reason: {report['rejections_by_reason'] or '{}'}",
        f"  Expected Net Edge ({edge['samples']}): avg={edge['avg']:.4f}, min={edge['min']:.4f}, max={edge['max']:.4f}",
        f"  Slippage vs Best Ask ({slip['samples']}): avg={slip['avg']:.4f}, p50={slip['p50']:.4f}, p95={slip['p95']:.4f}, max={slip['max']:.4f}",
        f"  News Latency ms ({lat['samples']}): avg={lat['avg']:.1f}ms, p50={lat['p50']:.1f}ms, p95={lat['p95']:.1f}ms, max={lat['max']:.1f}ms",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze 7-day Polymarket paper-trading JSONL logs (read-only)."
    )
    parser.add_argument(
        "--log",
        default="logs/paper_trades.jsonl",
        help="Path to the paper trades JSONL file (default: logs/paper_trades.jsonl)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of formatted text.",
    )
    args = parser.parse_args(argv)

    report = analyze_paper_log(args.log)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

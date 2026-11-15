"""Gamma API market discovery client for Polymarket public events.

Read-only public market discovery (Paper Trading phase — no wallet keys or signing).
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx

from contracts.models import MarketGroup, MarketOutcome

DEFAULT_GAMMA_URL = "https://gamma-api.polymarket.com"


def _parse_bool(value: Any, *, default: bool) -> bool:
    """Safely coerce a JSON/API boolean-like value using a fallback default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return default
    return bool(value)


def _safe_float(value: Any) -> float | None:
    """Convert a value to a finite float, returning None if invalid."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_str_list(raw: Any) -> list[str]:
    """Parse Polymarket list fields that may be JSON-encoded strings or Python lists."""
    if raw is None:
        return []
    parsed: Any = raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(item).strip() if item is not None else "" for item in parsed]


def _extract_market_liquidity(market: dict[str, Any]) -> float:
    """Extract numeric liquidity/volume safely from Gamma market fields."""
    for key in ("liquidityNum", "liquidity", "volumeNum", "volume"):
        parsed = _safe_float(market.get(key))
        if parsed is not None and parsed >= 0.0:
            return parsed
    return 0.0


def _select_yes_token_id(token_ids: list[str], outcomes: list[str]) -> str | None:
    """Select the YES outcome token_id, falling back to the first valid token_id."""
    for idx, label in enumerate(outcomes):
        if label.strip().lower() == "yes" and idx < len(token_ids) and token_ids[idx]:
            return token_ids[idx]
    for token_id in token_ids:
        if token_id:
            return token_id
    return None


class GammaDiscoveryClient:
    """Async read-only client for discovering Polymarket MarketGroup objects via Gamma API."""

    def __init__(
        self,
        base_url: str = DEFAULT_GAMMA_URL,
        min_liquidity_usdc: float = 0.0,
        timeout: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_liquidity_usdc = float(min_liquidity_usdc)
        self.timeout = float(timeout)
        self._http_client = http_client

    async def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        """Execute a GET request reusing injected AsyncClient when present."""
        if self._http_client is not None:
            response = await self._http_client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _extract_event_rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        return []

    async def fetch_active_events(
        self,
        limit: int = 100,
        offset: int = 0,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch active, non-closed events from Gamma API with optional pagination."""
        url = f"{self.base_url}/events"
        target_limit = max(1, int(limit))
        current_offset = max(0, int(offset))
        effective_page_size = max(1, int(page_size))

        if target_limit <= effective_page_size and current_offset == 0:
            params: dict[str, Any] = {
                "active": "true",
                "closed": "false",
                "limit": target_limit,
            }
            return self._extract_event_rows(await self._get_json(url, params))

        collected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        while len(collected) < target_limit:
            batch_limit = min(effective_page_size, target_limit - len(collected))
            params = {
                "active": "true",
                "closed": "false",
                "limit": batch_limit,
                "offset": current_offset,
            }
            rows = self._extract_event_rows(await self._get_json(url, params))
            if not rows:
                break
            for row in rows:
                event_id = str(row.get("id") or row.get("slug") or "").strip()
                if event_id:
                    if event_id in seen_ids:
                        continue
                    seen_ids.add(event_id)
                collected.append(row)
                if len(collected) >= target_limit:
                    break
            if len(rows) < batch_limit:
                break
            current_offset += len(rows)

        return collected

    def parse_market_groups(self, events_payload: list[dict[str, Any]]) -> list[MarketGroup]:
        """Parse raw Gamma event payloads into validated MarketGroup contracts."""
        groups: list[MarketGroup] = []

        for event in events_payload:
            if not isinstance(event, dict):
                continue
            if not _parse_bool(event.get("active"), default=True):
                continue
            if _parse_bool(event.get("closed"), default=False):
                continue
            if _parse_bool(event.get("archived"), default=False):
                continue

            raw_markets = event.get("markets")
            if not isinstance(raw_markets, list) or not raw_markets:
                continue

            event_id = str(event.get("id") or event.get("slug") or "").strip()
            title = str(event.get("title") or event.get("slug") or event_id).strip()
            is_neg_risk = _parse_bool(event.get("negRisk"), default=False) or _parse_bool(
                event.get("enableNegRisk"), default=False
            )
            valid_raw_markets = [market for market in raw_markets if isinstance(market, dict)]
            if not valid_raw_markets:
                continue
            is_multi_market = len(valid_raw_markets) > 1 or is_neg_risk

            active_markets: list[dict[str, Any]] = []
            for market in valid_raw_markets:
                if not _parse_bool(market.get("active"), default=True):
                    continue
                if _parse_bool(market.get("closed"), default=False):
                    continue
                if _parse_bool(market.get("archived"), default=False):
                    continue
                if _extract_market_liquidity(market) < self.min_liquidity_usdc:
                    continue
                active_markets.append(market)

            if not active_markets:
                continue

            outcomes: list[MarketOutcome] = []
            if is_multi_market:
                for market in active_markets:
                    token_ids = _parse_str_list(market.get("clobTokenIds"))
                    raw_outcomes = _parse_str_list(market.get("outcomes"))
                    yes_token_id = _select_yes_token_id(token_ids, raw_outcomes)
                    if not yes_token_id:
                        continue
                    market_id = str(market.get("id") or market.get("conditionId") or "").strip()
                    question = str(market.get("question") or title or market_id).strip()
                    group_item_title = str(market.get("groupItemTitle") or "").strip()
                    outcome_label = group_item_title or question
                    outcomes.append(
                        MarketOutcome(
                            market_id=market_id,
                            question=question,
                            outcome_label=outcome_label,
                            token_id=yes_token_id,
                        )
                    )
                # Sum-to-one arb requires an exhaustive negRisk group with zero filtered sub-markets.
                is_complete_group = len(outcomes) == len(valid_raw_markets)
                mutually_exclusive = bool(is_neg_risk and is_complete_group and len(outcomes) > 1)
            else:
                market = active_markets[0]
                token_ids = _parse_str_list(market.get("clobTokenIds"))
                raw_outcomes = _parse_str_list(market.get("outcomes"))
                market_id = str(market.get("id") or market.get("conditionId") or "").strip()
                question = str(market.get("question") or title or market_id).strip()
                default_labels = ["Yes", "No"] if len(token_ids) == 2 else []
                for idx, token_id in enumerate(token_ids):
                    if not token_id:
                        continue
                    if idx < len(raw_outcomes) and raw_outcomes[idx]:
                        label = raw_outcomes[idx]
                    elif idx < len(default_labels):
                        label = default_labels[idx]
                    else:
                        label = f"Outcome {idx + 1}"
                    outcomes.append(
                        MarketOutcome(
                            market_id=market_id,
                            question=question,
                            outcome_label=label,
                            token_id=token_id,
                        )
                    )
                mutually_exclusive = True

            if len(outcomes) < 2:
                continue

            if not event_id:
                event_id = outcomes[0].market_id
            if not title:
                title = outcomes[0].question

            groups.append(
                MarketGroup(
                    event_id=event_id,
                    title=title,
                    mutually_exclusive=mutually_exclusive,
                    outcomes=outcomes,
                )
            )

        return groups

    async def discover_market_groups(self, limit: int = 100) -> list[MarketGroup]:
        """Fetch active Gamma events and parse them into MarketGroup objects."""
        events = await self.fetch_active_events(limit=limit)
        return self.parse_market_groups(events)


async def discover_active_market_groups(
    base_url: str = DEFAULT_GAMMA_URL,
    limit: int = 100,
    min_liquidity_usdc: float = 0.0,
    http_client: httpx.AsyncClient | None = None,
) -> list[MarketGroup]:
    """Convenience helper to discover active MarketGroup instances in one call."""
    client = GammaDiscoveryClient(
        base_url=base_url,
        min_liquidity_usdc=min_liquidity_usdc,
        http_client=http_client,
    )
    return await client.discover_market_groups(limit=limit)

"""Async news feed collector and fast Gemini Flash classifier for news latency trading."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import inspect
import json
import logging
import math
import os
import re
from typing import Any
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel

from contracts.models import MarketGroup, MarketOutcome, NewsAssessment

logger = logging.getLogger(__name__)

PLACEHOLDER_GEMINI_API_KEY = "your_gemini_api_key_here"
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "will", "that", "this", "from", "are", "was", "has",
        "have", "not", "but", "its", "after", "before", "over", "into", "than", "who",
        "what", "when", "which", "yes", "new", "says", "said",
    }
)


def _keywords(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of 3+ characters, minus common stopwords."""
    return {
        word
        for word in _WORD_RE.findall(text.lower())
        if len(word) >= 3 and word not in _STOPWORDS
    }


class RawNewsItem(BaseModel):
    """Unclassified news item parsed from an RSS/Atom or external feed."""

    news_id: str
    source: str
    headline: str
    summary: str = ""
    published_at: datetime
    url: str = ""


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a naive or timezone-aware datetime to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _local_name(tag: str) -> str:
    """Return lowercase XML local element name without namespace URI."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1].lower()
    return tag.lower()


def _element_text(element: ET.Element) -> str:
    """Extract stripped text content (including nested inline tags) from an XML node."""
    return "".join(element.itertext()).strip()


def _find_child_text(parent: ET.Element, *names: str) -> str:
    """Return text of the first direct child matching any local element name."""
    target_names = {name.lower() for name in names}
    for child in parent:
        if _local_name(child.tag) in target_names:
            text = _element_text(child)
            if text:
                return text
    return ""


def _extract_atom_link(entry: ET.Element) -> str:
    """Extract preferred link URL from Atom <link> elements (href attr or text)."""
    fallback_href = ""
    fallback_text = ""
    for child in entry:
        if _local_name(child.tag) != "link":
            continue
        href = (child.attrib.get("href") or "").strip()
        rel = (child.attrib.get("rel") or "").strip().lower()
        if href:
            if rel in ("", "alternate"):
                return href
            if not fallback_href:
                fallback_href = href
        text = _element_text(child)
        if text and not fallback_text:
            fallback_text = text
    return fallback_href or fallback_text


def _parse_datetime_utc(raw: str) -> datetime | None:
    """Parse RFC 2822 or ISO-8601 timestamps into timezone-aware UTC datetimes."""
    value = raw.strip()
    if not value:
        return None

    iso_candidate = value[:-1] + "+00:00" if value[-1] in ("Z", "z") else value
    try:
        dt = datetime.fromisoformat(iso_candidate)
        return _ensure_utc(dt)
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(value)
        return _ensure_utc(dt)
    except (TypeError, ValueError, IndexError):
        return None


def _strip_json_fences(raw_text: str) -> str:
    """Remove optional Markdown code block wrappers around JSON payloads."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _clamp_prob(value: Any) -> float | None:
    """Parse a finite float and clamp it to [0.0, 1.0], returning None if invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


class GeminiNewsClassifier:
    """Low-latency async Gemini Flash classifier producing structured NewsAssessment."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        timeout_sec: float = 2.8,
        client: Any | None = None,
        max_prompt_candidates: int = 40,
    ) -> None:
        raw_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        cleaned_key = raw_key.strip() if isinstance(raw_key, str) else ""
        if not cleaned_key or cleaned_key == PLACEHOLDER_GEMINI_API_KEY:
            self.api_key: str | None = None
        else:
            self.api_key = cleaned_key

        self.model_name = model_name
        self.timeout_sec = float(timeout_sec)
        self._client = client
        self.max_prompt_candidates = max(1, int(max_prompt_candidates))

    def _select_candidates(
        self,
        item: RawNewsItem,
        candidates: list[tuple[MarketGroup, MarketOutcome]],
    ) -> list[tuple[MarketGroup, MarketOutcome]]:
        """Bound the prompt size: with many candidates keep only the best keyword matches.

        Sending every outcome of every tracked event makes the prompt grow with the market
        universe and pushes the call past `timeout_sec`. Small candidate sets are sent whole.
        """
        if len(candidates) <= self.max_prompt_candidates:
            return candidates

        news_words = _keywords(f"{item.headline} {item.summary}")
        scored: list[tuple[int, int]] = []
        for index, (group, outcome) in enumerate(candidates):
            candidate_words = _keywords(f"{group.title} {outcome.question} {outcome.outcome_label}")
            overlap = len(news_words & candidate_words)
            if overlap > 0:
                scored.append((overlap, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [candidates[index] for _, index in scored[: self.max_prompt_candidates]]

    @property
    def client(self) -> Any | None:
        """Return the underlying GenAI client instance if initialized or injected."""
        return self._client

    def _get_client(self) -> Any | None:
        """Return injected client or lazily initialize google.genai.Client."""
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            import google.genai as genai

            self._client = genai.Client(api_key=self.api_key)
        except Exception:
            logger.exception("Failed to initialize google.genai.Client")
            return None
        return self._client

    def _build_prompt(
        self,
        item: RawNewsItem,
        candidates: list[tuple[MarketGroup, MarketOutcome]],
    ) -> str:
        """Format a concise prompt with news metadata and candidate CLOB outcomes."""
        candidate_lines = [
            (
                f"- token_id={outcome.token_id} | event={group.title!r} | "
                f"question={outcome.question!r} | outcome={outcome.outcome_label!r}"
            )
            for group, outcome in candidates
        ]
        summary_snippet = item.summary.strip()[:500]
        published_iso = _ensure_utc(item.published_at).isoformat()
        candidates_block = "\n".join(candidate_lines)
        return (
            "Evaluate this breaking news item against candidate Polymarket outcomes.\n"
            "Select the single most directly impacted candidate token_id, or null if none apply.\n"
            f"Source: {item.source}\n"
            f"Published UTC: {published_iso}\n"
            f"Headline: {item.headline}\n"
            f"Summary: {summary_snippet}\n\n"
            "Candidate outcomes:\n"
            f"{candidates_block}\n\n"
            "Return STRICT JSON ONLY with keys: "
            '{"target_token_id": string|null, "estimated_fair_prob": float, "confidence": float}. '
            "Both probabilities must be in [0.0, 1.0]."
        )

    def _build_generation_config(self) -> Any:
        """Build JSON-mode generation config using google.genai.types when available."""
        try:
            from google.genai import types

            return types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        except ImportError:
            return {"response_mime_type": "application/json", "temperature": 0.0}

    async def classify(
        self,
        item: RawNewsItem,
        market_groups: list[MarketGroup],
    ) -> NewsAssessment | None:
        """Classify a news item against candidate MarketGroups within timeout_sec."""
        candidates: list[tuple[MarketGroup, MarketOutcome]] = []
        valid_token_ids: set[str] = set()
        for group in market_groups:
            for outcome in group.outcomes:
                if outcome.token_id and outcome.token_id not in valid_token_ids:
                    valid_token_ids.add(outcome.token_id)
                    candidates.append((group, outcome))

        if not candidates:
            return None

        candidates = self._select_candidates(item, candidates)
        if not candidates:
            logger.debug("No candidate outcome matches news_id=%s; skipping LLM call", item.news_id)
            return None
        valid_token_ids = {outcome.token_id for _, outcome in candidates}

        client = self._get_client()
        if client is None:
            return None

        prompt = self._build_prompt(item, candidates)
        config = self._build_generation_config()

        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                ),
                timeout=self.timeout_sec,
            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                "Gemini classification timed out after %.2fs for news_id=%s",
                self.timeout_sec,
                item.news_id,
            )
            return None
        except Exception:
            logger.exception("Gemini classification failed for news_id=%s", item.news_id)
            return None

        if response is None:
            return None

        if isinstance(response, dict):
            payload: Any = response
        else:
            raw_text = response if isinstance(response, str) else getattr(response, "text", None)
            if not isinstance(raw_text, str) or not raw_text.strip():
                return None
            cleaned_text = _strip_json_fences(raw_text)
            try:
                payload = json.loads(cleaned_text)
            except json.JSONDecodeError:
                logger.warning("Gemini returned invalid JSON for news_id=%s", item.news_id)
                return None

        if not isinstance(payload, dict):
            return None

        target_token_id = payload.get("target_token_id")
        if not isinstance(target_token_id, str) or target_token_id not in valid_token_ids:
            return None

        estimated_fair_prob = _clamp_prob(payload.get("estimated_fair_prob"))
        confidence = _clamp_prob(payload.get("confidence"))
        if estimated_fair_prob is None or confidence is None:
            return None

        return NewsAssessment(
            news_id=item.news_id,
            source=item.source,
            headline=item.headline,
            published_at=_ensure_utc(item.published_at),
            target_token_id=target_token_id,
            estimated_fair_prob=estimated_fair_prob,
            confidence=confidence,
        )


class NewsStreamCollector:
    """Async RSS/Atom polling collector with freshness filtering, deduplication, and AI classification."""

    def __init__(
        self,
        feed_urls: list[str] | None = None,
        classifier: Any | None = None,
        poll_interval_sec: float = 15.0,
        max_age_sec: float = 120.0,
        on_assessment: Callable[[NewsAssessment], Awaitable[None] | None] | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_seen_ids: int = 10_000,
        max_concurrent_classifications: int = 4,
    ) -> None:
        self.feed_urls: list[str] = list(feed_urls) if feed_urls is not None else []
        self.classifier: Any = classifier if classifier is not None else GeminiNewsClassifier()
        self.poll_interval_sec = float(poll_interval_sec)
        self.max_age_sec = float(max_age_sec)
        self.on_assessment = on_assessment
        self.http_client = http_client
        self._max_seen_ids = max(1, int(max_seen_ids))
        self.max_concurrent_classifications = max(1, int(max_concurrent_classifications))
        self._seen_ids: set[str] = set()
        self._seen_order: deque[str] = deque()

    def _remember_news_id(self, news_id: str) -> bool:
        """Record news_id in bounded deduplication cache; return False if already seen."""
        if news_id in self._seen_ids:
            return False
        if len(self._seen_order) >= self._max_seen_ids:
            evicted = self._seen_order.popleft()
            self._seen_ids.discard(evicted)
        self._seen_ids.add(news_id)
        self._seen_order.append(news_id)
        return True

    def parse_rss_xml(self, xml_text: str, source_name: str = "rss") -> list[RawNewsItem]:
        """Parse RSS 2.0 (<item>) and Atom (<entry>) XML feeds into RawNewsItem objects."""
        if not xml_text or not xml_text.strip():
            return []

        try:
            root = ET.fromstring(xml_text.strip())
        except ET.ParseError:
            logger.warning("Failed to parse XML feed from source=%s", source_name)
            return []

        items: list[RawNewsItem] = []
        for elem in root.iter():
            tag_name = _local_name(elem.tag)
            if tag_name not in ("item", "entry"):
                continue

            title = _find_child_text(elem, "title")
            if not title:
                continue

            if tag_name == "item":
                url = _find_child_text(elem, "link")
                raw_id = _find_child_text(elem, "guid", "id")
                summary = _find_child_text(elem, "description", "summary", "encoded")
                date_raw = _find_child_text(elem, "pubdate", "published", "updated", "date")
            else:
                url = _extract_atom_link(elem)
                raw_id = _find_child_text(elem, "id", "guid")
                summary = _find_child_text(elem, "summary", "content", "description")
                date_raw = _find_child_text(elem, "published", "updated", "pubdate")

            news_id = (
                raw_id
                if raw_id
                else hashlib.sha256(f"{source_name}|{title}|{url}".encode("utf-8")).hexdigest()
            )
            published_at = _parse_datetime_utc(date_raw) or _EPOCH_UTC

            items.append(
                RawNewsItem(
                    news_id=news_id,
                    source=source_name,
                    headline=title,
                    summary=summary,
                    published_at=published_at,
                    url=url,
                )
            )
        return items

    def is_fresh(self, item: RawNewsItem, now: datetime | None = None) -> bool:
        """Return True iff item.published_at is within [-60s, max_age_sec] of now (UTC)."""
        current_time = _ensure_utc(now) if now is not None else datetime.now(timezone.utc)
        published_at = _ensure_utc(item.published_at)
        age_sec = (current_time - published_at).total_seconds()
        return -60.0 <= age_sec <= self.max_age_sec

    async def fetch_feed(self, url: str, source_name: str | None = None) -> list[RawNewsItem]:
        """Fetch an RSS/Atom feed asynchronously over HTTP and parse RawNewsItem entries."""
        resolved_source = source_name or urlparse(url).netloc or url
        if self.http_client is not None:
            response = await self.http_client.get(url)
            response.raise_for_status()
            return self.parse_rss_xml(response.text, source_name=resolved_source)

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return self.parse_rss_xml(response.text, source_name=resolved_source)

    async def process_news_item(
        self,
        item: RawNewsItem,
        market_groups: list[MarketGroup],
        now: datetime | None = None,
    ) -> NewsAssessment | None:
        """Deduplicate, enforce freshness, classify with LLM, and invoke on_assessment."""
        if not self._remember_news_id(item.news_id):
            return None

        if not self.is_fresh(item, now=now):
            return None

        if self.classifier is None:
            return None

        try:
            if hasattr(self.classifier, "classify"):
                result = self.classifier.classify(item, market_groups)
            elif callable(self.classifier):
                result = self.classifier(item, market_groups)
            else:
                return None

            assessment = await result if inspect.isawaitable(result) else result
        except Exception:
            logger.exception("Classifier raised an error for news_id=%s", item.news_id)
            return None

        if assessment is None:
            return None

        if self.on_assessment is not None:
            try:
                callback_result = self.on_assessment(assessment)
                if inspect.isawaitable(callback_result):
                    await callback_result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("on_assessment callback raised an error for news_id=%s", item.news_id)

        return assessment

    async def poll_once(
        self,
        market_groups: list[MarketGroup],
        now: datetime | None = None,
    ) -> list[NewsAssessment]:
        """Poll all configured feed URLs once and return produced NewsAssessment entries."""

        async def _fetch(url: str) -> list[RawNewsItem]:
            try:
                return await self.fetch_feed(url)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed fetching news feed url=%s", url, exc_info=True)
                return []

        # One slow or failing feed must not delay the others.
        feeds = await asyncio.gather(*(_fetch(url) for url in self.feed_urls))
        items = [item for feed_items in feeds for item in feed_items]

        # Classify concurrently (bounded): a burst of fresh items must not queue behind
        # each other's LLM round trip. Each callback fires as soon as its item is classified.
        semaphore = asyncio.Semaphore(self.max_concurrent_classifications)

        async def _process(item: RawNewsItem) -> NewsAssessment | None:
            async with semaphore:
                return await self.process_news_item(item, market_groups, now=now)

        results = await asyncio.gather(*(_process(item) for item in items))
        return [assessment for assessment in results if assessment is not None]

    async def run(
        self,
        market_groups_provider: Callable[[], list[MarketGroup] | Awaitable[list[MarketGroup]]]
        | list[MarketGroup],
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Continuously poll feeds until stop_event is set."""
        while stop_event is None or not stop_event.is_set():
            try:
                if callable(market_groups_provider):
                    provided = market_groups_provider()
                    market_groups = (
                        await provided if inspect.isawaitable(provided) else provided
                    )
                else:
                    market_groups = market_groups_provider
                await self.poll_once(market_groups)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in NewsStreamCollector.run iteration")

            if stop_event is not None and stop_event.is_set():
                break

            if stop_event is not None:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_sec)
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    continue
            else:
                await asyncio.sleep(self.poll_interval_sec)

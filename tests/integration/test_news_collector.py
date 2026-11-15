"""Deterministic unit and integration tests for collector.news_stream."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from collector.news_stream import (
    PLACEHOLDER_GEMINI_API_KEY,
    GeminiNewsClassifier,
    NewsStreamCollector,
    RawNewsItem,
)
from contracts.models import MarketGroup, MarketOutcome, NewsAssessment


def _sample_market_groups() -> list[MarketGroup]:
    return [
        MarketGroup(
            event_id="evt-fed-1",
            title="Fed Rate Decision November",
            mutually_exclusive=True,
            outcomes=[
                MarketOutcome(
                    market_id="mkt-cut-25",
                    question="Will the Fed cut rates by 25 bps?",
                    outcome_label="25 bps cut",
                    token_id="tok-cut-25",
                ),
                MarketOutcome(
                    market_id="mkt-hold",
                    question="Will the Fed hold rates steady?",
                    outcome_label="No change",
                    token_id="tok-hold",
                ),
            ],
        )
    ]


def _make_mock_genai_client(handler: Any) -> Any:
    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=handler)
        )
    )


def test_parse_rss_20_and_deterministic_fallback_id() -> None:
    collector = NewsStreamCollector(classifier=lambda *_args: None)
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Macro Wire</title>
        <item>
          <title>Fed signals 25 bps cut &amp; easing</title>
          <link>https://news.example.com/fed-cut</link>
          <guid>guid-fed-1</guid>
          <pubDate>Sat, 14 Nov 2026 18:10:00 -0500</pubDate>
          <description><![CDATA[Officials pointed to cooling inflation.]]></description>
        </item>
        <item>
          <title>Secondary headline without guid</title>
          <link>https://news.example.com/no-guid</link>
          <pubDate>Sat, 14 Nov 2026 23:12:00 GMT</pubDate>
          <description>Brief note.</description>
        </item>
      </channel>
    </rss>"""

    items = collector.parse_rss_xml(rss_xml, source_name="reuters")
    assert len(items) == 2

    first, second = items
    assert first.news_id == "guid-fed-1"
    assert first.source == "reuters"
    assert first.headline == "Fed signals 25 bps cut & easing"
    assert first.summary == "Officials pointed to cooling inflation."
    assert first.url == "https://news.example.com/fed-cut"
    assert first.published_at == datetime(2026, 11, 14, 23, 10, 0, tzinfo=timezone.utc)

    expected_fallback = hashlib.sha256(
        "reuters|Secondary headline without guid|https://news.example.com/no-guid".encode("utf-8")
    ).hexdigest()
    assert second.news_id == expected_fallback
    assert second.published_at == datetime(2026, 11, 14, 23, 12, 0, tzinfo=timezone.utc)


def test_parse_atom_xml_namespaced_and_unnamespaced() -> None:
    collector = NewsStreamCollector(classifier=lambda *_args: None)
    atom_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Central Bank Feed</title>
      <entry>
        <id>atom-entry-1</id>
        <title>Emergency Rate Update</title>
        <published>2026-11-14T23:15:30Z</published>
        <updated>2026-11-14T23:16:00Z</updated>
        <link rel="self" href="https://feed.example.com/self" />
        <link rel="alternate" href="https://feed.example.com/article-1" />
        <summary>Surprise announcement issued.</summary>
      </entry>
      <entry>
        <id>atom-entry-2</id>
        <title>Updated Only Entry</title>
        <updated>2026-11-15T01:00:00+02:00</updated>
        <link>https://feed.example.com/article-2</link>
        <summary>Fallback updated timestamp.</summary>
      </entry>
    </feed>"""

    items = collector.parse_rss_xml(atom_xml, source_name="atom-feed")
    assert len(items) == 2

    assert items[0].news_id == "atom-entry-1"
    assert items[0].url == "https://feed.example.com/article-1"
    assert items[0].published_at == datetime(2026, 11, 14, 23, 15, 30, tzinfo=timezone.utc)
    assert items[0].summary == "Surprise announcement issued."

    assert items[1].news_id == "atom-entry-2"
    assert items[1].url == "https://feed.example.com/article-2"
    assert items[1].published_at == datetime(2026, 11, 14, 23, 0, 0, tzinfo=timezone.utc)


def test_parse_malformed_or_missing_date_xml() -> None:
    collector = NewsStreamCollector(classifier=lambda *_args: None)
    assert collector.parse_rss_xml("") == []
    assert collector.parse_rss_xml("<rss><broken>") == []

    no_date_xml = """<rss><channel><item>
      <title>Undated item</title>
      <guid>undated-1</guid>
    </item></channel></rss>"""
    items = collector.parse_rss_xml(no_date_xml, source_name="test")
    assert len(items) == 1
    assert collector.is_fresh(items[0], now=datetime(2026, 11, 14, 23, 15, 0, tzinfo=timezone.utc)) is False


def test_is_fresh_boundaries_and_naive_normalization() -> None:
    collector = NewsStreamCollector(max_age_sec=120.0, classifier=lambda *_args: None)
    now = datetime(2026, 11, 14, 23, 20, 0, tzinfo=timezone.utc)

    fresh_zero = RawNewsItem(
        news_id="n0",
        source="wire",
        headline="Now",
        published_at=now,
    )
    fresh_boundary = RawNewsItem(
        news_id="n120",
        source="wire",
        headline="At 120s",
        published_at=now - timedelta(seconds=120),
    )
    naive_fresh = RawNewsItem(
        news_id="n_naive",
        source="wire",
        headline="Naive UTC",
        published_at=datetime(2026, 11, 14, 23, 19, 0),
    )
    stale = RawNewsItem(
        news_id="n121",
        source="wire",
        headline="Too old",
        published_at=now - timedelta(seconds=120.1),
    )
    skew_ahead = RawNewsItem(
        news_id="n_skew",
        source="wire",
        headline="Slight clock skew",
        published_at=now + timedelta(seconds=10),
    )
    far_future = RawNewsItem(
        news_id="n_future",
        source="wire",
        headline="Invalid future",
        published_at=now + timedelta(seconds=3600),
    )

    assert collector.is_fresh(fresh_zero, now=now) is True
    assert collector.is_fresh(fresh_boundary, now=now) is True
    assert collector.is_fresh(naive_fresh, now=now) is True
    assert collector.is_fresh(stale, now=now) is False
    assert collector.is_fresh(skew_ahead, now=now) is True
    assert collector.is_fresh(far_future, now=now) is False


@pytest.mark.asyncio
async def test_deduplication_and_stale_filtering() -> None:
    calls: list[str] = []
    now = datetime(2026, 11, 14, 23, 20, 0, tzinfo=timezone.utc)
    groups = _sample_market_groups()

    async def fake_classifier(item: RawNewsItem, _groups: list[MarketGroup]) -> NewsAssessment:
        calls.append(item.news_id)
        return NewsAssessment(
            news_id=item.news_id,
            source=item.source,
            headline=item.headline,
            published_at=item.published_at,
            target_token_id="tok-cut-25",
            estimated_fair_prob=0.82,
            confidence=0.9,
        )

    collector = NewsStreamCollector(classifier=fake_classifier, max_age_sec=120.0, max_seen_ids=2)

    fresh_item = RawNewsItem(
        news_id="dup-1",
        source="wire",
        headline="Cut confirmed",
        published_at=now - timedelta(seconds=15),
    )
    stale_item = RawNewsItem(
        news_id="stale-1",
        source="wire",
        headline="Old rumor",
        published_at=now - timedelta(seconds=180),
    )

    first_res = await collector.process_news_item(fresh_item, groups, now=now)
    second_res = await collector.process_news_item(fresh_item, groups, now=now)
    stale_res = await collector.process_news_item(stale_item, groups, now=now)

    assert first_res is not None
    assert first_res.news_id == "dup-1"
    assert second_res is None
    assert stale_res is None
    assert calls == ["dup-1"]

    # Evict dup-1 by inserting two more distinct IDs when max_seen_ids=2
    await collector.process_news_item(
        fresh_item.model_copy(update={"news_id": "other-2"}), groups, now=now
    )
    reprocessed = await collector.process_news_item(fresh_item, groups, now=now)
    assert reprocessed is not None
    assert calls == ["dup-1", "other-2", "dup-1"]


@pytest.mark.asyncio
async def test_gemini_classifier_valid_json_and_markdown_fences() -> None:
    captured_calls: list[dict[str, Any]] = []
    now = datetime(2026, 11, 14, 23, 20, 0, tzinfo=timezone.utc)
    item = RawNewsItem(
        news_id="news-1",
        source="reuters",
        headline="Fed Chair hints at 25 bps rate cut",
        summary="Remarks delivered ahead of FOMC meeting.",
        published_at=now,
    )
    groups = _sample_market_groups()

    async def fake_generate_content(**kwargs: Any) -> Any:
        captured_calls.append(kwargs)
        return SimpleNamespace(
            text='```json\n{"target_token_id": "tok-cut-25", "estimated_fair_prob": 0.78, "confidence": 0.91}\n```'
        )

    classifier = GeminiNewsClassifier(
        api_key="test-key",
        model_name="gemini-2.5-flash",
        timeout_sec=2.5,
        client=_make_mock_genai_client(fake_generate_content),
    )

    result = await classifier.classify(item, groups)
    assert result is not None
    assert result.news_id == "news-1"
    assert result.target_token_id == "tok-cut-25"
    assert result.estimated_fair_prob == pytest.approx(0.78)
    assert result.confidence == pytest.approx(0.91)

    assert len(captured_calls) == 1
    assert captured_calls[0]["model"] == "gemini-2.5-flash"
    prompt = captured_calls[0]["contents"]
    assert "tok-cut-25" in prompt
    assert "tok-hold" in prompt
    assert "Fed Rate Decision November" in prompt
    assert "25 bps cut" in prompt


@pytest.mark.asyncio
async def test_gemini_classifier_validation_clamping_and_edge_cases() -> None:
    now = datetime(2026, 11, 14, 23, 20, 0, tzinfo=timezone.utc)
    item = RawNewsItem(news_id="news-2", source="ap", headline="Headline", published_at=now)
    groups = _sample_market_groups()

    # 1. Unknown token_id -> None
    async def unknown_tok(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            text=json.dumps({"target_token_id": "tok-unknown", "estimated_fair_prob": 0.6, "confidence": 0.8})
        )

    assert await GeminiNewsClassifier(client=_make_mock_genai_client(unknown_tok)).classify(item, groups) is None

    # 2. Null target -> None
    async def null_target(**_kwargs: Any) -> Any:
        return SimpleNamespace(text='{"target_token_id": null, "estimated_fair_prob": 0.0, "confidence": 0.0}')

    assert await GeminiNewsClassifier(client=_make_mock_genai_client(null_target)).classify(item, groups) is None

    # 3. Out-of-range probabilities clamped into [0.0, 1.0]
    async def out_of_range(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            text=json.dumps({"target_token_id": "tok-hold", "estimated_fair_prob": 1.45, "confidence": -0.3})
        )

    clamped = await GeminiNewsClassifier(client=_make_mock_genai_client(out_of_range)).classify(item, groups)
    assert clamped is not None
    assert clamped.target_token_id == "tok-hold"
    assert clamped.estimated_fair_prob == 1.0
    assert clamped.confidence == 0.0

    # 4. Non-finite NaN rejected cleanly
    async def nan_prob(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            text='{"target_token_id": "tok-hold", "estimated_fair_prob": NaN, "confidence": 0.5}'
        )

    assert await GeminiNewsClassifier(client=_make_mock_genai_client(nan_prob)).classify(item, groups) is None


@pytest.mark.asyncio
async def test_gemini_classifier_timeout_json_error_and_sdk_exception() -> None:
    now = datetime(2026, 11, 14, 23, 20, 0, tzinfo=timezone.utc)
    item = RawNewsItem(news_id="news-3", source="ap", headline="Headline", published_at=now)
    groups = _sample_market_groups()

    async def slow_call(**_kwargs: Any) -> Any:
        await asyncio.sleep(0.2)
        return SimpleNamespace(
            text='{"target_token_id": "tok-cut-25", "estimated_fair_prob": 0.5, "confidence": 0.5}'
        )

    timeout_classifier = GeminiNewsClassifier(
        timeout_sec=0.02,
        client=_make_mock_genai_client(slow_call),
    )
    assert await timeout_classifier.classify(item, groups) is None

    async def bad_json(**_kwargs: Any) -> Any:
        return SimpleNamespace(text="not valid json {{{")

    assert await GeminiNewsClassifier(client=_make_mock_genai_client(bad_json)).classify(item, groups) is None

    async def sdk_boom(**_kwargs: Any) -> Any:
        raise RuntimeError("API transient failure")

    assert await GeminiNewsClassifier(client=_make_mock_genai_client(sdk_boom)).classify(item, groups) is None


@pytest.mark.asyncio
async def test_gemini_classifier_missing_or_placeholder_key_empty_groups_and_lazy_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 11, 14, 23, 20, 0, tzinfo=timezone.utc)
    item = RawNewsItem(news_id="news-4", source="ap", headline="Headline", published_at=now)
    groups = _sample_market_groups()

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert await GeminiNewsClassifier().classify(item, groups) is None

    monkeypatch.setenv("GEMINI_API_KEY", PLACEHOLDER_GEMINI_API_KEY)
    assert await GeminiNewsClassifier().classify(item, groups) is None
    assert await GeminiNewsClassifier(api_key=PLACEHOLDER_GEMINI_API_KEY).classify(item, groups) is None

    # Empty outcomes returns None before touching client
    called = False

    async def should_not_run(**_kwargs: Any) -> Any:
        nonlocal called
        called = True
        return SimpleNamespace(text="{}")

    empty_group = MarketGroup(event_id="e0", title="Empty", mutually_exclusive=True, outcomes=[])
    res_empty = await GeminiNewsClassifier(client=_make_mock_genai_client(should_not_run)).classify(
        item, [empty_group]
    )
    assert res_empty is None
    assert called is False

    # Lazy initialization of google.genai.Client
    init_keys: list[str] = []

    async def lazy_generate(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            text='{"target_token_id": "tok-cut-25", "estimated_fair_prob": 0.7, "confidence": 0.85}'
        )

    def fake_client_factory(*, api_key: str) -> Any:
        init_keys.append(api_key)
        return _make_mock_genai_client(lazy_generate)

    import google.genai as genai

    monkeypatch.setattr(genai, "Client", fake_client_factory)

    lazy_classifier = GeminiNewsClassifier(api_key="real-test-key")
    assert lazy_classifier.client is None
    assert init_keys == []

    assessment = await lazy_classifier.classify(item, groups)
    assert assessment is not None
    assert assessment.target_token_id == "tok-cut-25"
    assert init_keys == ["real-test-key"]
    assert lazy_classifier.client is not None


@pytest.mark.asyncio
async def test_poll_once_with_mock_transport_and_callbacks() -> None:
    now = datetime(2026, 11, 14, 23, 20, 0, tzinfo=timezone.utc)
    fresh_rfc2822 = "Sat, 14 Nov 2026 23:19:30 GMT"
    stale_rfc2822 = "Sat, 14 Nov 2026 23:10:00 GMT"
    fresh_iso = "2026-11-14T23:19:45Z"

    rss_payload = f"""<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item>
        <guid>rss-fresh-1</guid>
        <title>Fed cuts 25 bps</title>
        <link>https://rss.example.com/1</link>
        <pubDate>{fresh_rfc2822}</pubDate>
        <description>Fresh item</description>
      </item>
      <item>
        <guid>rss-stale-1</guid>
        <title>Old news</title>
        <link>https://rss.example.com/old</link>
        <pubDate>{stale_rfc2822}</pubDate>
      </item>
    </channel></rss>"""

    atom_payload = f"""<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>atom-fresh-1</id>
        <title>Fed holds no more</title>
        <published>{fresh_iso}</published>
        <link href="https://atom.example.com/1" />
        <summary>Atom item</summary>
      </entry>
    </feed>"""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://rss.example.com/feed.xml":
            return httpx.Response(200, text=rss_payload)
        if str(request.url) == "https://atom.example.com/feed.xml":
            return httpx.Response(200, text=atom_payload)
        return httpx.Response(500, text="upstream error")

    transport = httpx.MockTransport(handler)
    groups = _sample_market_groups()

    async def fake_generate(**kwargs: Any) -> Any:
        if "Fed cuts 25 bps" in kwargs["contents"]:
            return SimpleNamespace(
                text='{"target_token_id": "tok-cut-25", "estimated_fair_prob": 0.9, "confidence": 0.95}'
            )
        return SimpleNamespace(
            text='{"target_token_id": "tok-hold", "estimated_fair_prob": 0.15, "confidence": 0.8}'
        )

    classifier = GeminiNewsClassifier(client=_make_mock_genai_client(fake_generate))
    async_received: list[NewsAssessment] = []
    sync_received: list[NewsAssessment] = []

    async def async_cb(assessment: NewsAssessment) -> None:
        async_received.append(assessment)

    async with httpx.AsyncClient(transport=transport) as http_client:
        collector = NewsStreamCollector(
            feed_urls=[
                "https://broken.example.com/500.xml",
                "https://rss.example.com/feed.xml",
                "https://atom.example.com/feed.xml",
            ],
            classifier=classifier,
            max_age_sec=120.0,
            on_assessment=async_cb,
            http_client=http_client,
        )

        assessments = await collector.poll_once(groups, now=now)
        assert len(assessments) == 2
        assert [a.news_id for a in assessments] == ["rss-fresh-1", "atom-fresh-1"]
        assert [a.target_token_id for a in async_received] == ["tok-cut-25", "tok-hold"]

        # Polling again immediately deduplicates all previously seen items
        second_poll = await collector.poll_once(groups, now=now)
        assert second_poll == []

        # Verify sync callback support as well
        sync_collector = NewsStreamCollector(
            feed_urls=["https://rss.example.com/feed.xml"],
            classifier=classifier,
            max_age_sec=120.0,
            on_assessment=sync_received.append,
            http_client=http_client,
        )
        sync_assessments = await sync_collector.poll_once(groups, now=now)
        assert len(sync_assessments) == 1
        assert len(sync_received) == 1


@pytest.mark.asyncio
async def test_collector_run_exits_cleanly_on_stop_event() -> None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    atom_payload = f"""<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>live-entry-1</id>
        <title>Breaking Fed announcement</title>
        <published>{now_iso}</published>
        <link href="https://feed.example.com/live-1" />
      </entry>
    </feed>"""

    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=atom_payload))
    stop_event = asyncio.Event()
    received: list[NewsAssessment] = []

    async def on_assessment(assessment: NewsAssessment) -> None:
        received.append(assessment)
        stop_event.set()

    async def fake_generate(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            text='{"target_token_id": "tok-cut-25", "estimated_fair_prob": 0.85, "confidence": 0.9}'
        )

    async with httpx.AsyncClient(transport=transport) as http_client:
        collector = NewsStreamCollector(
            feed_urls=["https://feed.example.com/atom.xml"],
            classifier=GeminiNewsClassifier(client=_make_mock_genai_client(fake_generate)),
            poll_interval_sec=10.0,
            max_age_sec=120.0,
            on_assessment=on_assessment,
            http_client=http_client,
        )
        await asyncio.wait_for(
            collector.run(_sample_market_groups, stop_event=stop_event),
            timeout=1.5,
        )

    assert len(received) == 1
    assert received[0].news_id == "live-entry-1"

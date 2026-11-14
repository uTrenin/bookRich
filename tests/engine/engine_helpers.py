"""Builder deterministici condivisi dai test di `/engine`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.models import (
    BookLevel,
    MarketGroup,
    MarketOutcome,
    NewsAssessment,
    OrderBookSnapshot,
)

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def levels(*pairs: tuple[float, float]) -> list[BookLevel]:
    return [BookLevel(price=price, size=size) for price, size in pairs]


def raw_level(price: object, size: object) -> BookLevel:
    """Livello NON validato, per simulare dati corrotti (NaN, None, size <= 0)."""
    return BookLevel.model_construct(price=price, size=size)


def make_book(
    token_id: str,
    asks: list[BookLevel] | None = None,
    bids: list[BookLevel] | None = None,
    *,
    is_stale: bool = False,
    updated_at: datetime = T0,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        bids=bids or [],
        asks=asks or [],
        updated_at=updated_at,
        is_stale=is_stale,
    )


def make_group(*token_ids: str, mutually_exclusive: bool = True, event_id: str = "evt-1") -> MarketGroup:
    return MarketGroup(
        event_id=event_id,
        title="Evento di test",
        mutually_exclusive=mutually_exclusive,
        outcomes=[
            MarketOutcome(
                market_id=f"mkt-{token_id}",
                question="Chi vince?",
                outcome_label=token_id.upper(),
                token_id=token_id,
            )
            for token_id in token_ids
        ],
    )


def make_news(
    *,
    fair: float = 0.70,
    confidence: float = 0.80,
    age_sec: float = 30.0,
    token_id: str = "yes",
    now: datetime = T0,
) -> NewsAssessment:
    return NewsAssessment(
        news_id="news-1",
        source="rss-test",
        headline="Titolo di test",
        published_at=now - timedelta(seconds=age_sec),
        target_token_id=token_id,
        estimated_fair_prob=fair,
        confidence=confidence,
    )

"""News latency evaluator (T2-C).

Data una `NewsAssessment` (fair prob stimata dall'LLM) e il book del token
target, decide se esiste ancora un edge eseguibile sull'ask prima che il
mercato prezzi la notizia.

Regole:
- la notizia deve essere fresca: `news_age_sec <= max_age_sec`;
- filtro di confidenza: `confidence >= min_confidence`;
- l'edge si misura contro il VWAP realmente eseguibile, non contro il best ask;
- si comprano solo livelli con prezzo <= `fair - fee_buffer - min_net_edge`,
  cosi anche il livello peggiore mantiene l'edge minimo;
- sizing: Kelly frazionario scalato per la confidenza e cappato a
  `max_position_usdc`.

Solo lato long (BUY del token target): il paper trader non ha inventario da
vendere. Modulo puro e deterministico: `now` e sempre passato dal chiamante.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from pydantic import ValidationError

from contracts.models import NewsAssessment, OrderBookSnapshot, OrderLeg, TradeSignal
from engine.book_math import EPS, best_price, fill_for_budget, is_finite_number

#: Allineati a `.env.example` (NEWS_MAX_AGE_SEC, MAX_POSITION_USDC, PAPER_STARTING_BANKROLL_USDC).
DEFAULT_MAX_AGE_SEC = 120.0
DEFAULT_MAX_POSITION_USDC = 50.0
DEFAULT_BANKROLL_USDC = 1000.0
DEFAULT_MIN_NET_EDGE = 0.03
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_MIN_POSITION_USDC = 1.0
DEFAULT_MAX_FUTURE_SKEW_SEC = 5.0


def news_age_seconds(published_at: object, now: object) -> float | None:
    """Eta della notizia in secondi; `None` se i timestamp mancano o non sono confrontabili."""
    if not isinstance(published_at, datetime) or not isinstance(now, datetime):
        return None
    try:
        return (now - published_at).total_seconds()
    except TypeError:  # mix di datetime naive e timezone-aware
        return None


def kelly_fraction_for_binary(fair_prob: float, price: float) -> float:
    """Frazione di Kelly per un contratto binario comprato a `price` che paga 1.

    f* = (q - p) / (1 - p). Restituisce 0 se non c'e edge o se p >= 1
    (nessuna divisione per zero).
    """
    if not is_finite_number(fair_prob) or not is_finite_number(price):
        return 0.0
    if price <= 0.0 or price >= 1.0 - EPS or fair_prob <= price:
        return 0.0
    return min(1.0, (fair_prob - price) / (1.0 - price))


def _signal_id(assessment: NewsAssessment, created_at: datetime) -> str:
    raw = "|".join(
        ["NEWS_LATENCY", str(assessment.news_id), assessment.target_token_id, created_at.isoformat()]
    )
    return "news-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluate_news(
    assessment: NewsAssessment | None,
    book: OrderBookSnapshot | None,
    *,
    now: datetime,
    event_id: str | None = None,
    max_age_sec: float = DEFAULT_MAX_AGE_SEC,
    min_net_edge: float = DEFAULT_MIN_NET_EDGE,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    fee_buffer: float = 0.0,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    bankroll_usdc: float = DEFAULT_BANKROLL_USDC,
    max_position_usdc: float = DEFAULT_MAX_POSITION_USDC,
    min_position_usdc: float = DEFAULT_MIN_POSITION_USDC,
    max_future_skew_sec: float = DEFAULT_MAX_FUTURE_SKEW_SEC,
    fee_rate: float = 0.0,
) -> TradeSignal | None:
    """Valuta una notizia contro il book e restituisce un segnale BUY o `None`.

    Args:
        assessment: valutazione strutturata della notizia.
        book: book L2 del token `assessment.target_token_id`.
        now: istante della valutazione (iniettato dal chiamante).
        event_id: evento a cui attribuire l'esposizione; default il token target.
        max_age_sec: eta massima della notizia (oltre e considerata gia prezzata).
        min_net_edge: edge netto minimo per share, anche sul livello peggiore.
        min_confidence: confidenza minima del classificatore.
        fee_buffer: fee stimate per share.
        kelly_fraction: moltiplicatore conservativo del Kelly pieno, in (0, 1].
        bankroll_usdc: bankroll virtuale su cui si applica il Kelly.
        max_position_usdc: cap assoluto alla spesa del singolo segnale.
        min_position_usdc: sotto questa spesa il segnale non viene emesso.
        max_future_skew_sec: tolleranza per notizie "dal futuro" (clock skew).
        fee_rate: commissione taker come frazione del controvalore speso
            (es. 0.02 = 2%). Il prezzo effettivo per share diventa
            `prezzo * (1 + fee_rate) + fee_buffer`.

    `max_position_usdc` limita la spesa commissioni incluse; `total_cost_usdc`
    del segnale resta il costo di book, senza commissioni.
    """
    if assessment is None or book is None:
        return None

    fair = getattr(assessment, "estimated_fair_prob", None)
    confidence = getattr(assessment, "confidence", None)
    if not is_finite_number(fair) or not 0.0 <= fair <= 1.0:
        return None
    if not is_finite_number(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    numeric_params = (
        max_age_sec, min_net_edge, min_confidence, fee_buffer, kelly_fraction,
        bankroll_usdc, max_position_usdc, min_position_usdc, max_future_skew_sec, fee_rate,
    )
    if any(not is_finite_number(param) or param < 0.0 for param in numeric_params):
        return None
    if kelly_fraction > 1.0:
        return None

    # 1. Freschezza della notizia.
    age_sec = news_age_seconds(getattr(assessment, "published_at", None), now)
    if age_sec is None or age_sec < -max_future_skew_sec:
        return None
    age_sec = max(0.0, age_sec)
    if age_sec > max_age_sec:
        return None

    # 2. Filtro di confidenza.
    if confidence < min_confidence:
        return None

    # 3. Book utilizzabile e riferito al token giusto.
    if getattr(book, "is_stale", True) or book.token_id != assessment.target_token_id:
        return None
    best_ask = best_price(book, "BUY")
    if best_ask is None:
        return None

    fee_multiplier = 1.0 + fee_rate

    def effective_price(book_price: float) -> float:
        """Costo reale di una share: prezzo di book piu commissione proporzionale e fissa."""
        return book_price * fee_multiplier + fee_buffer

    # 4. Edge sul top of book: se manca qui, manca ovunque piu in profondita.
    # Prezzo di book massimo a cui, pagate le commissioni, resta ancora l'edge minimo.
    limit_price = (fair - fee_buffer - min_net_edge) / fee_multiplier
    if best_ask > limit_price + EPS:
        return None

    # 5. Sizing: Kelly frazionario x confidenza, cappato. Il budget include le commissioni.
    scale = bankroll_usdc * kelly_fraction * confidence
    budget = min(scale * kelly_fraction_for_binary(fair, effective_price(best_ask)), max_position_usdc)
    if budget <= 0.0 or budget + EPS < min_position_usdc:
        return None
    fill = fill_for_budget(book, budget / fee_multiplier, limit_price=limit_price)
    if fill.vwap is None:
        return None
    # Il VWAP reale e >= best ask: si ricalcola il Kelly sul prezzo effettivo e si
    # riduce il budget se serve, cosi la spesa non supera mai il Kelly frazionario.
    refined_budget = min(budget, scale * kelly_fraction_for_binary(fair, effective_price(fill.vwap)))
    if refined_budget + EPS < budget:
        fill = fill_for_budget(book, refined_budget / fee_multiplier, limit_price=limit_price)
    if fill.vwap is None or fill.worst_price is None or fill.filled_shares <= 0.0:
        return None
    if fill.notional_usdc + EPS < min_position_usdc:
        return None

    net_edge = fair - effective_price(fill.vwap)
    if net_edge <= 0.0 or net_edge + EPS < min_net_edge:
        return None

    rationale = (
        f"News '{assessment.news_id}' ({assessment.source}) eta {age_sec:.1f}s <= {max_age_sec:.0f}s: "
        f"fair {fair:.4f} vs VWAP ask {fill.vwap:.6f} (commissione {fee_rate:.4%}, fee {fee_buffer:.4f}) "
        f"-> edge netto {net_edge:.6f}; "
        f"confidenza {confidence:.2f}, Kelly x{kelly_fraction:.2f}, "
        f"{fill.filled_shares:.6f} shares per {fill.notional_usdc:.4f} USDC "
        f"(cap {max_position_usdc:.2f})."
    )
    try:
        leg = OrderLeg(
            token_id=assessment.target_token_id,
            side="BUY",
            target_shares=fill.filled_shares,
            expected_vwap=fill.vwap,
            max_acceptable_price=fill.worst_price,
        )
        return TradeSignal(
            signal_id=_signal_id(assessment, now),
            strategy="NEWS_LATENCY",
            event_id=event_id if event_id else assessment.target_token_id,
            legs=[leg],
            expected_net_edge=net_edge,
            total_cost_usdc=fill.notional_usdc,
            created_at=now,
            rationale=rationale,
        )
    except ValidationError:
        return None

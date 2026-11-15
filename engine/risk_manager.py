"""Risk manager (T2-D).

Ultimo cancello prima del paper trader: ogni segnale prodotto da
`combinatorial_arb` o `news_latency` passa da `approve_signal`, che lo
ri-verifica contro i book CORRENTI e contro i limiti di capitale virtuale.

Modulo puro e deterministico: non tiene stato. L'esposizione aperta viene
passata dal chiamante (paper trader) a ogni chiamata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Mapping

from contracts.models import OrderBookSnapshot, TradeSignal
from engine.book_math import EPS, is_finite_number, simulate_fill

#: Allineati a `.env.example` (PAPER_STARTING_BANKROLL_USDC, MAX_POSITION_USDC).
DEFAULT_BANKROLL_USDC = 1000.0
DEFAULT_MAX_EVENT_EXPOSURE_USDC = 50.0


class RiskReason(StrEnum):
    """Codici di esito, pensati per finire nei log `SIGNAL_REJECTED`."""

    APPROVED = "APPROVED"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_SIGNAL = "INVALID_SIGNAL"
    NON_POSITIVE_EDGE = "NON_POSITIVE_EDGE"
    MISSING_BOOK = "MISSING_BOOK"
    STALE_BOOK = "STALE_BOOK"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"
    EVENT_EXPOSURE_EXCEEDED = "EVENT_EXPOSURE_EXCEEDED"
    TOTAL_EXPOSURE_EXCEEDED = "TOTAL_EXPOSURE_EXCEEDED"
    BANKROLL_EXCEEDED = "BANKROLL_EXCEEDED"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Esito del controllo rischio. `bool(decision)` equivale a `decision.approved`."""

    approved: bool
    reason: RiskReason
    detail: str
    required_usdc: float = 0.0

    def __bool__(self) -> bool:
        return self.approved


def _reject(reason: RiskReason, detail: str, required_usdc: float = 0.0) -> RiskDecision:
    return RiskDecision(approved=False, reason=reason, detail=detail, required_usdc=required_usdc)


def _book_age_seconds(book: OrderBookSnapshot, now: datetime) -> float | None:
    updated_at = getattr(book, "updated_at", None)
    if not isinstance(updated_at, datetime):
        return None
    try:
        return (now - updated_at).total_seconds()
    except TypeError:  # mix di datetime naive e timezone-aware
        return None


def approve_signal(
    signal: TradeSignal | None,
    books: Mapping[str, OrderBookSnapshot] | None,
    current_exposure_usdc: float,
    *,
    current_event_exposure_usdc: float = 0.0,
    bankroll_usdc: float = DEFAULT_BANKROLL_USDC,
    max_event_exposure_usdc: float = DEFAULT_MAX_EVENT_EXPOSURE_USDC,
    max_total_exposure_usdc: float | None = None,
    now: datetime | None = None,
    max_book_age_sec: float | None = None,
) -> RiskDecision:
    """Approva o blocca un segnale. Non solleva mai eccezioni.

    Blocca il segnale se:
    - anche un solo book coinvolto manca o ha `is_stale=True` (oppure, se sono
      forniti `now` e `max_book_age_sec`, e piu vecchio della soglia);
    - anche un solo leg non e interamente eseguibile entro il suo
      `max_acceptable_price` sul book corrente (liquidita mancante);
    - il capitale richiesto supera il bankroll libero, l'esposizione massima
      per evento o (se impostata) l'esposizione massima totale.

    Args:
        signal: segnale da validare.
        books: mappa `{token_id: OrderBookSnapshot}` aggiornata.
        current_exposure_usdc: esposizione aperta totale su tutti gli eventi.
        current_event_exposure_usdc: esposizione gia aperta su `signal.event_id`.
        bankroll_usdc: bankroll virtuale complessivo.
        max_event_exposure_usdc: cap di esposizione per singolo evento.
        max_total_exposure_usdc: cap opzionale di esposizione totale.
        now, max_book_age_sec: controllo opzionale di freschezza sui timestamp.

    Il capitale richiesto e il massimo tra `signal.total_cost_usdc` e il costo
    dei leg BUY ricalcolato sul book corrente (stima conservativa).
    """
    if signal is None or books is None:
        return _reject(RiskReason.INVALID_INPUT, "segnale o mappa dei book mancante")

    limits = (current_exposure_usdc, current_event_exposure_usdc, bankroll_usdc, max_event_exposure_usdc)
    if any(not is_finite_number(value) or value < 0.0 for value in limits):
        return _reject(RiskReason.INVALID_INPUT, "esposizione o limiti non numerici, NaN o negativi")
    if max_total_exposure_usdc is not None and (
        not is_finite_number(max_total_exposure_usdc) or max_total_exposure_usdc < 0.0
    ):
        return _reject(RiskReason.INVALID_INPUT, "max_total_exposure_usdc non valido")
    # Il controllo di eta si attiva solo con `max_book_age_sec`; allora serve anche `now`.
    check_age = max_book_age_sec is not None
    if check_age and (
        not isinstance(now, datetime) or not is_finite_number(max_book_age_sec) or max_book_age_sec < 0.0
    ):
        return _reject(RiskReason.INVALID_INPUT, "max_book_age_sec richiede un `now` valido e una soglia >= 0")

    legs = list(getattr(signal, "legs", None) or [])
    signal_cost = getattr(signal, "total_cost_usdc", None)
    edge = getattr(signal, "expected_net_edge", None)
    if not legs:
        return _reject(RiskReason.INVALID_SIGNAL, "segnale senza leg")
    if not is_finite_number(signal_cost) or signal_cost <= 0.0:
        return _reject(RiskReason.INVALID_SIGNAL, "total_cost_usdc non valido")
    if not is_finite_number(edge):
        return _reject(RiskReason.INVALID_SIGNAL, "expected_net_edge non valido")
    if edge <= 0.0:
        return _reject(RiskReason.NON_POSITIVE_EDGE, f"edge netto atteso {edge:.6f} <= 0")

    # 1. Tutti i book devono esistere ed essere freschi PRIMA di qualunque calcolo.
    for leg in legs:
        book = books.get(leg.token_id)
        if book is None:
            return _reject(RiskReason.MISSING_BOOK, f"book assente per token {leg.token_id}")
        if getattr(book, "is_stale", True):
            return _reject(RiskReason.STALE_BOOK, f"book stale per token {leg.token_id}")
        if check_age:
            age_sec = _book_age_seconds(book, now)
            if age_sec is None or age_sec > max_book_age_sec:
                return _reject(
                    RiskReason.STALE_BOOK,
                    f"book di {leg.token_id} senza timestamp valido o piu vecchio di {max_book_age_sec}s",
                )

    # 2. Liquidita: ogni leg deve essere interamente eseguibile entro il suo limite.
    recomputed_buy_cost = 0.0
    for leg in legs:
        if leg.side not in ("BUY", "SELL") or not is_finite_number(leg.target_shares):
            return _reject(RiskReason.INVALID_SIGNAL, f"leg non valido per token {leg.token_id}")
        fill = simulate_fill(
            books[leg.token_id], leg.side, leg.target_shares, limit_price=leg.max_acceptable_price
        )
        if not fill.fully_filled:
            return _reject(
                RiskReason.INSUFFICIENT_LIQUIDITY,
                f"token {leg.token_id}: eseguibili {fill.filled_shares:.6f} su "
                f"{leg.target_shares:.6f} shares entro {leg.max_acceptable_price}",
            )
        if leg.side == "BUY":
            recomputed_buy_cost += fill.notional_usdc

    required = max(float(signal_cost), recomputed_buy_cost)

    # 3. Limiti di capitale.
    if current_event_exposure_usdc + required > max_event_exposure_usdc + EPS:
        return _reject(
            RiskReason.EVENT_EXPOSURE_EXCEEDED,
            f"evento {signal.event_id}: {current_event_exposure_usdc:.4f} + {required:.4f} "
            f"> cap {max_event_exposure_usdc:.4f} USDC",
            required,
        )
    if (
        max_total_exposure_usdc is not None
        and current_exposure_usdc + required > max_total_exposure_usdc + EPS
    ):
        return _reject(
            RiskReason.TOTAL_EXPOSURE_EXCEEDED,
            f"esposizione totale {current_exposure_usdc:.4f} + {required:.4f} "
            f"> cap {max_total_exposure_usdc:.4f} USDC",
            required,
        )
    if current_exposure_usdc + required > bankroll_usdc + EPS:
        return _reject(
            RiskReason.BANKROLL_EXCEEDED,
            f"bankroll libero {bankroll_usdc - current_exposure_usdc:.4f} < richiesto {required:.4f} USDC",
            required,
        )

    return RiskDecision(
        approved=True,
        reason=RiskReason.APPROVED,
        detail=f"approvato: {required:.4f} USDC su evento {signal.event_id}",
        required_usdc=required,
    )

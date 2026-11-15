"""Arbitraggio combinatorio sum-to-one su gruppi mutuamente esclusivi (T2-B).

Comprando `N` shares di OGNI esito di un gruppo in cui esattamente un esito
paga 1 USDC, il payoff a scadenza e `N` USDC certi. Se il costo reale per set
(somma dei VWAP eseguibili sull'intero ask-side L2) e strettamente inferiore a
`1.0 - fee_buffer - slippage_buffer - min_net_edge`, il profitto e bloccato.

ASSUNZIONE: `MarketGroup.mutually_exclusive=True` deve significare anche
"esaustivo" (uno e un solo esito vince). Il contratto non ha un flag separato:
e responsabilita del collector emettere solo gruppi completi (es. coppia
YES/NO dello stesso mercato o evento negRisk con tutti gli esiti).

Modulo puro e deterministico: nessun I/O, nessun orologio di sistema.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Mapping

from pydantic import ValidationError

from contracts.models import MarketGroup, OrderBookSnapshot, OrderLeg, TradeSignal
from engine.book_math import (
    EPS,
    FillResult,
    available_shares,
    is_finite_number,
    simulate_fill,
)

#: Allineato a `MIN_ARB_NET_EDGE` di `.env.example`.
DEFAULT_MIN_NET_EDGE = 0.015


def _non_negative(value: object) -> bool:
    return is_finite_number(value) and value >= 0.0


def _latest_book_timestamp(books: list[OrderBookSnapshot]) -> datetime | None:
    """Timestamp piu recente tra i book, `None` se mancante o non confrontabile."""
    stamps = [getattr(book, "updated_at", None) for book in books]
    if not stamps or any(not isinstance(stamp, datetime) for stamp in stamps):
        return None
    try:
        return max(stamps)
    except TypeError:  # mix di datetime naive e timezone-aware
        return None


def _signal_id(event_id: str, token_ids: list[str], shares: float, created_at: datetime) -> str:
    raw = "|".join(["COMBINATORIAL_ARB", event_id, *token_ids, repr(shares), created_at.isoformat()])
    return "arb-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def evaluate_group(
    group: MarketGroup | None,
    books: Mapping[str, OrderBookSnapshot] | None,
    target_shares: float,
    *,
    fee_buffer: float = 0.0,
    slippage_buffer: float = 0.0,
    min_net_edge: float = DEFAULT_MIN_NET_EDGE,
    min_executable_shares: float | None = None,
    now: datetime | None = None,
    fee_rate: float = 0.0,
) -> TradeSignal | None:
    """Valuta un gruppo mutuamente esclusivo e restituisce un segnale bilanciato o `None`.

    Args:
        group: gruppo di esiti; deve avere `mutually_exclusive=True` e >= 2 esiti.
        books: mappa `{token_id: OrderBookSnapshot}` con il book di ogni esito.
        target_shares: numero di set completi (shares per leg) da comprare.
        fee_buffer: fee stimate per set, in USDC su 1$ di payoff.
        slippage_buffer: margine extra per slippage tra segnale ed esecuzione.
        min_net_edge: edge netto minimo richiesto per set (strettamente superato).
        min_executable_shares: se valorizzato (<= target), consente di ridurre la
            size alla profondita comune dei book purche resti >= di questa soglia.
            Con `None` il segnale richiede l'intera `target_shares` su ogni leg.
        now: istante della valutazione; se `None` si usa il timestamp piu recente
            tra i book coinvolti (resta deterministico).
        fee_rate: commissione taker come frazione del controvalore speso
            (es. 0.02 = 2%). Si somma a `fee_buffer`, che resta un importo fisso per set.

    `total_cost_usdc` del segnale resta il costo di book, senza commissioni:
    le commissioni entrano in `expected_net_edge`.

    Returns:
        `TradeSignal(strategy="COMBINATORIAL_ARB")` con un leg BUY per esito, tutti
        della stessa size, oppure `None` se l'opportunita non e valida/eseguibile.
    """
    if group is None or books is None:
        return None
    if not is_finite_number(target_shares) or target_shares <= 0.0:
        return None
    if not (_non_negative(fee_buffer) and _non_negative(slippage_buffer) and _non_negative(min_net_edge)):
        return None
    if not _non_negative(fee_rate):
        return None
    if min_executable_shares is not None and (
        not is_finite_number(min_executable_shares) or min_executable_shares <= 0.0
    ):
        return None
    if not getattr(group, "mutually_exclusive", False):
        return None

    outcomes = list(getattr(group, "outcomes", None) or [])
    token_ids = [outcome.token_id for outcome in outcomes]
    if len(token_ids) < 2 or len(set(token_ids)) != len(token_ids):
        return None

    group_books: list[OrderBookSnapshot] = []
    for token_id in token_ids:
        book = books.get(token_id)
        if book is None or getattr(book, "is_stale", True):
            return None
        group_books.append(book)

    # Size bilanciata: identica su ogni leg, limitata dal book meno profondo.
    shares = float(target_shares)
    common_depth = min(available_shares(book, "BUY") for book in group_books)
    if common_depth + EPS < shares:
        if min_executable_shares is None or common_depth + EPS < min_executable_shares:
            return None
        shares = common_depth
    if shares <= 0.0:
        return None

    fills: list[FillResult] = []
    for book in group_books:
        fill = simulate_fill(book, "BUY", shares)
        if not fill.fully_filled or fill.vwap is None or fill.worst_price is None:
            return None
        fills.append(fill)

    book_cost_per_set = sum(fill.vwap for fill in fills)
    # La commissione taker e proporzionale a quanto si spende: gonfia il costo del set.
    cost_per_set = book_cost_per_set * (1.0 + fee_rate)
    threshold = 1.0 - fee_buffer - slippage_buffer - min_net_edge
    # Disuguaglianza stretta: costo == soglia NON e un'opportunita.
    if threshold - cost_per_set <= EPS:
        return None

    created_at = now if isinstance(now, datetime) else _latest_book_timestamp(group_books)
    if created_at is None:
        return None

    net_edge = 1.0 - cost_per_set - fee_buffer - slippage_buffer
    total_cost = sum(fill.notional_usdc for fill in fills)
    rationale = (
        f"Sum-to-one su {len(fills)} esiti: costo VWAP per set {book_cost_per_set:.6f}, "
        f"{cost_per_set:.6f} con commissione {fee_rate:.4%}, "
        f"< soglia {threshold:.6f} (fee {fee_buffer:.4f}, slippage {slippage_buffer:.4f}, "
        f"min edge {min_net_edge:.4f}); {shares:.6f} shares per leg, "
        f"edge netto {net_edge:.6f} per 1$ di payoff."
    )
    try:
        legs = [
            OrderLeg(
                token_id=token_id,
                side="BUY",
                target_shares=shares,
                expected_vwap=fill.vwap,
                max_acceptable_price=fill.worst_price,
            )
            for token_id, fill in zip(token_ids, fills)
        ]
        return TradeSignal(
            signal_id=_signal_id(group.event_id, token_ids, shares, created_at),
            strategy="COMBINATORIAL_ARB",
            event_id=group.event_id,
            legs=legs,
            expected_net_edge=net_edge,
            total_cost_usdc=total_cost,
            created_at=created_at,
            rationale=rationale,
        )
    except ValidationError:
        return None

"""Matematica di profondita del book L2 (T2-A).

Funzioni pure e deterministiche: nessun I/O, nessuna rete, nessun orologio.
Tutte le funzioni camminano i livelli reali del book (zero assunzioni di
liquidita infinita sul top-of-book) e non sollevano mai eccezioni su input
degeneri: book vuoto, prezzi 0/NaN, size non valide o liquidita insufficiente
producono un `FillResult` vuoto/parziale, mai un errore a runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

from contracts.models import BookLevel, OrderBookSnapshot

Side = Literal["BUY", "SELL"]

#: Tolleranza numerica per confronti tra float (shares, USDC, prezzi).
EPS = 1e-9


@dataclass(frozen=True, slots=True)
class FillResult:
    """Esito di una camminata sui livelli del book.

    `notional_usdc` e il costo pagato (BUY) o l'incasso ricevuto (SELL).
    `vwap` e `worst_price` sono `None` se non e stata eseguita alcuna share.
    `fully_filled` e True solo se l'intera size richiesta e stata eseguita.
    """

    requested_shares: float
    filled_shares: float
    notional_usdc: float
    vwap: float | None
    worst_price: float | None
    levels_consumed: int
    fully_filled: bool


def is_finite_number(value: object) -> bool:
    """True se `value` e un numero reale finito (esclude bool, None, NaN, inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def clean_levels(
    levels: Iterable[BookLevel] | None, *, descending: bool
) -> list[tuple[float, float]]:
    """Restituisce i livelli validi come coppie `(price, size)` ordinate.

    Scarta i livelli con prezzo o size mancanti, NaN, infiniti, prezzo <= 0
    (un ask a 0 non e eseguibile ed e quasi sempre un dato corrotto), prezzo > 1
    o size <= 0. Riordina sempre i livelli in modo difensivo: non ci si fida
    dell'ordinamento ricevuto dal collector.
    """
    cleaned: list[tuple[float, float]] = []
    for level in levels or ():
        price = getattr(level, "price", None)
        size = getattr(level, "size", None)
        if not is_finite_number(price) or not is_finite_number(size):
            continue
        if price <= 0.0 or price > 1.0 or size <= 0.0:
            continue
        cleaned.append((float(price), float(size)))
    cleaned.sort(key=lambda item: item[0], reverse=descending)
    return cleaned


def executable_levels(
    book: OrderBookSnapshot | None, side: Side
) -> list[tuple[float, float]]:
    """Livelli contro cui si esegue: `asks` crescenti per BUY, `bids` decrescenti per SELL."""
    if book is None:
        return []
    if side == "BUY":
        return clean_levels(getattr(book, "asks", None), descending=False)
    if side == "SELL":
        return clean_levels(getattr(book, "bids", None), descending=True)
    return []


def _within_limit(price: float, limit_price: float | None, side: Side) -> bool:
    if limit_price is None:
        return True
    if side == "BUY":
        return price <= limit_price + EPS
    return price >= limit_price - EPS


def _empty_fill(requested_shares: float) -> FillResult:
    return FillResult(
        requested_shares=requested_shares,
        filled_shares=0.0,
        notional_usdc=0.0,
        vwap=None,
        worst_price=None,
        levels_consumed=0,
        fully_filled=False,
    )


def _build_fill(
    requested_shares: float,
    filled_shares: float,
    notionals: list[float],
    worst_price: float | None,
    fully_filled: bool,
) -> FillResult:
    if filled_shares <= 0.0 or not notionals:
        return _empty_fill(requested_shares)
    notional = math.fsum(notionals)
    # Il VWAP e una media pesata di prezzi in (0, 1]: il clamp assorbe solo
    # il rumore floating point, cosi il valore rispetta sempre i contratti Pydantic.
    vwap = min(1.0, max(0.0, notional / filled_shares))
    return FillResult(
        requested_shares=requested_shares,
        filled_shares=filled_shares,
        notional_usdc=notional,
        vwap=vwap,
        worst_price=worst_price,
        levels_consumed=len(notionals),
        fully_filled=fully_filled,
    )


def simulate_fill(
    book: OrderBookSnapshot | None,
    side: Side,
    target_shares: float,
    *,
    limit_price: float | None = None,
) -> FillResult:
    """Cammina il book per eseguire `target_shares` e ne calcola costo reale e VWAP.

    BUY consuma gli `asks` dal piu economico; SELL consuma i `bids` dal piu alto.
    Con `limit_price` ci si ferma al primo livello peggiore del limite.
    Se la liquidita non basta il risultato e parziale (`fully_filled=False`).
    """
    if not is_finite_number(target_shares) or target_shares <= 0.0:
        return _empty_fill(0.0)
    target = float(target_shares)
    if limit_price is not None and not is_finite_number(limit_price):
        return _empty_fill(target)

    remaining = target
    filled = 0.0
    notionals: list[float] = []
    worst_price: float | None = None
    for price, size in executable_levels(book, side):
        if remaining <= EPS:
            break
        if not _within_limit(price, limit_price, side):
            break
        take = min(size, remaining)
        notionals.append(take * price)
        filled += take
        remaining -= take
        worst_price = price

    fully_filled = filled > 0.0 and remaining <= EPS
    if fully_filled:
        filled = target  # elimina la deriva floating point sulla size
    return _build_fill(target, filled, notionals, worst_price, fully_filled)


def vwap_for_shares(
    book: OrderBookSnapshot | None, side: Side, target_shares: float
) -> float | None:
    """VWAP eseguibile per l'intera size, oppure `None` se non e interamente eseguibile."""
    fill = simulate_fill(book, side, target_shares)
    return fill.vwap if fill.fully_filled else None


def available_shares(
    book: OrderBookSnapshot | None,
    side: Side,
    *,
    limit_price: float | None = None,
) -> float:
    """Massima size eseguibile sul lato indicato, opzionalmente entro `limit_price`."""
    if limit_price is not None and not is_finite_number(limit_price):
        return 0.0
    sizes = [
        size
        for price, size in executable_levels(book, side)
        if _within_limit(price, limit_price, side)
    ]
    return math.fsum(sizes)


def best_price(book: OrderBookSnapshot | None, side: Side) -> float | None:
    """Miglior prezzo eseguibile (best ask per BUY, best bid per SELL) o `None` se assente."""
    levels = executable_levels(book, side)
    return levels[0][0] if levels else None


def fill_for_budget(
    book: OrderBookSnapshot | None,
    budget_usdc: float,
    *,
    limit_price: float | None = None,
) -> FillResult:
    """Massimo acquisto eseguibile sugli `asks` spendendo al piu `budget_usdc`.

    In questo caso `requested_shares` coincide con le shares ottenute e
    `fully_filled` indica soltanto che e stato eseguito qualcosa: il vincolo
    e il budget, non una size richiesta.
    """
    if not is_finite_number(budget_usdc) or budget_usdc <= 0.0:
        return _empty_fill(0.0)
    if limit_price is not None and not is_finite_number(limit_price):
        return _empty_fill(0.0)

    remaining_budget = float(budget_usdc)
    filled = 0.0
    notionals: list[float] = []
    worst_price: float | None = None
    for price, size in executable_levels(book, "BUY"):
        if remaining_budget <= EPS:
            break
        if not _within_limit(price, limit_price, "BUY"):
            break
        take = min(size, remaining_budget / price)
        if take <= 0.0:
            break
        notionals.append(take * price)
        filled += take
        remaining_budget -= take * price
        worst_price = price

    return _build_fill(filled, filled, notionals, worst_price, filled > 0.0)

"""Virtual portfolio for the paper bot: open positions, mark-to-market and drawdown.

The log already says what was bought and at what cost. This module answers the question
the 7-day run exists for: is the bot making or losing (virtual) money?

Three numbers are kept apart on purpose, because they mean different things:

- **unrealized P&L**: what the open positions would fetch if sold now at the best bid,
  minus what they cost. It moves with the market.
- **locked arbitrage P&L**: for a sum-to-one set, exactly one leg pays 1 USDC at
  resolution whatever happens, so the profit is fixed the moment the set is bought.
  It is only as good as the group being exhaustive (see engine/combinatorial_arb.py).
- **realized P&L**: profit of positions that have actually been settled.

Costs include taker fees. Nothing here sends, signs or prices real orders.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any, Mapping

from contracts.models import OrderBookSnapshot, PaperFill, TradeSignal

ARB_STRATEGY = "COMBINATORIAL_ARB"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class Lot:
    """One filled signal: the shares bought per token and what they cost, fees included."""

    signal_id: str
    event_id: str
    strategy: str
    legs: dict[str, float]  # token_id -> shares bought
    cost_usdc: float
    fees_usdc: float = 0.0
    opened_at: str = ""

    @property
    def is_arbitrage(self) -> bool:
        return self.strategy == ARB_STRATEGY and len(self.legs) >= 2

    @property
    def sets(self) -> float:
        """Complete sum-to-one sets held: limited by the smallest leg."""
        return min(self.legs.values()) if self.legs else 0.0

    @property
    def locked_pnl_usdc(self) -> float:
        """Profit fixed at purchase for an arbitrage lot (one leg pays 1 per set)."""
        return self.sets - self.cost_usdc if self.is_arbitrage else 0.0


@dataclass
class PaperPortfolio:
    """In-memory ledger of paper positions, rebuilt from the JSONL log after a restart."""

    starting_bankroll_usdc: float = 1000.0
    open_lots: list[Lot] = field(default_factory=list)
    realized_pnl_usdc: float = 0.0
    fees_paid_usdc: float = 0.0
    settled_lots: int = 0
    # Events confirmed closed: the risk manager stops counting them, even if a
    # single-outcome lot is still waiting for its result.
    closed_events: set[str] = field(default_factory=set)
    peak_equity_usdc: float = field(init=False)
    max_drawdown_usdc: float = 0.0
    _marks: dict[str, float] = field(default_factory=dict)  # token_id -> last usable best bid

    def __post_init__(self) -> None:
        self.peak_equity_usdc = float(self.starting_bankroll_usdc)

    # ------------------------------------------------------------------ recording

    def record_fill(self, signal: TradeSignal | None, fill: PaperFill, fees_usdc: float | None = None) -> Lot | None:
        """Open a lot for a FILLED simulation; anything else leaves the portfolio untouched."""
        if fill.status != "FILLED":
            return None
        cost = _finite(fill.actual_simulated_cost_usdc)
        if cost is None or cost <= 0.0:
            return None
        legs: dict[str, float] = {}
        for leg in fill.filled_legs:
            shares = _finite(leg.target_shares)
            if leg.side != "BUY" or shares is None or shares <= 0.0:
                continue
            legs[leg.token_id] = legs.get(leg.token_id, 0.0) + shares
        if not legs:
            return None
        fees = max(0.0, _finite(fees_usdc) or 0.0)
        lot = Lot(
            signal_id=fill.signal_id,
            event_id=signal.event_id if signal is not None else "",
            strategy=fill.strategy,
            legs=legs,
            cost_usdc=cost,
            fees_usdc=fees,
            opened_at=fill.executed_at.astimezone(timezone.utc).isoformat()
            if fill.executed_at.tzinfo
            else fill.executed_at.replace(tzinfo=timezone.utc).isoformat(),
        )
        self.open_lots.append(lot)
        self.fees_paid_usdc += fees
        return lot

    # ------------------------------------------------------------------ valuation

    def update_marks(self, books: Mapping[str, OrderBookSnapshot] | None) -> None:
        """Remember the best bid of every fresh book; stale or empty books keep the last mark."""
        for token_id, book in (books or {}).items():
            if book is None or getattr(book, "is_stale", True):
                continue
            bids = [
                price
                for price in (_finite(getattr(level, "price", None)) for level in book.bids)
                if price is not None and 0.0 < price <= 1.0
            ]
            if bids:
                self._marks[token_id] = max(bids)

    def _market_value(self) -> tuple[float, int]:
        """Value of open lots at the last best bid; tokens never priced count as 0."""
        value = 0.0
        unpriced: set[str] = set()
        for lot in self.open_lots:
            for token_id, shares in lot.legs.items():
                mark = self._marks.get(token_id)
                if mark is None:
                    unpriced.add(token_id)
                    continue
                value += shares * mark
        return value, len(unpriced)

    def snapshot(self, books: Mapping[str, OrderBookSnapshot] | None = None) -> dict[str, Any]:
        """Current P&L picture; also updates the peak equity and the maximum drawdown."""
        if books is not None:
            self.update_marks(books)
        cost_basis = sum(lot.cost_usdc for lot in self.open_lots)
        market_value, unpriced = self._market_value()
        unrealized = market_value - cost_basis
        equity = self.starting_bankroll_usdc + self.realized_pnl_usdc + unrealized
        if equity > self.peak_equity_usdc:
            self.peak_equity_usdc = equity
        drawdown = self.peak_equity_usdc - equity
        if drawdown > self.max_drawdown_usdc:
            self.max_drawdown_usdc = drawdown
        return {
            "starting_bankroll_usdc": self.starting_bankroll_usdc,
            "open_lots": len(self.open_lots),
            "settled_lots": self.settled_lots,
            "cost_basis_usdc": cost_basis,
            "fees_paid_usdc": self.fees_paid_usdc,
            "market_value_usdc": market_value,
            "unpriced_tokens": unpriced,
            "unrealized_pnl_usdc": unrealized,
            "locked_arb_pnl_usdc": sum(lot.locked_pnl_usdc for lot in self.open_lots),
            "realized_pnl_usdc": self.realized_pnl_usdc,
            "equity_usdc": equity,
            "peak_equity_usdc": self.peak_equity_usdc,
            "max_drawdown_usdc": self.max_drawdown_usdc,
            "max_drawdown_pct": (
                self.max_drawdown_usdc / self.peak_equity_usdc * 100.0 if self.peak_equity_usdc > 0.0 else 0.0
            ),
        }

    # ------------------------------------------------------------------ settlement

    def _close(self, lot: Lot, payout_usdc: float) -> float:
        pnl = payout_usdc - lot.cost_usdc
        self.realized_pnl_usdc += pnl
        self.settled_lots += 1
        self.open_lots.remove(lot)
        return pnl

    def settle_event(self, event_id: str) -> dict[str, Any]:
        """Settle an event confirmed closed, without knowing which outcome won.

        Arbitrage lots pay 1 USDC per complete set whoever wins, so they are realized.
        Single-outcome lots (news) need the actual result: they stay open and are counted
        in `awaiting_outcome` until `settle_tokens` is called with real payouts.
        """
        realized = 0.0
        settled = 0
        awaiting = 0
        if event_id:
            self.closed_events.add(event_id)
        for lot in [candidate for candidate in self.open_lots if candidate.event_id == event_id]:
            if lot.is_arbitrage:
                realized += self._close(lot, payout_usdc=lot.sets)
                settled += 1
            else:
                awaiting += 1
        return {"realized_pnl_usdc": realized, "settled_lots": settled, "awaiting_outcome": awaiting}

    def settle_tokens(self, payouts: Mapping[str, float]) -> dict[str, Any]:
        """Settle every open lot whose tokens all have a known payout per share (0..1)."""
        realized = 0.0
        settled = 0
        for lot in list(self.open_lots):
            values = [_finite(payouts.get(token_id)) for token_id in lot.legs]
            if any(value is None or not 0.0 <= value <= 1.0 for value in values):
                continue
            payout = sum(shares * value for shares, value in zip(lot.legs.values(), values))
            realized += self._close(lot, payout_usdc=payout)
            settled += 1
        return {"realized_pnl_usdc": realized, "settled_lots": settled}

    def exposure_by_event(self) -> dict[str, float]:
        """Money committed per event that is still open, fees included.

        This is what the risk manager's per-event caps must see after a restart: the cost
        of every open lot, grouped by event, leaving out events already confirmed closed.
        """
        exposure: dict[str, float] = {}
        for lot in self.open_lots:
            if not lot.event_id or lot.event_id in self.closed_events:
                continue
            exposure[lot.event_id] = exposure.get(lot.event_id, 0.0) + lot.cost_usdc
        return exposure

    # ------------------------------------------------------------------ restart

    @classmethod
    def from_log(cls, log_path: str | Path, starting_bankroll_usdc: float = 1000.0) -> "PaperPortfolio":
        """Rebuild the portfolio from PAPER_FILL and PAPER_SETTLED entries of a JSONL log.

        Without this a restart in the middle of the run would silently reset the P&L.
        Marks are not stored in the log: positions are unpriced until fresh books arrive.
        """
        portfolio = cls(starting_bankroll_usdc=starting_bankroll_usdc)
        path = Path(log_path)
        if not path.exists():
            return portfolio
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
                    continue
                payload = entry["payload"]
                if entry.get("event_type") == "PAPER_FILL":
                    portfolio._restore_fill(payload)
                elif entry.get("event_type") == "PAPER_SETTLED":
                    event_id = str(payload.get("event_id") or "")
                    if event_id:
                        portfolio.settle_event(event_id)
        return portfolio

    def _restore_fill(self, payload: dict[str, Any]) -> None:
        cost = _finite(payload.get("actual_simulated_cost_usdc"))
        if payload.get("status") != "FILLED" or cost is None or cost <= 0.0:
            return
        legs: dict[str, float] = {}
        for leg in payload.get("filled_legs") or []:
            if not isinstance(leg, dict) or leg.get("side") != "BUY":
                continue
            shares = _finite(leg.get("target_shares"))
            token_id = str(leg.get("token_id") or "")
            if token_id and shares is not None and shares > 0.0:
                legs[token_id] = legs.get(token_id, 0.0) + shares
        if not legs:
            return
        fees = max(0.0, _finite(payload.get("fees_usdc")) or 0.0)
        self.open_lots.append(
            Lot(
                signal_id=str(payload.get("signal_id") or ""),
                event_id=str(payload.get("event_id") or ""),
                strategy=str(payload.get("strategy") or ""),
                legs=legs,
                cost_usdc=cost,
                fees_usdc=fees,
                opened_at=str(payload.get("executed_at") or ""),
            )
        )
        self.fees_paid_usdc += fees

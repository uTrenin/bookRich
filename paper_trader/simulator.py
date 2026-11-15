"""Deterministic L2 order-book fill simulator for Polymarket paper trading."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from contracts.models import BookLevel, OrderBookSnapshot, OrderLeg, PaperFill, TradeSignal


class PaperOrderSimulator:
    """Simulates realistic multi-level L2 execution without signing or sending orders."""

    def __init__(self, fill_tolerance: float = 1e-9, price_tolerance: float = 1e-12) -> None:
        self.fill_tolerance = fill_tolerance
        self.price_tolerance = price_tolerance

    def simulate_signal(
        self,
        signal: TradeSignal,
        books: dict[str, OrderBookSnapshot],
        executed_at: datetime | None = None,
    ) -> PaperFill:
        """Walk real L2 ask/bid levels to compute VWAP, cost, and slippage for a signal."""
        exec_ts = executed_at if executed_at is not None else datetime.now(timezone.utc)

        if not signal.legs:
            return self._rejected_fill(
                signal=signal,
                executed_at=exec_ts,
                status="SKIPPED_RISK",
                note="Signal has no legs",
            )

        for leg in signal.legs:
            book = books.get(leg.token_id)
            if book is None:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="SKIPPED_RISK",
                    note=f"Missing order book for token_id={leg.token_id}",
                )
            if book.is_stale:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="SKIPPED_RISK",
                    note=f"Order book is stale for token_id={leg.token_id}",
                )

        filled_legs: list[OrderLeg] = []
        total_cost_usdc = 0.0
        total_shares = 0.0
        weighted_slippage_sum = 0.0

        for leg in signal.legs:
            if not math.isfinite(leg.target_shares) or leg.target_shares <= 0.0:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="PARTIAL_REJECTED",
                    note=f"Invalid target_shares={leg.target_shares} for token_id={leg.token_id}",
                )

            book = books[leg.token_id]
            levels = self._ordered_levels(book=book, side=leg.side)
            if not levels:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="PARTIAL_REJECTED",
                    note=f"Empty {leg.side} book side for token_id={leg.token_id}",
                )

            best_price = levels[0].price
            if not math.isfinite(best_price) or best_price <= 0.0:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="PARTIAL_REJECTED",
                    note=f"Invalid top-of-book price={best_price} for token_id={leg.token_id}",
                )

            # A limit order never trades through its limit: only levels priced within
            # max_acceptable_price are executable, whatever the resulting average would be.
            filled_shares, leg_cost = self._walk_levels(
                levels,
                leg.target_shares,
                side=leg.side,
                limit_price=leg.max_acceptable_price,
                price_tolerance=self.price_tolerance,
            )
            if filled_shares + self.fill_tolerance < leg.target_shares:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="PARTIAL_REJECTED",
                    note=(
                        f"Insufficient L2 depth within max_acceptable_price "
                        f"{leg.max_acceptable_price:.6f} for token_id={leg.token_id}: "
                        f"filled={filled_shares:.6f} < target={leg.target_shares:.6f}"
                    ),
                )

            raw_vwap = leg_cost / leg.target_shares
            actual_vwap = max(0.0, min(1.0, raw_vwap))

            if leg.side == "BUY" and actual_vwap > leg.max_acceptable_price + self.price_tolerance:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="PARTIAL_REJECTED",
                    note=(
                        f"BUY VWAP {actual_vwap:.6f} exceeds max_acceptable_price "
                        f"{leg.max_acceptable_price:.6f} for token_id={leg.token_id}"
                    ),
                )
            if leg.side == "SELL" and actual_vwap < leg.max_acceptable_price - self.price_tolerance:
                return self._rejected_fill(
                    signal=signal,
                    executed_at=exec_ts,
                    status="PARTIAL_REJECTED",
                    note=(
                        f"SELL VWAP {actual_vwap:.6f} below max_acceptable_price "
                        f"{leg.max_acceptable_price:.6f} for token_id={leg.token_id}"
                    ),
                )

            leg_slippage = (
                actual_vwap - best_price if leg.side == "BUY" else best_price - actual_vwap
            )
            weighted_slippage_sum += leg_slippage * leg.target_shares
            total_shares += leg.target_shares
            total_cost_usdc += leg_cost

            filled_legs.append(
                OrderLeg(
                    token_id=leg.token_id,
                    side=leg.side,
                    target_shares=leg.target_shares,
                    expected_vwap=actual_vwap,
                    max_acceptable_price=leg.max_acceptable_price,
                )
            )

        slippage_vs_best_ask = (
            weighted_slippage_sum / total_shares if total_shares > 0.0 else 0.0
        )

        return PaperFill(
            signal_id=signal.signal_id,
            strategy=signal.strategy,
            executed_at=exec_ts,
            filled_legs=filled_legs,
            actual_simulated_cost_usdc=total_cost_usdc,
            slippage_vs_best_ask=slippage_vs_best_ask,
            status="FILLED",
            note=f"Simulated {len(filled_legs)} leg(s) against L2 order book",
        )

    @staticmethod
    def _ordered_levels(book: OrderBookSnapshot, side: str) -> list[BookLevel]:
        """Return ask levels sorted ascending for BUY or bid levels sorted descending for SELL."""
        if side == "BUY":
            return sorted(book.asks, key=lambda lvl: lvl.price)
        return sorted(book.bids, key=lambda lvl: lvl.price, reverse=True)

    @staticmethod
    def _walk_levels(
        levels: list[BookLevel],
        target_shares: float,
        side: str = "BUY",
        limit_price: float | None = None,
        price_tolerance: float = 0.0,
    ) -> tuple[float, float]:
        """Accumulate share quantity and notional USDC cost across ordered L2 levels.

        With `limit_price`, levels priced beyond the limit (above it for BUY, below it for
        SELL) are not executable and are skipped.
        """
        remaining = target_shares
        filled_shares = 0.0
        total_cost = 0.0
        has_limit = limit_price is not None and math.isfinite(limit_price)

        for level in levels:
            if remaining <= 0.0:
                break
            if (
                not math.isfinite(level.price)
                or not math.isfinite(level.size)
                or level.price <= 0.0
                or level.size <= 0.0
            ):
                continue
            if has_limit:
                beyond_limit = (
                    level.price > limit_price + price_tolerance
                    if side == "BUY"
                    else level.price < limit_price - price_tolerance
                )
                if beyond_limit:
                    continue
            fill_qty = min(remaining, level.size)
            filled_shares += fill_qty
            total_cost += fill_qty * level.price
            remaining -= fill_qty

        return filled_shares, total_cost

    @staticmethod
    def _rejected_fill(
        signal: TradeSignal,
        executed_at: datetime,
        status: str,
        note: str,
    ) -> PaperFill:
        """Build a zero-cost non-filled PaperFill response."""
        return PaperFill(
            signal_id=signal.signal_id,
            strategy=signal.strategy,
            executed_at=executed_at,
            filled_legs=[],
            actual_simulated_cost_usdc=0.0,
            slippage_vs_best_ask=0.0,
            status=status,  # type: ignore[arg-type]
            note=note,
        )

"""Core quant engine di bookRich: funzioni pure e deterministiche, zero I/O.

Riceve snapshot tipizzati da `contracts.models` e restituisce `TradeSignal | None`.
"""

from engine.book_math import (
    FillResult,
    available_shares,
    best_price,
    fill_for_budget,
    simulate_fill,
    vwap_for_shares,
)
from engine.combinatorial_arb import evaluate_group
from engine.news_latency import evaluate_news
from engine.risk_manager import RiskDecision, RiskReason, approve_signal

__all__ = [
    "FillResult",
    "RiskDecision",
    "RiskReason",
    "approve_signal",
    "available_shares",
    "best_price",
    "evaluate_group",
    "evaluate_news",
    "fill_for_budget",
    "simulate_fill",
    "vwap_for_shares",
]

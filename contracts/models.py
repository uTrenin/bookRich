"""Contratti Pydantic condivisi tra /collector, /engine e /paper_trader (T0).

ATTENZIONE: Nessuno deve modificare questo file senza accordo esplicito tra i due membri del team.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class BookLevel(BaseModel):
    """Singolo livello di prezzo e liquidita nell'order book L2 di Polymarket."""

    price: float = Field(ge=0.0, le=1.0)
    size: float = Field(gt=0.0)


class OrderBookSnapshot(BaseModel):
    """Stato corrente del book L2 per un token_id."""

    token_id: str
    bids: list[BookLevel]  # ordinati dal prezzo piu alto al piu basso
    asks: list[BookLevel]  # ordinati dal prezzo piu basso al piu alto
    updated_at: datetime
    is_stale: bool = False


class MarketOutcome(BaseModel):
    """Singolo esito negoziabile associato a un token_id CLOB."""

    market_id: str
    question: str
    outcome_label: str  # es. "Yes", "No", oppure nome candidato
    token_id: str


class MarketGroup(BaseModel):
    """Gruppo di esiti collegati (es. evento multi-outcome mutuamente esclusivo)."""

    event_id: str
    title: str
    mutually_exclusive: bool
    outcomes: list[MarketOutcome]


class NewsAssessment(BaseModel):
    """Valutazione strutturata di una notizia rispetto a un token target."""

    news_id: str
    source: str
    headline: str
    published_at: datetime
    target_token_id: str
    estimated_fair_prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class OrderLeg(BaseModel):
    """Singola gamba di un segnale o ordine simulato."""

    token_id: str
    side: Literal["BUY", "SELL"]
    target_shares: float = Field(gt=0.0)
    expected_vwap: float = Field(ge=0.0, le=1.0)
    max_acceptable_price: float = Field(ge=0.0, le=1.0)


class TradeSignal(BaseModel):
    """Segnale emesso da /engine e consumato da /paper_trader."""

    signal_id: str
    strategy: Literal["COMBINATORIAL_ARB", "NEWS_LATENCY"]
    event_id: str
    legs: list[OrderLeg]
    expected_net_edge: float  # es. 0.024 = +2.4% netto su 1$ payoff
    total_cost_usdc: float
    created_at: datetime
    rationale: str


class PaperFill(BaseModel):
    """Esito della simulazione di esecuzione sul book L2 reale."""

    signal_id: str
    strategy: str
    executed_at: datetime
    filled_legs: list[OrderLeg]
    actual_simulated_cost_usdc: float
    slippage_vs_best_ask: float
    status: Literal["FILLED", "PARTIAL_REJECTED", "SKIPPED_RISK"]
    note: str

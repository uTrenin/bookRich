# PIANO ARCHITETTURALE & CONTRATTI — Polymarket Arbitrage & Latency Paper-Bot

## 1. Architettura a 2 Recinti (Zero Conflitti Git)

```text
Polymarket_Arbitrage/
├── GEMINI.md & CLAUDE.md          # Costituzione condivisa
├── docs/                          # spec.md, plan.md, tasks.md
├── contracts/                     # Contratti Pydantic condivisi (Agreement-first)
│   └── models.py                  # OrderBookSnapshot, MarketGroup, NewsEvent, TradeSignal, PaperFill
├── collector/                     # RECINTO 1 (Tu - Antigravity Swarm)
│   ├── gamma_discovery.py         # REST Gamma API -> MarketGroup & token_ids
│   ├── clob_ws_stream.py          # WebSocket L2 order book + PING 10s + reconnect
│   └── news_stream.py             # Polling/stream notizie + estrattore AI rapido
├── engine/                        # RECINTO 2 (Amico - Claude Max / Fable 5.2)
│   ├── book_math.py               # VWAP / depth-walking su book L2 (calc costo reale per N shares)
│   ├── combinatorial_arb.py       # Rilevamento sum(asks) < 1.0 - buffer su MarketGroup
│   ├── news_latency.py            # Confronto fair_prob AI vs book ask + decadimento temporale
│   └── risk_manager.py            # Cap esposizione, check book stale, sizing finale
├── paper_trader/                  # RECINTO 1 (Tu - Antigravity Swarm)
│   ├── simulator.py               # Simula fill sul book reale + calcola slippage
│   ├── logger.py                  # Scrittura JSONL append-only + summary giornaliero
│   └── runner.py                  # Loop asyncio principale 24/7 (collega collector -> engine -> simulator)
├── infra/                         # RECINTO 1 (Tu) — Dockerfile, compose per run 7 giorni
├── tests/
│   ├── engine/                    # RECINTO 2 (Amico) — pytest deterministici su matematica/risk
│   └── integration/               # RECINTO 1 (Tu) — test replay WebSocket -> engine -> JSONL log
└── logs/                          # Output locale escluso da Git (`paper_trades.jsonl`)
```

## 2. Endpoint Polymarket Reali di Riferimento
- **Market Discovery (Gamma API):** `GET https://gamma-api.polymarket.com/events?active=true&closed=false&limit=100` (estrae eventi multi-outcome e `clobTokenIds`).
- **Snapshot Iniziale Book (CLOB REST):** `GET https://clob.polymarket.com/book?token_id=<TOKEN_ID>`
- **Stream Live L2 (CLOB WebSocket):** `wss://ws-subscriptions-clob.polymarket.com/ws/market`
  - Payload sottoscrizione: `{"assets_ids": ["<TOKEN_ID_1>", ...], "type": "market"}`
  - Heartbeat obbligatorio: frame `PING` ogni ~10 secondi.

## 3. Contratti Dati Condivisi (`/contracts/models.py` — Pydantic v2)

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class BookLevel(BaseModel):
    price: float = Field(ge=0.0, le=1.0)
    size: float = Field(gt=0.0)

class OrderBookSnapshot(BaseModel):
    token_id: str
    bids: list[BookLevel]  # ordinati dal prezzo più alto al più basso
    asks: list[BookLevel]  # ordinati dal prezzo più basso al più alto
    updated_at: datetime
    is_stale: bool = False

class MarketOutcome(BaseModel):
    market_id: str
    question: str
    outcome_label: str     # es. "Yes", "No", oppure nome candidato
    token_id: str

class MarketGroup(BaseModel):
    event_id: str
    title: str
    mutually_exclusive: bool
    outcomes: list[MarketOutcome]

class NewsAssessment(BaseModel):
    news_id: str
    source: str
    headline: str
    published_at: datetime
    target_token_id: str
    estimated_fair_prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

class OrderLeg(BaseModel):
    token_id: str
    side: Literal["BUY", "SELL"]
    target_shares: float = Field(gt=0.0)
    expected_vwap: float = Field(ge=0.0, le=1.0)
    max_acceptable_price: float = Field(ge=0.0, le=1.0)

class TradeSignal(BaseModel):
    signal_id: str
    strategy: Literal["COMBINATORIAL_ARB", "NEWS_LATENCY"]
    event_id: str
    legs: list[OrderLeg]
    expected_net_edge: float       # es. 0.024 = +2.4% netto su 1$ payoff
    total_cost_usdc: float
    created_at: datetime
    rationale: str

class PaperFill(BaseModel):
    signal_id: str
    strategy: str
    executed_at: datetime
    filled_legs: list[OrderLeg]
    actual_simulated_cost_usdc: float
    slippage_vs_best_ask: float
    status: Literal["FILLED", "PARTIAL_REJECTED", "SKIPPED_RISK"]
    note: str
```

## 4. Schema del Log 7 Giorni (`logs/paper_trades.jsonl`)
Ogni riga è un oggetto JSON autonomo con:
`{"ts": "...", "event_type": "PAPER_FILL | SIGNAL_REJECTED | DAILY_SUMMARY", "payload": { ... }}`
Così al giorno 7 basta uno script di analisi per calcolare:
- Totale opportunità combinatorie viste vs effettivamente eseguibili sulla liquidità reale.
- Latenza media (ms) tra `published_at` notizia e `executed_at`.
- P&L virtuale totale, drawdown massimo e distribuzione slippage.

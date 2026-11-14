# Prompt Operativo per il Team — bookRich

## Per Amico (Claude Max / Claude Code · Branch `feat/quant-engine`)

### 1. Setup iniziale da terminale
```bash
git clone https://github.com/uTrenin/bookRich.git
cd bookRich
git checkout -b feat/quant-engine
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate
pip install pydantic pytest
claude
```

### 2. Impostazioni in Claude Code
1. Seleziona il modello di punta con `/model` (Opus / Fable 5.2).
2. Attiva **Plan Mode** premendo `Shift + Tab` (oppure `/plan`).
3. Incolla il prompt qui sotto, verifica il piano matematico proposto e poi avvia l'esecuzione.

### 3. Prompt da incollare in Claude Code
```text
Leggi `CLAUDE.md`, `docs/spec.md`, `docs/plan.md`, `docs/tasks.md` e `contracts/models.py`.

REGOLE DI RECINTO (NON NEGOZIABILI):
- Il tuo recinto esclusivo è `/engine` e `/tests/engine` (crea anche `engine/__init__.py`).
- NON creare né modificare file in `/collector`, `/paper_trader`, `/infra` o `contracts/models.py` (importa i tipi esclusivamente da `contracts.models`).
- Zero chiamate HTTP/WebSocket o I/O dentro `/engine`: devono essere funzioni e classi pure deterministiche.

OBIETTIVO (Task T2-A .. T2-D):
Progetta e implementa i 4 moduli quantitativi:
1. `engine/book_math.py`: calcolo del prezzo medio ponderato eseguibile (VWAP) e della massima size eseguibile camminando i livelli `asks` (e `bids`) di `OrderBookSnapshot`. Gestisci esplicitamente book vuoto, liquidità insufficiente per la size richiesta e prezzi 0/NaN.
2. `engine/combinatorial_arb.py`: dato un `MarketGroup` e la mappa `{token_id: OrderBookSnapshot}`, verifica se l'acquisto simultaneo di tutti gli esiti mutuamente esclusivi per una `target_shares` produce un costo VWAP totale inferiore a `1.0 - fee_buffer - min_net_edge`. Restituisci `TradeSignal(strategy="COMBINATORIAL_ARB", ...)` bilanciato oppure `None`.
3. `engine/news_latency.py`: dato `NewsAssessment` e `OrderBookSnapshot`, verifica che `news_age_sec <= max_age_sec`, calcola l'edge tra `estimated_fair_prob` e VWAP eseguibile sull'ask, applica un sizing conservativo (Kelly frazionario cappato a `max_position_usdc`) e restituisci `TradeSignal(strategy="NEWS_LATENCY", ...)` o `None`.
4. `engine/risk_manager.py`: funzione `approve_signal(signal, books, current_exposure_usdc, ...)` che blocca qualunque segnale se anche un solo book coinvolto ha `is_stale=True`, se manca liquidità o se supera l'esposizione massima per evento/bankroll.

Scrivi almeno 20 unit test deterministici in `/tests/engine/` coprendo happy-path ed edge case critici (book vuoto, liquidità parziale su più livelli, notizia scaduta, book stale, somma ask == 1.0, divisione per zero). Al termine esegui `pytest tests/engine` finché passano tutti al 100%.
```

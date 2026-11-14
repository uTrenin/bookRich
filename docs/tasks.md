# BACKLOG ATOMICO DI SPRINT — Polymarket Arbitrage & Latency Paper-Bot

## Fase 0 · Allineamento Iniziale (Insieme — 30 min)
- [x] `[T0]` **Shared Contract:** Condividere e bloccare `/contracts/models.py` esattamente come da `docs/plan.md` su `main`.

## Fase 1 · Sviluppo Parallelo nei 2 Recinti

### Recinto 1 — TU (`Antigravity Swarm / Gemini Pro`) · Branch `feat/collector-paper-runner`
- [ ] `[T1-A]` **Market Discovery (`collector/gamma_discovery.py`):** interrogare Gamma API per estrarre coppie/gruppi di mercati attivi e mappare i `clobTokenIds` in `MarketGroup`.
- [ ] `[T1-B]` **Order Book Stream (`collector/clob_ws_stream.py`):** client WebSocket asincrono verso `wss://ws-subscriptions-clob.polymarket.com/ws/market` + snapshot REST iniziale (`GET /book`) + `PING` ogni 10s + gestione `is_stale` e riconnessione automatica.
- [ ] `[T1-C]` **News Feed + Fast Classifier (`collector/news_stream.py`):** ingestion feed notizie/RSS/X mockabile + chiamata asincrona a Gemini Flash per produrre `NewsAssessment` strutturato con latenza < 3s.
- [ ] `[T1-D]` **Paper Trader & 7-Day Logger (`paper_trader/`):** simulatore fill su book L2, file append `logs/paper_trades.jsonl`, report giornaliero e `runner.py` 24/7 (`infra/Dockerfile`).

### Recinto 2 — AMICO (`Claude Max / Fable 5.2 + Claude Code`) · Branch `feat/quant-engine`
- [ ] `[T2-A]` **L2 Book Depth Math (`engine/book_math.py`):** funzione pura per calcolare costo reale VWAP e max shares eseguibili camminando i livelli `asks` del book (zero assunzioni di liquidità infinita sul top-of-book).
- [ ] `[T2-B]` **Combinatorial Arbitrage Engine (`engine/combinatorial_arb.py`):** rilevamento incoerenze su `MarketGroup` mutuamente esclusivi (e coppie `YES`/`NO`) al netto di fee/slippage buffer, restituendo `TradeSignal` multi-leg bilanciato.
- [ ] `[T2-C]` **News Latency Evaluator (`engine/news_latency.py`):** calcolo edge tra `estimated_fair_prob` e `executable_vwap`, filtro `news_age_sec <= max_age`, sizing proporzionale alla confidenza (Kelly frazionario conservativo).
- [ ] `[T2-D]` **Risk Manager & Test Suite (`engine/risk_manager.py` + `tests/engine/`):** controllo bankroll virtuale, esposizione massima per evento, blocco su `is_stale=True` e 20+ unit test deterministici su edge case.

## Fase 2 · Cross-Model Review & Integrazione (Insieme)
- [ ] `[T3-A]` **Cross-Review #1 (Tu con Antigravity revisioni `PR feat/quant-engine`):** verifica matematica dell'arbitraggio, assenza di `ZeroDivisionError`, gestione book vuoti/parziali e aderenza a `contracts/models.py`.
- [ ] `[T3-B]` **Cross-Review #2 (Amico con Claude Code revisiona `PR feat/collector-paper-runner`):** verifica gestione asincrona WebSocket, calcolo P&L/slippage nel paper trader, assenza di race condition su book aggiornati in parallelo.
- [ ] `[T3-C]` **Test di Integrazione Replay + Avvio Dry-Run 7 Giorni:** esecuzione test end-to-end su dati reali live (zero euro investiti) e controllo dei primi log in `logs/paper_trades.jsonl`.

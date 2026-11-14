# COSTITUZIONE DEL PROGETTO — Polymarket Arbitrage & Latency Paper-Bot

Leggi sempre `docs/spec.md` e `docs/plan.md` prima di creare o modificare qualsiasi file.

## 1. Obiettivo & Regola Zero-Soldi-Reali (FASE PAPER-TRADING 7 GIORNI)
- Questo sistema opera **ESCLUSIVAMENTE in modalità simulazione (Paper Trading)** per almeno 7 giorni su stream reale di Polymarket.
- **VIETATO** firmare transazioni on-chain, inviare ordini reali al CLOB o salvare chiavi private wallet (`PRIVATE_KEY`) nel codice o in `.env` durante questa fase. Solo lettura pubblica (`Gamma API`, `CLOB public book`, `WebSocket market channel`).

## 2. Stack Tecnologico Obbligatorio
- **Linguaggio:** Python 3.11+ (`asyncio`, `httpx`, `websockets`, `pydantic` v2, `pytest`)
- **Market Data:** Polymarket Gamma API (`https://gamma-api.polymarket.com`), CLOB REST (`https://clob.polymarket.com`), CLOB WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`)
- **AI Latency Pipeline:** Gemini Flash / LLM adapter asincrono per valutazione rapida notizie
- **Persistenza Simulation Log:** File locale `logs/paper_trades.jsonl` (append-only) + report giornaliero JSON/CSV (opzionale sync su Supabase)

## 3. Regole dei 2 Recinti (Zero Conflitti Git)
- **Recinto 1 — TU (Antigravity Swarm / Gemini Pro):** `/collector`, `/paper_trader`, `/infra`, `/tests/integration`.
  - Responsabilità: acquisizione market/order-book L2, stream news, simulatore di esecuzione su book reale, log 7 giorni e runner 24/7.
- **Recinto 2 — AMICO (Claude Max / Fable 5.2 + Claude Code):** `/engine`, `/tests/engine`.
  - Responsabilità: matematica dell'arbitraggio combinatorio, calcolo depth-weighted fill price, modello di valutazione news-latency e risk manager deterministico.
- **Contratti condivisi:** `/contracts` (o modelli Pydantic definiti in `docs/plan.md`) — modificabili solo previo accordo tra entrambi.
- **Regola di ferro:** NON modificare file fuori dal tuo recinto. Lavora sempre su branch `feat/...` e apri Pull Request su `main` per la **Cross-Model Code Review**.

## 4. Standard di Qualità & Sicurezza
- Tutti gli importi, probabilità e prezzi devono usare `float` validati (`0.0 <= price <= 1.0`, nessuna divisione per zero su book vuoto o `best_ask == 0`).
- Gestione esplicita di disconnessioni WebSocket (heartbeat `PING` ogni ~10s, reconnect con backoff esponenziale, marcatura book come `stale` se fermo oltre soglia).
- Ogni modulo in `/engine` e `/paper_trader` richiede unit test `pytest` (happy-path + edge case: book vuoto, spread enorme, liquidità parziale, latenza notizia scaduta).
- Prima di dichiarare finito un task, esegui `pytest` e verifica 0 errori.

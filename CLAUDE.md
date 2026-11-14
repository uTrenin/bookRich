# COSTITUZIONE DEL PROGETTO — Polymarket Arbitrage & Latency Paper-Bot

Leggi sempre `docs/spec.md` e `docs/plan.md` prima di creare o modificare qualsiasi file.

## 1. Obiettivo & Regola Zero-Soldi-Reali (FASE PAPER-TRADING 7 GIORNI)
- Questo sistema opera **ESCLUSIVAMENTE in modalità simulazione (Paper Trading)** per almeno 7 giorni su stream reale di Polymarket.
- **VIETATO** firmare transazioni on-chain, inviare ordini reali al CLOB o introdurre dipendenze da chiavi private wallet. Solo lettura pubblica dell'order book e simulazione locale.

## 2. Stack Tecnologico Obbligatorio
- **Linguaggio:** Python 3.11+ (`asyncio`, `pydantic` v2, `pytest`)
- **Core Quant Engine (`/engine`):** Python puro deterministico. Nessuna chiamata HTTP/WebSocket diretta dentro `/engine`: riceve snapshot tipizzati (`OrderBookSnapshot`, `NewsEvent`) e restituisce `TradeSignal | None`.

## 3. Regole dei 2 Recinti (Zero Conflitti Git)
- **Recinto 2 — TUO COMPITO (Claude Max / Fable 5.2 + Claude Code):** `/engine`, `/tests/engine`.
  - NON modificare `/collector`, `/paper_trader` o `/infra`.
- **Recinto 1 — COMPAGNO (Antigravity Swarm / Gemini Pro):** `/collector`, `/paper_trader`, `/infra`, `/tests/integration`.
- Lavora sempre sul tuo branch (`feat/quant-engine` o `feat/arb-math`) e apri Pull Request per far revisionare il codice da Gemini/Antigravity (**Cross-Model Review**).

## 4. Regole Matematiche & Qualità in `/engine`
- **Arbitraggio Combinatorio:** verifica sempre sull'intero ask-side L2 (non solo `best_ask` per 1$!) includendo fee stimate, buffer slippage e dimensione minima eseguibile. Un'opportunità è valida SOLO se il costo totale per comprare tutti i leg mutuamente esclusivi è `< 1.0 - min_edge_buffer`.
- **News Latency Arbitrage:** calcola sempre la differenza tra `model_fair_prob` e `executable_ask_price`, applica un filtro di confidenza e scarta segnali se `news_age_sec > max_latency_sec` (es. notizia già prezzata dal mercato).
- **Zero eccezioni a runtime:** gestisci esplicitamente book vuoti, `ask == 0` o `NaN`, liquidità insufficiente, timestamp mancanti.
- Scrivi sempre test `pytest` rigorosi in `/tests/engine/` ed eseguili prima di ogni commit/PR.

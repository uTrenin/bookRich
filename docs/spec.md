# SPEC FUNZIONALE — Polymarket Arbitrage & News Latency Paper-Bot

## 1. Obiettivo del Prodotto (Weekend MVP + Test 7 Giorni)
Costruire un sistema Python asincrono capace di osservare in tempo reale i mercati e gli order book di **Polymarket** (senza investire soldi reali) per individuare, valutare e simulare su file log due strategie di alpha per **7 giorni consecutivi**:
1. **Arbitraggio Combinatorio (Book Imbalance / Sum-to-One):** rileva gruppi di mercati mutuamente esclusivi (o binari `YES`/`NO`) in cui il costo reale di acquisto di tutti gli esiti sul lato Ask (tenendo conto della profondità L2 del book e dei costi) è inferiore a `1.00 USDC - soglia_minima` (es. `< 0.985`), bloccando un profitto matematico teorico a scadenza.
2. **News Latency Arbitrage:** legge notizie fresche da feed primari (RSS / API X / feed ufficiali), stima in < 3 secondi tramite LLM (`Gemini Flash` / `Fable 5.2`) se la probabilità implicita è cambiata rispetto al prezzo Ask corrente su mercati meno liquidi, e registra un trade virtuale se `edge > soglia` e la notizia è entro la finestra temporale (`news_age_sec <= 120s`).

## 2. Flusso Operativo End-to-End (Core Loop)
1. **Discovery (`collector`):** interroga la Gamma API di Polymarket per selezionare mercati attivi e gruppi di mercati mutuamente esclusivi (`negRisk` / eventi multi-outcome) con liquidità minima configurata.
2. **Streaming Order Book (`collector`):** si sottoscrive via WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) mantenendo in memoria lo stato L2 (Bids/Asks ordinati) di ogni `token_id`, con heartbeat `PING` ogni 10s e verifica freschezza (`stale_ms`).
3. **Valutazione Segnali (`engine`):**
   - Ogni aggiornamento book su un gruppo di mercati attiva `combinatorial_arb.evaluate_group(...)`.
   - Ogni nuova notizia ricevuta attiva il classificatore AI e poi `news_latency.evaluate_news(...)`.
   - Entrambi passano da `risk_manager.approve_signal(...)` (budget virtuale, esposizione massima per evento, liquidità disponibile).
4. **Simulazione & Log 7 Giorni (`paper_trader`):**
   - Esegue un **fill simulato realistico** camminando i livelli Ask/Bid effettivi del book in memoria (niente prezzi "magici" a size infinita!).
   - Scrive ogni decisione (`SIGNAL_REJECTED`, `PAPER_BUY`, `PAPER_SETTLED`, `HEARTBEAT_STATS`) in `logs/paper_trades.jsonl`.
   - Genera un riepilogo giornaliero (numero opportunità rilevate, edge medio atteso, slippage simulato, P&L virtuale marcato a mercato e a risoluzione).

## 3. Criteri di Accettazione (Definition of Done)
- [ ] Lo script gira ininterrottamente su stream reale Polymarket gestendo riconnessioni WebSocket senza crashare.
- [ ] Nessun ordine reale o firma blockchain viene mai inviata (modalità `PAPER_ONLY` hardcoded e verificata da test).
- [ ] L'arbitraggio combinatorio calcola il VWAP effettivo sulla profondità del book per una size target (es. `25 USDC`), non solo sul `best_ask` da `1$`.
- [ ] Il log `logs/paper_trades.jsonl` contiene tutte le prove necessarie dopo 7 giorni per capire se l'edge copre spread + slippage + latenza.

## 4. Esplicitamente FUORI Scope nel Weekend
- Invio ordini reali on-chain / gestione chiavi private Polygon / wallet signing.
- Arbitraggio cross-exchange (es. Polymarket vs Kalshi) — in questa fase si resta su Polymarket.
- Frontend complesso: al massimo una CLI/reportistica chiara o mini-dashboard di sola lettura dei log.

# hmm-regime-detection

Market-regime detection with Hidden Markov Models.

Two tracks live in this repo:

1. **Daily PoC** (original) — `GaussianHMM` on public daily index data via yfinance.
   See `src/Testrun.py` and `src/{data_loader,model,evaluate}.py`.
2. **Intraday pipeline** (new) — ~10-minute regime detection on proprietary
   order-flow data (order book + aggressor flow, *hajar kiri/kanan*), built to run
   live but validated offline first. Plan: `../.claude/plans/the-past-week-...md`.

## Intraday pipeline — quickstart

The live feed (a REST/websocket endpoint you "ngehit") isn't wired up yet, so the
pipeline ships with a **synthetic order-flow generator** that makes everything
runnable today. The architecture isolates the data source behind one interface, so
going live later means implementing a single file.

```bash
python run_demo.py
```

This generates + records a synthetic feed, replays it into session-aware 10-min
bars, builds microstructure features, de-seasonalizes, selects the number of states
by BIC, trains the HMM, runs the **online filter** bar-by-bar, and writes:

- `results/figures/intraday_regimes.png` — price shaded by detected regime
- `data/processed/backtest_regimes.csv` / `regimes.sqlite` — regimes + probabilities

### Module map (`src/`)

| File | Role |
|------|------|
| `config.py` | sessions, features, paths, model params (single source of truth) |
| `ingest/base.py` | `FeedAdapter` interface + canonical event schema |
| `ingest/synthetic_adapter.py` | synthetic order-flow generator (with ground truth) |
| `ingest/recorder.py` / `replay.py` | record feed to JSONL / replay it deterministically |
| `ingest/websocket_adapter.py` | **live feed stub** — implement after Phase 0 data contract |
| `bars.py` | events → session-aware rich 10-min bars (OFI, signed vol, …) |
| `features.py` | rich bars → model features (point-in-time, session-reset returns) |
| `seasonality.py` | per-time-of-day de-seasonalization (fit on train only) |
| `model.py` | HMM training, AIC/BIC selection, artifact save/load |
| `labels.py` | interpret states + Hungarian matching across retrains |
| `online.py` | forward-only filtering (honest real-time inference) |
| `detector.py` | one-bar orchestration (live == backtest code path) |
| `backtest.py` | fit-on-train + run detector over history |
| `evaluate.py` | persistence, economic coherence, filter-vs-smoother, ground-truth ARI |
| `store.py` | persist outputs (CSV / SQLite) |

### Going live (later)

Implement `ingest/websocket_adapter.py` against the real endpoint (it maps raw
messages to the canonical event shape). Nothing else changes — `bars → features →
detector` already depend only on that shape.
```

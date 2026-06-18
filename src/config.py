"""
Central configuration for the intraday regime-detection pipeline.

Everything that is environment- or market-specific lives here so the rest of the
code depends on names, not magic numbers. Most of these values (session times,
tick handling, instrument) must be CONFIRMED with the team lead in Phase 0 — the
defaults below are sensible IDX-style placeholders so the pipeline runs today on
synthetic data.
"""
from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # recorded feed (JSONL), our backtest dataset
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"   # versioned model artifacts (pickle)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Market / instrument
# --------------------------------------------------------------------------- #
TZ = ZoneInfo("Asia/Jakarta")
DEFAULT_SYMBOL = "DEMO"

# IDX/BEI continuous-trading sessions (WIB), as (start, end) local-time strings.
# We model the CONTINUOUS sessions only; the pre-opening (08:45–08:59) and
# pre-closing (15:50–16:00) auctions are excluded on purpose (no continuous
# trading there). Friday has a longer lunch break and shorter morning session.
REGULAR_SESSIONS = [("09:00", "12:00"), ("13:30", "15:49")]   # Mon–Thu
FRIDAY_SESSIONS = [("09:00", "11:30"), ("14:00", "15:49")]    # Fri


def sessions_for(weekday: int):
    """Return the session list for a given weekday (Mon=0 .. Sun=6)."""
    return FRIDAY_SESSIONS if weekday == 4 else REGULAR_SESSIONS

BAR_MINUTES = 10          # detection interval


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
# The model feature vector. Keep this list as the single source of truth — the
# scaler, seasonality stats and HMM are all built against exactly these columns.
FEATURE_COLS = [
    "ret",              # log return of bar close vs previous bar (within session)
    "signed_vol_imb",   # (buy_vol - sell_vol) / volume  -> net hajar kanan/kiri
    "ofi",              # order-flow imbalance (Cont-Kukanov-Stoikov, top level)
    "rel_spread",       # average bid-ask spread / mid
    "realized_vol",     # std of intra-bar mid log-returns
    "trade_intensity",  # number of trades in the bar
]

# Features whose intraday U-shape must be removed before modelling.
SEASONAL_COLS = ["rel_spread", "realized_vol", "trade_intensity"]

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
N_STATES_RANGE = range(2, 7)   # AIC/BIC sweep range
RANDOM_STATE = 42
HMM_N_ITER = 200
TRAIN_RATIO = 0.7              # time-ordered train/test split for backtest

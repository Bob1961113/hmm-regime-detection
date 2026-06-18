"""
Time-of-day de-seasonalization.

Intraday volume / volatility / spread follow a U-shape (huge at the open and
close, quiet midday). If we feed raw features to the HMM it mostly learns
"open vs lunch" instead of real regimes — the single biggest intraday pitfall.

Fix: for each time-of-day slot (e.g. "09:10") learn a robust center and scale
(median / IQR) from the TRAINING data only, then express every feature as a
robust z-score relative to its own slot. Fitting on train only keeps the
transform point-in-time (no future leakage into the stats).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_EPS = 1e-9


def fit_seasonality(train_features: pd.DataFrame, cols=None) -> dict:
    """Learn per-slot median and IQR for the given columns from training data."""
    cols = cols or config.SEASONAL_COLS
    stats: dict[str, pd.DataFrame] = {}
    for col in cols:
        g = train_features.groupby("slot")[col]
        med = g.median()
        iqr = g.quantile(0.75) - g.quantile(0.25)
        iqr = iqr.replace(0, np.nan)            # avoid divide-by-zero on flat slots
        stats[col] = pd.DataFrame({"median": med, "iqr": iqr})
    # global fallbacks for slots/cols unseen in training
    fallback = {c: {"median": float(train_features[c].median()),
                    "iqr": float((train_features[c].quantile(0.75)
                                  - train_features[c].quantile(0.25)) or 1.0)}
                for c in cols}
    return {"cols": cols, "per_slot": stats, "fallback": fallback}


def apply_seasonality(features: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Return a copy of `features` with seasonal columns replaced by robust
    per-slot z-scores."""
    out = features.copy()
    for col in stats["cols"]:
        slot_stats = stats["per_slot"][col]
        fb = stats["fallback"][col]
        med = out["slot"].map(slot_stats["median"]).fillna(fb["median"])
        iqr = out["slot"].map(slot_stats["iqr"]).fillna(fb["iqr"])
        out[col] = (out[col] - med.values) / (iqr.values + _EPS)
    return out

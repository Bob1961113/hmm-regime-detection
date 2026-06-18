"""
Derive the model feature vector from rich bars.

Every feature here is point-in-time: it uses only the current bar and earlier
bars *within the same session*. The bar-to-bar log return is computed per session
group so it never bridges the lunch break or the overnight gap (the first bar of
each session has no return and is dropped).

Output columns are exactly `config.FEATURE_COLS`, plus context columns
(close, slot, session_id, date) that downstream code needs but the model ignores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_EPS = 1e-9


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Map rich bars -> model features. Returns a DataFrame aligned to bar_start."""
    if bars.empty:
        return bars.copy()

    df = bars.copy()

    # Bar-to-bar log return, reset within each (date, session) group so it is
    # never computed across a session boundary.
    grp = df.groupby(["date", "session_id"], sort=False)
    df["ret"] = grp["close"].transform(lambda s: np.log(s / s.shift(1)))

    df["signed_vol_imb"] = df["signed_vol"] / (df["volume"] + _EPS)
    df["rel_spread"] = df["avg_spread"] / (df["close"] + _EPS)
    # ofi, realized_vol, n_trades come straight from the rich bar
    df["trade_intensity"] = df["n_trades"].astype(float)

    keep = config.FEATURE_COLS + ["close", "slot", "session_id", "date"]
    out = df[keep].copy()

    # Drop the first bar of each session (no return). All other features are valid.
    out = out.dropna(subset=["ret"])
    return out

"""
Detector: turn one bar's raw features into a labelled regime, online.

It wires together the persisted artifact (seasonality stats + scaler + HMM +
labels) and the `OnlineFilter`. The exact same code path is used in backtest and
in the live service, so what we validate offline is what runs in production.

Session handling: when the (date, session_id) changes, the filter is reset so the
hidden chain restarts from the model's initial distribution rather than carrying
state across the lunch break / overnight gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .seasonality import apply_seasonality
from .online import OnlineFilter


class Detector:
    def __init__(self, artifact: dict):
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.seasonality_stats = artifact["seasonality_stats"]
        self.feature_cols = artifact["feature_cols"]
        self.state_labels = artifact.get("state_labels") or {}
        self.filter = OnlineFilter(self.model)
        self._session_key = None

    def process_bar(self, row) -> dict:
        """`row` is a mapping/Series with the model feature columns plus
        'slot', 'session_id', 'date'. Returns the filtered regime + probabilities."""
        row = dict(row)
        key = (row["date"], row["session_id"])
        if key != self._session_key:
            self.filter.reset()
            self._session_key = key

        one = pd.DataFrame([row])
        one = apply_seasonality(one, self.seasonality_stats)
        x = self.scaler.transform(one[self.feature_cols].values)[0]

        posterior = self.filter.update(x)
        state = int(np.argmax(posterior))
        return {
            "state": state,
            "label": self.state_labels.get(state, f"state_{state}"),
            "probs": posterior,
        }

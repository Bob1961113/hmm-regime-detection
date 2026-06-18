"""
Backtest: fit the pipeline on a training window, then run the live detector
bar-by-bar over the full history.

Pipeline fit order (all learned from TRAIN only, to stay point-in-time):
    raw features -> de-seasonalize -> RobustScaler -> HMM (+ AIC/BIC selection)
                 -> interpret states -> save artifact

The detector is then run forward over every bar (true filtering, identical to the
live path). For diagnostics we also compute the full-sequence smoothed states so
we can measure how much a smoother would have "cheated" by using the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from . import config
from .seasonality import fit_seasonality, apply_seasonality
from .model import train_hmm, select_n_states
from .labels import label_states
from .detector import Detector


def fit_pipeline(features_raw: pd.DataFrame, n_states: int | None = None,
                 train_ratio: float = config.TRAIN_RATIO) -> tuple[dict, int]:
    """Fit seasonality + scaler + HMM on the training window. Returns
    (artifact_bundle, split_index)."""
    split = int(len(features_raw) * train_ratio)
    train = features_raw.iloc[:split]

    stats = fit_seasonality(train)
    train_des = apply_seasonality(train, stats)

    scaler = RobustScaler().fit(train_des[config.FEATURE_COLS].values)
    X_train = scaler.transform(train_des[config.FEATURE_COLS].values)

    selection = None
    if n_states is None:
        selection, n_states = select_n_states(X_train, config.N_STATES_RANGE,
                                              n_iter=config.HMM_N_ITER,
                                              random_state=config.RANDOM_STATE)

    model = train_hmm(X_train, n_components=n_states,
                      n_iter=config.HMM_N_ITER, random_state=config.RANDOM_STATE)
    print(f"HMM converged: {model.monitor_.converged}")

    train_states = model.predict(X_train)
    state_labels = label_states(train_des, train_states, n_states)

    artifact = {
        "model": model,
        "scaler": scaler,
        "seasonality_stats": stats,
        "feature_cols": config.FEATURE_COLS,
        "state_labels": state_labels,
        "ref_means": model.means_,
        "meta": {"n_states": n_states, "train_ratio": train_ratio,
                 "selection": selection},
    }
    return artifact, split


def run_backtest(features_raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Run the online detector over every bar; attach filtered + smoothed regimes."""
    detector = Detector(artifact)
    n_states = artifact["meta"]["n_states"]

    states, labels, prob_rows = [], [], []
    for _, row in features_raw.iterrows():
        res = detector.process_bar(row)
        states.append(res["state"])
        labels.append(res["label"])
        prob_rows.append(res["probs"])

    df = features_raw.copy()
    df["filt_state"] = states
    df["filt_label"] = labels
    probs = np.vstack(prob_rows)
    for s in range(n_states):
        df[f"prob_{s}"] = probs[:, s]

    # Full-sequence smoothed states (uses the future) — diagnostics only.
    # `lengths` makes the smoother reset at every session boundary too, exactly
    # like the online filter does. Without it the comparison is unfair: the filter
    # restarts each session while the smoother would treat the whole multi-day
    # history as one continuous chain (bridging lunch/overnight gaps).
    des = apply_seasonality(features_raw, artifact["seasonality_stats"])
    X = artifact["scaler"].transform(des[artifact["feature_cols"]].values)
    lengths = features_raw.groupby(["date", "session_id"], sort=False).size().tolist()
    df["smooth_state"] = artifact["model"].predict(X, lengths=lengths)

    return df


def train_and_backtest(features_raw: pd.DataFrame, n_states: int | None = None):
    """Convenience: fit on train window, backtest over all bars.
    Returns (df, artifact, split_index)."""
    artifact, split = fit_pipeline(features_raw, n_states=n_states)
    df = run_backtest(features_raw, artifact)
    return df, artifact, split

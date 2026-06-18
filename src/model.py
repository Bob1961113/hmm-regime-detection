"""
HMM training, state-count selection, and artifact I/O for the intraday detector.

These operate on ALREADY-SCALED feature matrices; scaling + de-seasonalization
are owned by the detector/backtest so they can be persisted alongside the model.

NOTE — the original *daily* index PoC (yfinance SPY/IHSG, returns/volatility/
momentum, Viterbi `predict` + k-step `predict_regime` forecasting) is NOT here
anymore. It lived in `Testrun.py` / `data_loader.py` and is preserved in the
Jupyter notebooks under `notebooks/` (04–08). This module is intraday-only.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from hmmlearn.hmm import GaussianHMM


def train_hmm(X_scaled, n_components=4, n_iter=200, random_state=42):
    """Fit a full-covariance GaussianHMM on a scaled feature matrix."""
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_scaled)
    return model


def select_n_states(X_scaled, n_range=range(2, 7), n_iter=200, random_state=42):
    """AIC/BIC sweep over the number of hidden states (uses hmmlearn's built-in
    .aic()/.bic()). Returns (results, best_n_by_bic)."""
    results = []
    for n in n_range:
        m = train_hmm(X_scaled, n_components=n, n_iter=n_iter, random_state=random_state)
        aic, bic = m.aic(X_scaled), m.bic(X_scaled)
        results.append((n, aic, bic, m.monitor_.converged))
        print(f"n={n} | AIC: {aic:.2f} | BIC: {bic:.2f} | Converged: {m.monitor_.converged}")
    best_n = min(results, key=lambda r: r[2])[0]
    print(f"\nBest number of states by BIC: {best_n}")
    return results, best_n


def save_artifact(path, *, model, scaler, seasonality_stats, feature_cols,
                  state_labels=None, ref_means=None, meta=None):
    """Persist everything needed to reproduce inference as a single pickle:
    the HMM, the scaler, the seasonality stats, the feature list, and the
    state->label mapping. Versioning the whole bundle keeps train and inference
    consistent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "seasonality_stats": seasonality_stats,
        "feature_cols": list(feature_cols),
        "state_labels": state_labels,
        "ref_means": ref_means,
        "meta": meta or {},
    }
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    return path


def load_artifact(path):
    """Load a bundle saved by `save_artifact`."""
    with Path(path).open("rb") as f:
        return pickle.load(f)

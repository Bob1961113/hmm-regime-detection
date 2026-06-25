"""
Walk-forward (expanding-window) cross-validation for the forward-prediction model.

A single 70/30 split gives ONE out-of-sample number - possibly a lucky/unlucky
slice. Walk-forward CS trains on the growing past and tests on each successive
future block, giving several honest scores so you can report mean +- spread. The
spread tells you how much the single-splt number could have swung.

Two time-series safeguards, both applied as a gap removed just before each test
block (where leakage would enter):
  * embargo       - drop bars right before the test block to break train/test
                    autocorrelation
  * purge_horizon - drop train bars whose FORWARD label window reaches into the 
                    test block (a sample at i has a label using bars up to
                    i + horizon; if that touches the test block it must go).

With horizon = 1 the label is one future bar, so purging removes a single bar and
barely matters; it only becomes important once you forecast several bars ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

def expanding_walk_forward(n_samples: int, n_folds: int = 5, 
                           embargo: int = 0, purge_horizon: int = 0):
    """Yield (train_index, test_index) for an expanding-window walk-forward.
    
    The timeline is cut into n_folds + 1 contiguous blocks: block 0 is the intial 
    train-only warmup, and each later block is one test fold (oldest first). A gap
    of `embargo + purge_horizon` bars is removed just before each test block.
    """
    block_edges = np.linspace(0, n_samples, n_folds + 2, dtype=int)
    gap = embargo + purge_horizon
    for fold in range(1, n_folds + 1):
        test_start, test_end = block_edges[fold], block_edges[fold + 1]
        train_end = max(0, test_start - gap)
        train_index = np.arange(0, train_end)
        test_index = np.arange(test_start, test_end)
        if len(train_index) and len(test_index):
            yield train_index, test_index

def cross_validate_forward(feature_matrix: pd.DataFrame, target: pd.Series,
                           n_folds: int = 5, embargo: int = 0,
                           purge_horizon: int = 1) -> np.ndarray:
    """Walk-forward CV of the XGBoost forward model; print per-fold and aggregate
    macro-F1. Returns the array of per-fold scores."""
    from sklearn.metrics import f1_score
    from .model_xgb import train_classifier

    fold_scores = []
    for fold_num, (train_index, test_index) in enumerate(
            expanding_walk_forward(len(target), n_folds, embargo, purge_horizon), start=1):
        features_train, target_train = feature_matrix.iloc[train_index], target.iloc[train_index]
        features_test, target_test = feature_matrix.iloc[test_index], target.iloc[test_index]

        classifier = train_classifier(features_train, target_train)
        predictions = classifier.predict(features_test)
        score = f1_score(target_test, predictions, average="macro")
        fold_scores.append(score)
        print(f"fold {fold_num}: train={len(train_index):4d} test={len(test_index):4d}  "
              f"macro-F1={score:.3f}")
        
    fold_scores = np.array(fold_scores)
    print(f"\nWalk-forward macro-F1: {fold_scores.mean():.3f} +/- {fold_scores.std():.3f}  "
          f"(mean over {len(fold_scores)}folds )")
    return fold_scores


def _smoke_test() -> None:
    """build synthetic data + the forward vol target, then run walk-forward CV.
    Run with:  python -m src.walk_forward  (from the repo root)."""
    from ..ingest.synthetic_adapter import SyntheticAdapter
    from ..bars import build_bars
    from ..features import compute_features
    from .targets import make_vol_regime_target

    adapter = SyntheticAdapter(symbol="DEMO", n_days=40, seed=7)
    features = compute_features(build_bars(adapter.stream()))
    
    # Bucket edges fit ion the first 70% (your fixed "what is volatile" definition);
    # walk-forward then tests the MODEL across later blocks. Re-fitting edges per
    # fold is a later refinement - the coarse quantile cut barely moves results.
    split_index = int(len(features) * config.TRAIN_RATIO)
    target, _edges, valid_mask = make_vol_regime_target(features, split_index, horizon=1)

    feature_matrix = features.loc[valid_mask, config.FEATURE_COLS]
    target = target[valid_mask].astype(int)

    cross_validate_forward(feature_matrix, target, n_folds=5, embargo=0, purge_horizon=1)


if __name__ == "__main__":
    _smoke_test()
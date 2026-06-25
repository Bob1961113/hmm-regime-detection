"""
XGBoost classifier for the supervised forward-prediction task (gen-2).

Companion to the unsupervised HMM: instead of clustering latent regimes, this
predicts a FORWARD-LOOKING, observable target (see targets.py) - e.g. the next
bar's volatility regime. A thin, point-in-time-honest wrapper around
xgboost.XGBClassifier with class balancing and the metrics we care about.

Note vs gen-1: we do NOT de-seasonalize here. For regime *detection* the intraday
U-shape was contamination to remove; for *forecasting next-bar volatility* the
time-of-day level is genuinely predictive, so we keep the raw features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .. import config


def train_classifier(features_train: pd.DataFrame, target_train: pd.Series,
                     random_state: int = config.RANDOM_STATE) -> XGBClassifier:
    """Fit a gradient-boosted classifier on point-in-time features -> target.
    Classes are balanced so the score is not just the majority bucket."""
    sample_weight = compute_sample_weight("balanced", target_train)
    classifier = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(features_train, target_train, sample_weight=sample_weight)
    return classifier

def evaluate_classifier(classifier, features_test:pd.DataFrame,
                        target_test: pd.Series, class_names: dict):
    """Print the out-of-sample report; return (predictions, macro_fl)."""
    predictions = classifier.predict(features_test)
    ordered_classes = sorted(class_names)
    print(classification_report(
        target_test, predictions, labels=ordered_classes,
        target_names=[class_names[c] for c in ordered_classes], digits=3))
    macro_f1 = f1_score(target_test, predictions, average="macro")
    print(f"macro-F1L {macro_f1:.3f}   (random baseline = {1/len(ordered_classes):.3f})")
    print("confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(target_test, predictions, labels=ordered_classes))
    return predictions, macro_f1

def feature_importance(classifier, feature_names) -> pd.Series:
    """Gain-based importance as a sorted, human-readable Series."""
    return pd.Series(classifier.feature_importances_, index=feature_names
                     ).sort_values(ascending=False)

def _smoke_test() -> None:
    """End-to-endd check: synthetic -> features -> foward target -> train -> eval.
    Run with:  python -m src.model_xgb  (from the repo root)."""
    from ..ingest.synthetic_adapter import SyntheticAdapter
    from ..bars import build_bars
    from ..features import compute_features
    from .targets import make_vol_regime_target, VOL_REGIME_NAMES

    adapter = SyntheticAdapter(symbol="DEMO", n_days=40, seed=7)
    features = compute_features(build_bars(adapter.stream()))
    split_index = int(len(features) * config.TRAIN_RATIO)

    target, _edges, valid_mask = make_vol_regime_target(features, split_index, horizon=1)

    # Train/test by ORIGINAL time order, then keep only rows with a defined target.
    is_train = np.arange(len(features)) < split_index
    train_rows = valid_mask.values & is_train
    test_rows = valid_mask.values & ~is_train

    feature_matrix = features[config.FEATURE_COLS]
    features_train, target_train = feature_matrix[train_rows], target[train_rows].astype(int)
    features_test, target_test = feature_matrix[test_rows], target[test_rows].astype(int)

    classifier = train_classifier(features_train, target_train)
    print(f"train bars: {len(features_train)}   test bars: {len(features_test)}\n")
    print("=== forward vol-regime prediction (out-of-sample) ===")
    evaluate_classifier(classifier, features_test, target_test, VOL_REGIME_NAMES)
    print("\nfeature importance (gain):")
    print(feature_importance(classifier, config.FEATURE_COLS).round(3))

if __name__ == "__main__":
    _smoke_test()
"""
run_xgb_forward.py — end-to-end gen-2 REGIME forecaster.

Synthetic order flow -> bars -> features -> next-bar REGIME target -> XGBoost.
Forecasts which regime (calm / buy_pressure / sell_pressure / volatile) the NEXT
bar will be in. Reports a single time-ordered split and a walk-forward CV, and
saves a diagnostics panel + a predictions-vs-truth timeline.

On synthetic data the labels are the generator's ground-truth regimes; on the real
feed you'd forecast the HMM's next-bar state instead (HMM-as-teacher).

Run:  python run_xgb_forward.py
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src import config
from src.ingest.synthetic_adapter import SyntheticAdapter
from src.bars import build_bars
from src.features import compute_features
from src.xgb.targets import make_regime_forecast_target, REGIME_NAMES
from src.xgb.model_xgb import train_classifier, evaluate_classifier, feature_importance
from src.xgb.walk_forward import cross_validate_forward

# match the HMM's regime colors so the two are visually comparable
REGIME_COLORS = {0: "#aec7e8", 1: "#2ca02c", 2: "#d62728", 3: "#ff7f0e"}


def build_dataset(n_days: int = 40, seed: int = 7, horizon: int = 1):
    """Synthetic feed -> features + next-bar regime target (valid rows only)."""
    adapter = SyntheticAdapter(symbol="DEMO", n_days=n_days, seed=seed)
    features = compute_features(build_bars(adapter.stream()))
    truth = adapter.truth
    split_index = int(len(features) * config.TRAIN_RATIO)

    target, valid_mask = make_regime_forecast_target(features, truth, horizon=horizon)

    is_train_full = np.arange(len(features)) < split_index
    feature_matrix = features.loc[valid_mask, config.FEATURE_COLS]
    target = target[valid_mask].astype(int)
    is_train = is_train_full[valid_mask.values]
    return feature_matrix, target, is_train


def save_diagnostics(target_test, predictions, importance):
    """Static panel: confusion matrix (single split) + feature importance."""
    fig, (ax_cm, ax_imp) = plt.subplots(1, 2, figsize=(14, 5))
    classes = sorted(REGIME_NAMES)
    names = [REGIME_NAMES[c] for c in classes]

    matrix = confusion_matrix(target_test, predictions, labels=classes)
    ax_cm.imshow(matrix, cmap="Blues")
    ax_cm.set_xticks(range(len(names))); ax_cm.set_xticklabels(names, rotation=20, ha="right")
    ax_cm.set_yticks(range(len(names))); ax_cm.set_yticklabels(names)
    ax_cm.set_xlabel("predicted"); ax_cm.set_ylabel("true")
    ax_cm.set_title("Confusion matrix (single split)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax_cm.text(j, i, matrix[i, j], ha="center", va="center",
                       color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    importance.sort_values().plot.barh(ax=ax_imp, color="#1f77b4")
    ax_imp.set_title("Feature importance (gain)"); ax_imp.set_xlabel("importance")

    fig.suptitle("XGBoost regime forecaster — diagnostics")
    plt.tight_layout()
    out = config.FIGURES_DIR / "xgb_regime_diagnostics.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Saved figure -> {out}")


def save_prediction_timeline(target_test, predictions):
    """Two stacked ribbons over the test period: TRUE (top) vs PREDICTED (bottom)
    next-bar regime. Where the colorings differ, the forecast was wrong."""
    truth = np.asarray(target_test)
    pred = np.asarray(predictions)
    x = np.arange(len(truth))

    fig, (ax_true, ax_pred) = plt.subplots(2, 1, figsize=(15, 4.5), sharex=True)
    for ax, labels, title in [(ax_true, truth, "TRUE next-bar regime"),
                              (ax_pred, pred, "PREDICTED next-bar regime")]:
        for cls in sorted(REGIME_NAMES):
            ax.fill_between(x, 0, 1, where=(labels == cls), step="mid",
                            color=REGIME_COLORS[cls], label=REGIME_NAMES[cls])
        ax.set_yticks([]); ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax_pred.set_xlim(0, len(x) - 1)
    ax_pred.set_xlabel("test bar # (time order)")
    fig.suptitle("Regime forecaster — predictions vs. truth over the test period")
    plt.tight_layout()
    out = config.FIGURES_DIR / "xgb_regime_timeline.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Saved figure -> {out}")


def main():
    config.ensure_dirs()
    feature_matrix, target, is_train = build_dataset()
    print(f"dataset: {len(feature_matrix)} usable bars "
          f"(train {int(is_train.sum())}, test {int((~is_train).sum())})")
    print(f"target = next-bar regime, classes = {REGIME_NAMES}\n")

    # --- 1) single time-ordered split: detailed, per-class ---
    classifier = train_classifier(feature_matrix[is_train], target[is_train])
    print("=== single split (out-of-sample) ===")
    predictions, _macro_f1 = evaluate_classifier(
        classifier, feature_matrix[~is_train], target[~is_train], REGIME_NAMES)
    importance = feature_importance(classifier, config.FEATURE_COLS)
    print("\nfeature importance (gain):")
    print(importance.round(3))

    # --- 2) walk-forward CV: robust mean +/- spread ---
    print("\n=== walk-forward cross-validation ===")
    cross_validate_forward(feature_matrix, target, n_folds=5, embargo=0, purge_horizon=1)

    # --- 3) figures ---
    save_diagnostics(target[~is_train], predictions, importance)
    save_prediction_timeline(target[~is_train], predictions)


if __name__ == "__main__":
    main()
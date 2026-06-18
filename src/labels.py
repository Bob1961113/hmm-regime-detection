"""
State interpretation and cross-retrain stability.

HMM state indices are arbitrary: each fit can permute them, and "state 2" today
may be "state 0" after tomorrow's retrain. Two tools here:

1. `label_states` - attach human-readable, microstructure-meaningful names by
   ranking states on their average flow imbalance and realized volatility.
2. `match_states` - Hungarian matching of a new model's state means to a
   reference set, so downstream consumers keep a stable mapping across retrains.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def label_states(features_deseason, states, n_states):
    """Return {state_index: label} using each state's mean signed-volume
    imbalance (net hajar kanan/kiri) and realized volatility.

    features_deseason: DataFrame with at least 'signed_vol_imb' and 'realized_vol'
    states: integer array aligned to features_deseason rows.
    """
    imb = np.array([features_deseason["signed_vol_imb"].values[states == s].mean()
                    for s in range(n_states)])
    vol = np.array([features_deseason["realized_vol"].values[states == s].mean()
                    for s in range(n_states)])

    labels: dict[int, str] = {}
    remaining = list(range(n_states))

    # Most volatile state -> "Volatile / Choppy"
    volatile = int(np.argmax(vol))
    labels[volatile] = "Volatile / Choppy"
    remaining.remove(volatile)

    if remaining:
        buy = max(remaining, key=lambda s: imb[s])
        labels[buy] = "Buy pressure (hajar kanan)"
        remaining.remove(buy)
    if remaining:
        sell = min(remaining, key=lambda s: imb[s])
        labels[sell] = "Sell pressure (hajar kiri)"
        remaining.remove(sell)
    for s in remaining:
        labels[s] = "Quiet / Ranging"
    return labels


def match_states(new_means: np.ndarray, ref_means: np.ndarray) -> np.ndarray:
    """Hungarian-match new states to reference states by mean-vector distance.

    Returns an array `mapping` where mapping[new_index] = ref_index, so a retrained
    model's states can be relabelled to stay consistent with the reference model.
    """
    n = new_means.shape[0]
    cost = np.linalg.norm(new_means[:, None, :] - ref_means[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    mapping = np.empty(n, dtype=int)
    mapping[row] = col
    return mapping

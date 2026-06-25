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


def label_states(features_deseason, states, n_states, imb_thresh: float = 0.10):
    """Return {state_index: label} from each state's mean signed-volume imbalance
    (net hajar kanan/kiri) and realized volatility.

    A state earns a DIRECTION label only if its mean imbalance clears `imb_thresh`
    (signed_vol_imb runs -1..+1, so 0.10 = at least 10% net imbalance). Otherwise
    it is "Quiet / Ranging". This replaces the old rank-based logic, which always
    handed out a Buy and a Sell label even to near-neutral clusters — so a merged,
    direction-less blob could be mislabelled "Sell pressure". Honest labels also
    flag under-fitting: if buy and sell merge (too few states), neither direction
    label appears at all.

    features_deseason: DataFrame with at least 'signed_vol_imb' and 'realized_vol'
    states: integer array aligned to features_deseason rows.
    """
    imb = np.array([features_deseason["signed_vol_imb"].values[states == s].mean()
                    for s in range(n_states)])
    vol = np.array([features_deseason["realized_vol"].values[states == s].mean()
                    for s in range(n_states)])

    volatile = int(np.argmax(vol))          # the most volatile state
    labels: dict[int, str] = {}
    for s in range(n_states):
        if s == volatile:
            labels[s] = "Volatile / Choppy"
        elif imb[s] > imb_thresh:
            labels[s] = "Buy pressure (hajar kanan)"
        elif imb[s] < -imb_thresh:
            labels[s] = "Sell pressure (hajar kiri)"
        else:
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

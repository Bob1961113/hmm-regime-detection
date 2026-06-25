"""
Supervised prediction targets for the gradient boosted (XGBoost) regime model.

The HMM is unsupervised, and the synthetic's generator's ground-truth regimes only 
exist on synthetic data. To train a SUPERVISED model on the real feed - where no 
regime labels exist - we instead predict a FORWARD-LOOKING, observable outcome:
what the *next* bar(s) actually do. This module builds those targets. 

Design rules (the same point-in-time discipline the rest of the pipeline uses):
    * The target is allowed to use the FUTURE - that is the whole point, it is what 
      we predict. The feature vector stays point-in-time (features.compute_features),
      so at inference time t we predict t+1 before it is known. No future leakage.
    * Forward windows never bridge a session boudary (lunch / overnight); they use 
      the same (date, session_id) grouping as the rest of the code. The last
      `horizon` bars of each session have no full forward window -> NaN target.
    * Bucket thresholds are fit on the TRAIN slice only and returned, so the exact 
      same edges can be applied to the test slice without leaking its distribution.

v1 target: next-bar realized-volatility regime (calm / normal / volatility), an
ordinal 3-class label. For a directional target, use forward_return_sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

VOL_REGIME_NAMES = {0: "calm", 1: "normal", 2: "volatile"}
RETURN_SIGN_NAMES = {0: "down", 1: "flat", 2: "up"}

REGIME_NAMES = {0: "calm", 1: "buy_pressure", 2: "sell_pressure", 3: "volatile"}

def make_regime_forecast_target(features: pd.DataFrame, truth: dict, horizon: int = 1):
    """Target = the market regime `horizon` bars AHEAD, within session - forecast
    which regime we'll be in next, not just how volatile.

    On synthetic data, the regime comes from the generator's ground truth (`truth`,
    a dict of bar_start.isoformat() -> regime id). On the real feed there is no
    truth column, so you'd pass the HMM's per-bar states instead (HMM-as-teacher) - 
    identical shape, identical downstream code.

    Returns (target, valid_mask). The last `horizon` bars of each sessiion have no
    forward regime -> NaN -> dropped via valid_mask 
    """
    current_regime = pd.Series(
        [truth.get(ts.isoformat()) for ts in features.index],
        index=features.index, name="regime", dtype="float",
    )
    target = current_regime.groupby(
        [features["date"], features["session_id"]], sort=False).shift(-horizon)
    valid_mask = target.notna()
    return target, valid_mask

def forward_window_mean(features: pd.DataFrame, column:str, horizon: int = 1) -> pd.Series:
    """Mean of `column` over the next `horizon` bars, within each session.

    The shift is done per (date, session_id) group so the forward window never 
    bridges the lunch break or the overnight gap. Returns a Series aligned to 
    `features.index`; the final `horizon` bars of each session are NaN.
    """
    grouped = features.groupby(["date", "session_id"], sort=False)[column]
    forward_steps = [grouped.shift(-step) for step in range(1, horizon + 1)]
    return sum(forward_steps) / horizon

def fit_bucket_edges(forward_values_train: pd.Series, n_buckets: int = 3) -> np.ndarray:
    """Quantile edges that split the TRAIN forward values into `n_buckets` equal
    groups. Returns the interior edges only (length n_bucket - 1), e.g. the
    33rd/66th percentiles for 3 buckets."""
    interior_quantiles = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    return forward_values_train.quantile(interior_quantiles).values

def apply_buckets(forward_values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Assign each forward value to an ordinal bucket using fixed edges.
    np.digitize keeps the order: 0 = lowest band ... n = highest band."""
    buckets = np.digitize(forward_values.values, edges)
    return pd.Series(buckets, index=forward_values.index, name="target")

def make_vol_regime_target(features: pd.DataFrame, split_index: int,
                           horizon: int = 1, n_buckets: int = 3):
    """Build the forward volatility-regime target.
    
    Parameters
    ----------
    features    : output of features.compute_features (has realized_vol + context)
    split_index : row index of the train/test cut; edges are fit on rows < this
    horizon     : how many bars ahead the volatility window looks
    n_buckets   : number of ordinal classes (3 -> calm/normal/volatile)

    Returns
    -------
    target      : ordinal class per bar (NaN where no forward window exists)
    edges       : the train-only bucket edges actually used
    valid_mask  : boolean Series, True where the target is defined (model on these)
    """
    forward_vol = forward_window_mean(features, "realized_vol", horizon=horizon)
    train_forward_vol = forward_vol.iloc[:split_index].dropna()
    edges = fit_bucket_edges(train_forward_vol, n_buckets=n_buckets)

    target = apply_buckets(forward_vol, edges).astype(float)
    target[forward_vol.isna()] = np.nan             # digitize mis-bins NaN; fix it
    valid_mask = target.notna()
    return target, edges, valid_mask

def forward_return_sign(features: pd.DataFrame, horizon: int = 1,
                        flat_threshold: float = 0.0) -> pd.Series:
    """Directional target: sign of the cumulative forward return over the next
    `horizon` bars, within session. 0 = down, 1 = flat, 2 = up.
    
    `flat_threshold` (in log-return units) sets a dead-zone treated as 'flat';
    0.0 collapses it to a pure down/up split. Nan where no forward window exists.
    """
    grouped = features.groupby(["date", "session_id"], sort=False)["ret"]
    forward_return = sum(grouped.shift(-step) for step in range(1, horizon + 1))

    sign = pd.Series(1.0, index=features.index, name="target")   # default = flat
    sign[forward_return > flat_threshold] =  2.0
    sign[forward_return < -flat_threshold] = 0.0
    sign[forward_return.isna()] = np.nan
    return sign

def _smoke_test() -> None:
    """Quick self-check: build synthetic features and summarize the vol target.
    Run with:  python -m src.targets   (from the repo root)."""
    from ..ingest.synthetic_adapter import SyntheticAdapter
    from ..bars import build_bars
    from ..features import compute_features

    adapter = SyntheticAdapter(symbol="DEMO", n_days=10, seed=7)
    features = compute_features(build_bars(adapter.stream()))
    split_index = int(len(features) * config.TRAIN_RATIO)

    target, edges, valid_mask = make_vol_regime_target(features, split_index, horizon=1)
    usable_target = target[valid_mask].astype(int)

    print(f"bars with features       : {len(features)}")
    print(f"bars with a valid target : {int(valid_mask.sum())}  "
          f"(dropped {int((~valid_mask).sum())} session-final bars with no 'next')")
    print(f"train-only bucket edges  : {np.round(edges, 6)}")
    print("class distribution (vol regime):")
    for bucket, count in usable_target.value_counts().sort_index().items():
        print(f"   {bucket} {VOL_REGIME_NAMES[bucket]:9s}: {count}")

    direction = forward_return_sign(features, horizon=1)
    print("class distribution (return sign):")
    for bucket, count in direction.dropna().astype(int).value_counts().sort_index().items():
        print(f"   {bucket} {RETURN_SIGN_NAMES[bucket]:9s}: {count}")


if __name__ == "__main__":
    _smoke_test()
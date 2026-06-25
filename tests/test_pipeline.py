"""Unit tests for the intraday pipeline
    - OFI sign convention (locks in the bug we fixed)
    - point-in-time correctness (no look-ahead)
    - session-boundary resets
Run: python -m pytest tests/ -v
"""
import pandas as pd
import pytest

from src import config
from src.ingest.synthetic_adapter import SyntheticAdapter
from src.bars import build_bars, _ofi_increment
from src.features import compute_features



# A fixture: builts once, shared by every test that asks for `synthetic_bars`.
@pytest.fixture(scope="module")
def synthetic_bars():
    adapter = SyntheticAdapter(n_days=5, seed=1)
    events = list(adapter.stream())
    return build_bars(iter(events))

# small helper to build a top-of-book quote dict
def _quote(bid, ask, bid_size, ask_size):
    return {"bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size}

# ---------- OFI sign convention (the bug we fixed must stay fixed) ----------
def test_ofi_bid_price_up_is_positive():
    # bid lifted -> buy pressure -> OFI > 0
    assert _ofi_increment(_quote(100, 102, 10, 8), _quote(101, 102, 5, 8)) > 0

def test_ofi_bid_price_down_is_negative():
    # bid pulled -> selling -> OFI < 0
    assert _ofi_increment(_quote(100, 102, 10, 8), _quote(99, 102, 7, 8)) < 0

def test_ofi_ask_price_up_is_positive():
    # ask retreats -> sellers stepping back -> bullish (this is the case that was buggy)
    assert _ofi_increment(_quote(100, 102, 10, 8), _quote(100, 103, 10, 4)) > 0

def test_ofi_ask_price_down_is_negative():
    # ask pushed down -> aggressive selling -> OFI < 0
    assert _ofi_increment(_quote(100, 102, 10, 8), _quote(100, 101, 10, 6)) < 0

def test_ofi_same_price_uses_size_delta():
    # prices unchanged, bid grows + ask shrinks -> buy pressure
    # e_b = 15-10 = 5 ; e_a = 5-8 = -3, OFI = 5 - (-3) = 8
    assert _ofi_increment(_quote(100, 102, 10, 8), _quote(100, 102, 15, 5)) == pytest.approx(8.0)


# ---------- Point-in-time: features must not depend on the future ----------------
def test_features_do_not_depend_on_future(synthetic_bars):
    feats_full = compute_features(synthetic_bars)
    k = len(synthetic_bars) // 2
    feats_trunc = compute_features(synthetic_bars.iloc[:k])     # delete the future
    common = feats_trunc.index
    # the surviving past rows must be byte-for-byte identical
    pd.testing.assert_frame_equal(
        feats_trunc[config.FEATURE_COLS],
        feats_full.loc[common, config.FEATURE_COLS],
    )

# ------------ Session resets: returns never bridge a session boundary --------------
def test_first_bar_of_each_session_is_dropped(synthetic_bars):
    feats = compute_features(synthetic_bars)
    firsts = synthetic_bars.reset_index().groupby(["date", "session_id"])["bar_start"].min()
    for first_bar in firsts:
        # the first bar of each session has no valid return -> must be dropped
        assert first_bar not in feats.index

    
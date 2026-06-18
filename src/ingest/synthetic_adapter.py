"""
Synthetic order-flow generator (a FeedAdapter).

Purpose: make the ENTIRE pipeline runnable and testable before the real
proprietary feed exists. It simulates a hidden, sticky regime chain and emits
canonical quote + aggressor-trade events with realistic intraday seasonality.

Because we know the ground-truth regime behind each bar, we can later check that
the unsupervised HMM actually recovers something meaningful (see backtest).

Ground-truth regimes:
    0 calm          - low vol, balanced flow, ~no drift
    1 buy_pressure  - positive drift, buy-heavy aggressor (net hajar kanan)
    2 sell_pressure - negative drift, sell-heavy aggressor (net hajar kiri)
    3 volatile      - high vol, wide spread, two-sided
"""
from __future__ import annotations

from datetime import datetime, timedelta, date as date_cls
from typing import Iterator

import numpy as np

from .. import config
from .base import FeedAdapter, QUOTE, TRADE, AGGRESSOR_BUY, AGGRESSOR_SELL

# Per-tick regime parameters (drift/vol are per intra-bar tick, they accumulate).
REGIMES = {
    0: dict(name="calm",          drift=0.0,     vol=0.0006, intensity=0.7, spread_mult=1.0, p_buy=0.50),
    1: dict(name="buy_pressure",  drift=+0.0004, vol=0.0010, intensity=1.3, spread_mult=1.1, p_buy=0.68),
    2: dict(name="sell_pressure", drift=-0.0004, vol=0.0010, intensity=1.3, spread_mult=1.1, p_buy=0.32),
    3: dict(name="volatile",      drift=0.0,     vol=0.0024, intensity=1.6, spread_mult=2.4, p_buy=0.50),
}

# Sticky transition matrix so regimes persist over several bars (realistic dwell).
TRANSITION = np.array([
    [0.90, 0.04, 0.04, 0.02],
    [0.06, 0.88, 0.02, 0.04],
    [0.06, 0.02, 0.88, 0.04],
    [0.10, 0.05, 0.05, 0.80],
])

BASE_DEPTH = 1000   # base top-of-book size
BASE_TICKS = 25     # mean intra-bar quote updates at neutral seasonality

# Instrument "profiles": the same regime machinery, different market character.
# DEMO keeps the original behaviour (so run_demo.py stays reproducible). BBCA is
# tuned to resemble a blue chip: high price, Rp25 tick, low vol, very liquid.
PROFILES = {
    "DEMO": dict(base_price=5000.0, tick=1.0, vol_scale=1.0, drift_scale=1.0,
                 spread_frac=0.0006, intensity_scale=1.0, mean_revert=0.02),
    # Blue chip: high price, Rp25 tick, low vol, gentle trends, anchored near base.
    # (intensity is kept modest because in this generator more ticks => more drift;
    # cranking it makes a calm stock explode, which is not BBCA-like.)
    "BBCA": dict(base_price=9500.0, tick=25.0, vol_scale=0.45, drift_scale=0.30,
                 spread_frac=0.0006, intensity_scale=1.2, mean_revert=0.05),
}


def _bar_starts(day: date_cls) -> list[tuple[datetime, int]]:
    """Enumerate (bar_start, session_index) for one trading day from config."""
    out: list[tuple[datetime, int]] = []
    for sess_idx, (start_s, end_s) in enumerate(config.sessions_for(day.weekday())):
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        start = datetime(day.year, day.month, day.day, sh, sm, tzinfo=config.TZ)
        end = datetime(day.year, day.month, day.day, eh, em, tzinfo=config.TZ)
        t = start
        while t < end:
            out.append((t, sess_idx))
            t += timedelta(minutes=config.BAR_MINUTES)
    return out


class SyntheticAdapter(FeedAdapter):
    def __init__(self, symbol: str = config.DEFAULT_SYMBOL, n_days: int = 30,
                 start_date: date_cls | None = None, seed: int = 7,
                 base_price: float = 5000.0, tick: float = 1.0,
                 vol_scale: float = 1.0, drift_scale: float = 1.0,
                 spread_frac: float = 0.0006, intensity_scale: float = 1.0,
                 mean_revert: float = 0.02):
        self.symbol = symbol
        self.n_days = n_days
        self.start_date = start_date or date_cls(2026, 1, 5)  # a Monday
        self.rng = np.random.default_rng(seed)
        # Instrument character (see PROFILES for presets)
        self.base_price = base_price
        self.tick = tick
        self.vol_scale = vol_scale
        self.drift_scale = drift_scale
        self.spread_frac = spread_frac
        self.intensity_scale = intensity_scale
        self.mean_revert = mean_revert
        # Ground truth, filled in during stream(): bar_start ISO -> regime id.
        self.truth: dict[str, int] = {}

    @classmethod
    def from_profile(cls, name: str, symbol: str | None = None, **kwargs):
        """Build an adapter from a named PROFILES preset (e.g. "BBCA")."""
        return cls(symbol=symbol or name, **PROFILES[name], **kwargs)

    def _round(self, price: float) -> float:
        return round(price / self.tick) * self.tick

    def _trading_days(self) -> list[date_cls]:
        days, d = [], self.start_date
        while len(days) < self.n_days:
            if d.weekday() < 5:           # Mon-Fri only
                days.append(d)
            d += timedelta(days=1)
        return days

    def stream(self) -> Iterator[dict]:
        regime = 0
        mid = self.base_price
        for day in self._trading_days():
            bars = _bar_starts(day)
            n_bars = len(bars)
            for i, (bar_start, _sess) in enumerate(bars):
                # advance the hidden regime chain
                regime = int(self.rng.choice(4, p=TRANSITION[regime]))
                self.truth[bar_start.isoformat()] = regime
                p = REGIMES[regime]

                # intraday U-shape: busy/wide at open & close, quiet midday
                f = i / max(1, n_bars - 1)
                u = 0.6 + 1.4 * (2 * f - 1) ** 2
                n_ticks = max(3, int(self.rng.poisson(
                    BASE_TICKS * p["intensity"] * self.intensity_scale * u)))
                dt = timedelta(minutes=config.BAR_MINUTES) / n_ticks

                for k in range(n_ticks):
                    ts = bar_start + dt * k
                    # mid-price random walk (drift/vol scaled by the instrument profile)
                    mid *= float(np.exp(p["drift"] * self.drift_scale
                                        + p["vol"] * self.vol_scale * self.rng.standard_normal()))
                    spread = max(self.tick, self._round(
                        mid * self.spread_frac * p["spread_mult"] * (0.8 + 0.4 * u)))
                    bid = self._round(mid - spread / 2)
                    ask = self._round(mid + spread / 2)
                    # depth: tilt bid heavier under buy pressure (drives positive OFI)
                    tilt = (p["p_buy"] - 0.5) * 2.0
                    bid_size = max(1, int(BASE_DEPTH * (1 + 0.4 * tilt) * (0.5 + self.rng.random())))
                    ask_size = max(1, int(BASE_DEPTH * (1 - 0.4 * tilt) * (0.5 + self.rng.random())))
                    yield {
                        "ts": ts.isoformat(), "type": QUOTE, "symbol": self.symbol,
                        "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size,
                    }
                    # a trade may print against the book
                    if self.rng.random() < 0.6:
                        is_buy = self.rng.random() < p["p_buy"]
                        price = ask if is_buy else bid
                        size = int(np.ceil(self.rng.lognormal(mean=4.0, sigma=0.8)))
                        yield {
                            "ts": (ts + dt / 2).isoformat(), "type": TRADE, "symbol": self.symbol,
                            "price": price, "size": size,
                            "aggressor": AGGRESSOR_BUY if is_buy else AGGRESSOR_SELL,
                        }
            # small overnight pull toward base to avoid runaway prices (gap-ish)
            mid = (1 - self.mean_revert) * mid + self.mean_revert * self.base_price

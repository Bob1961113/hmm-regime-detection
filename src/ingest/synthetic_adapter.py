"""
Synthetic order-flow generator (a FeedAdapter).

Purpose: make the ENTIRE pipeline runnable and testable before the real
proprietary feed exists. It simulates a hidden, sticky regime chain and emits
canonical quote + aggressor-trade events with realistic intraday microstructure.

Because we know the ground-truth regime behind each bar, we can later check that
the unsupervised HMM actually recovers something meaningful (see backtest).

Ground-truth regimes:
    0 calm          - low vol, balanced flow, ~no drift
    1 buy_pressure  - positive drift, buy-heavy aggressor (net hajar kanan)
    2 sell_pressure - negative drift, sell-heavy aggressor (net hajar kiri)
    3 volatile      - high vol, wide spread, two-sided, jump-prone

Realism choices (tuned so the synthetic data is a *meaningful* test, not a toy):
  * Per-BAR drift, spread across the bar's ticks, so a bar's expected move does
    NOT depend on how busy it is (busy != directional).
  * Strong, SEPARATE intraday U-shapes for activity, volatility and spread, so
    realized_vol / rel_spread carry genuine time-of-day signal that the model
    must de-seasonalize away (the #1 intraday pitfall).
  * Fat-tailed (Student-t) innovations + occasional jumps, so returns are not
    Gaussian and the RobustScaler / volatile regime have something to do.
  * Per-bar idiosyncratic jitter on drift and flow, so regimes OVERLAP and are
    not trivially separable (keeps ARI, detection lag and filter-vs-smoother
    disagreement in a realistic, non-perfect range).
"""
from __future__ import annotations

from datetime import datetime, timedelta, date as date_cls
from typing import Iterator

import numpy as np

from .. import config
from .base import FeedAdapter, QUOTE, TRADE, AGGRESSOR_BUY, AGGRESSOR_SELL

# Regime parameters. `drift` is now PER BAR (spread across the bar's ticks);
# `vol` is per-tick volatility; `jump_p` is the per-tick probability of a jump.
REGIMES = {
    0: dict(name="calm",          drift=0.0,     vol=0.0004, intensity=0.6, spread_mult=0.9, p_buy=0.50, jump_p=0.000),
    1: dict(name="buy_pressure",  drift=+0.0028, vol=0.0008, intensity=1.2, spread_mult=1.2, p_buy=0.68, jump_p=0.000),
    2: dict(name="sell_pressure", drift=-0.0028, vol=0.0008, intensity=1.2, spread_mult=1.2, p_buy=0.32, jump_p=0.000),
    3: dict(name="volatile",      drift=0.0,     vol=0.0013, intensity=1.5, spread_mult=2.4, p_buy=0.50, jump_p=0.012),
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


def _u_shapes(f: float) -> tuple[float, float, float]:
    """Three intraday U-shapes as a function of session progress f in [0, 1]
    (0 = open, 1 = close). `bow` is 1 at the edges and 0 at midday.
    Returns (activity, volatility, spread) multipliers.

    Volatility and spread get STRONG U-shapes on purpose: that is the time-of-day
    contamination the de-seasonalization step exists to remove.
    """
    bow = (2.0 * f - 1.0) ** 2
    activity = 0.5 + 1.7 * bow       # 0.5 .. 2.2
    volatility = 0.7 + 1.3 * bow     # 0.7 .. 2.0  (strong)
    spread = 0.7 + 1.8 * bow         # 0.7 .. 2.5  (strong)
    return activity, volatility, spread


# Instrument "profiles": the same regime machinery, different market character.
PROFILES = {
    "DEMO": dict(base_price=5000.0, tick=1.0, vol_scale=1.0, drift_scale=1.0,
                 spread_frac=0.0006, intensity_scale=1.0, mean_revert=0.04),
    # Blue chip: high price, Rp25 tick, lower vol, gentler trends. Drift is now
    # decoupled from activity, so a higher intensity no longer makes it explode.
    "BBCA": dict(base_price=9500.0, tick=25.0, vol_scale=0.55, drift_scale=0.55,
                 spread_frac=0.0006, intensity_scale=1.3, mean_revert=0.06),
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
                 mean_revert: float = 0.04, regime_noise: float = 1.0,
                 tail_df: float = 6.0):
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
        # Realism knobs
        self.regime_noise = regime_noise     # 0 = clean regimes, 1 = realistic overlap
        self.tail_df = tail_df               # Student-t dof: lower = fatter tails
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
        t_norm = float(np.sqrt(self.tail_df / (self.tail_df - 2.0)))   # unit-variance Student-t
        for day in self._trading_days():
            bars = _bar_starts(day)
            n_bars = len(bars)
            for i, (bar_start, _sess) in enumerate(bars):
                # advance the hidden regime chain
                regime = int(self.rng.choice(4, p=TRANSITION[regime]))
                self.truth[bar_start.isoformat()] = regime
                p = REGIMES[regime]

                f = i / max(1, n_bars - 1)
                season_act, season_vol, season_spd = _u_shapes(f)

                # Per-bar idiosyncratic jitter: regimes OVERLAP, not perfectly
                # separable. This is what keeps the problem realistically hard.
                nz = self.regime_noise
                drift_bar = (p["drift"] + self.rng.normal(0.0, 0.0005 * nz)) * self.drift_scale
                vol_bar = max(1e-5, p["vol"] * self.vol_scale
                              * (1.0 + 0.18 * nz * self.rng.standard_normal()))
                p_buy = float(np.clip(p["p_buy"] + self.rng.normal(0.0, 0.02 * nz), 0.05, 0.95))

                n_ticks = max(3, int(self.rng.poisson(
                    BASE_TICKS * p["intensity"] * self.intensity_scale * season_act)))
                dt = timedelta(minutes=config.BAR_MINUTES) / n_ticks
                drift_tick = drift_bar / n_ticks          # spread the bar's drift over its ticks

                for k in range(n_ticks):
                    ts = bar_start + dt * k
                    # fat-tailed (Student-t) unit-variance innovation, scaled by
                    # per-tick vol and the volatility U-shape
                    z = float(self.rng.standard_t(self.tail_df)) / t_norm
                    shock = vol_bar * season_vol * z
                    if self.rng.random() < p["jump_p"]:               # occasional jump
                        shock += vol_bar * season_vol * float(self.rng.normal(0.0, 4.0))
                    mid *= float(np.exp(drift_tick + shock))

                    spread = max(self.tick, self._round(
                        mid * self.spread_frac * p["spread_mult"] * season_spd))
                    bid = self._round(mid - spread / 2)
                    ask = self._round(mid + spread / 2)
                    # depth: tilt bid heavier under buy pressure (drives positive OFI)
                    tilt = (p_buy - 0.5) * 2.0
                    bid_size = max(1, int(BASE_DEPTH * (1 + 0.6 * tilt) * (0.5 + self.rng.random())))
                    ask_size = max(1, int(BASE_DEPTH * (1 - 0.6 * tilt) * (0.5 + self.rng.random())))
                    yield {
                        "ts": ts.isoformat(), "type": QUOTE, "symbol": self.symbol,
                        "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size,
                    }
                    # a trade may print against the book
                    if self.rng.random() < 0.6:
                        is_buy = self.rng.random() < p_buy
                        price = ask if is_buy else bid
                        size = int(np.ceil(self.rng.lognormal(mean=4.0, sigma=0.8)))
                        yield {
                            "ts": (ts + dt / 2).isoformat(), "type": TRADE, "symbol": self.symbol,
                            "price": price, "size": size,
                            "aggressor": AGGRESSOR_BUY if is_buy else AGGRESSOR_SELL,
                        }
            # small overnight pull toward base to avoid runaway prices (gap-ish)
            mid = (1 - self.mean_revert) * mid + self.mean_revert * self.base_price

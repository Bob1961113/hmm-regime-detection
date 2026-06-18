"""
Aggregate canonical events into session-aware 10-minute "rich bars".

A rich bar holds everything needed to derive model features later:
mid OHLC, volume split into buy/sell aggressor (hajar kanan/kiri), order-flow
imbalance, average spread, realized intra-bar volatility, trade count, and the
seasonality slot.

Session awareness is critical: order-flow imbalance is a *difference between
consecutive quotes*, so it must NOT be computed across the lunch break or the
overnight gap. We reset the previous-quote reference whenever the session changes.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config
from .ingest.base import QUOTE, TRADE, AGGRESSOR_BUY


def _classify(ts: datetime) -> Optional[tuple[datetime, int, int]]:
    """Map a timestamp to (bar_start, session_index, bar_in_session), or None
    if it falls outside any configured trading session."""
    for sess_idx, (start_s, end_s) in enumerate(config.sessions_for(ts.weekday())):
        sh, sm = map(int, start_s.split(":"))
        eh, em = map(int, end_s.split(":"))
        start = ts.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = ts.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start <= ts < end:
            bar_in_session = int((ts - start).total_seconds() // 60 // config.BAR_MINUTES)
            bar_start = start + timedelta(minutes=bar_in_session * config.BAR_MINUTES)
            return bar_start, sess_idx, bar_in_session
    return None


def _ofi_increment(prev: dict, cur: dict) -> float:
    """Order-flow imbalance contribution between two consecutive quotes
    (Cont, Kukanov & Stoikov). Positive = bid side strengthening = buy pressure."""
    pb0, pb1 = prev["bid"], cur["bid"]
    qb0, qb1 = prev["bid_size"], cur["bid_size"]
    pa0, pa1 = prev["ask"], cur["ask"]
    qa0, qa1 = prev["ask_size"], cur["ask_size"]

    if pb1 > pb0:
        e_b = qb1
    elif pb1 == pb0:
        e_b = qb1 - qb0
    else:
        e_b = -qb0

    if pa1 > pa0:
        e_a = -qa0
    elif pa1 == pa0:
        e_a = qa1 - qa0
    else:
        e_a = qa1

    return float(e_b - e_a)


def build_bars(events: Iterable[dict]) -> pd.DataFrame:
    """Consume canonical events (time-ordered) and return a rich-bar DataFrame
    indexed by bar_start (tz-aware)."""
    bars: dict[datetime, dict] = {}
    prev_quote: Optional[dict] = None
    prev_session_key = None

    for ev in events:
        ts = datetime.fromisoformat(ev["ts"])
        info = _classify(ts)
        if info is None:
            continue
        bar_start, sess_idx, bar_in_session = info

        session_key = (bar_start.date(), sess_idx)
        if session_key != prev_session_key:
            prev_quote = None                      # reset OFI across session boundary
            prev_session_key = session_key

        b = bars.get(bar_start)
        if b is None:
            b = {
                "session_id": sess_idx,
                "bar_in_session": bar_in_session,
                "slot": bar_start.strftime("%H:%M"),
                "date": bar_start.date().isoformat(),
                "mids": [], "spreads": [],
                "volume": 0, "buy_vol": 0, "sell_vol": 0, "n_trades": 0,
                "turnover": 0.0, "ofi": 0.0,
            }
            bars[bar_start] = b

        if ev["type"] == QUOTE:
            mid = (ev["bid"] + ev["ask"]) / 2.0
            b["mids"].append(mid)
            b["spreads"].append(ev["ask"] - ev["bid"])
            if prev_quote is not None:
                b["ofi"] += _ofi_increment(prev_quote, ev)
            prev_quote = ev
        elif ev["type"] == TRADE:
            size = ev["size"]
            b["volume"] += size
            b["n_trades"] += 1
            b["turnover"] += ev["price"] * size
            if ev["aggressor"] == AGGRESSOR_BUY:
                b["buy_vol"] += size
            else:
                b["sell_vol"] += size

    if not bars:
        return pd.DataFrame()

    rows = []
    for bar_start in sorted(bars):
        b = bars[bar_start]
        mids = np.asarray(b["mids"], dtype=float)
        if len(mids) == 0:
            continue
        if len(mids) >= 2:
            log_ret = np.diff(np.log(mids))
            realized_vol = float(np.std(log_ret))
        else:
            realized_vol = 0.0
        volume = b["volume"]
        rows.append({
            "bar_start": bar_start,
            "open": mids[0], "high": mids.max(), "low": mids.min(), "close": mids[-1],
            "vwap": (b["turnover"] / volume) if volume else mids[-1],
            "volume": volume, "buy_vol": b["buy_vol"], "sell_vol": b["sell_vol"],
            "signed_vol": b["buy_vol"] - b["sell_vol"],
            "n_trades": b["n_trades"],
            "ofi": b["ofi"],
            "avg_spread": float(np.mean(b["spreads"])) if b["spreads"] else 0.0,
            "realized_vol": realized_vol,
            "session_id": b["session_id"],
            "bar_in_session": b["bar_in_session"],
            "slot": b["slot"],
            "date": b["date"],
        })

    df = pd.DataFrame(rows).set_index("bar_start").sort_index()
    return df

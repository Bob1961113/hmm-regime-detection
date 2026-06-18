"""
Ingestion interface + canonical market-event schema.

The live data contract ("ngehit" a REST/websocket link) is still unknown, so we
hide every source behind one small interface, `FeedAdapter`. Each adapter is
responsible for mapping whatever the source sends into ONE canonical event shape
defined below. Everything downstream (bars -> features -> model) depends only on
the canonical shape, so swapping the source (synthetic -> real feed) touches a
single file.

Canonical event (a plain dict, JSON-serialisable so it can be recorded as-is):

    {
        "ts":      "2026-06-15T09:00:03+07:00",  # ISO8601, tz-aware (Asia/Jakarta)
        "type":    "quote" | "trade",
        "symbol":  "DEMO",

        # quote (top of book) fields:
        "bid": 5000.0, "ask": 5005.0, "bid_size": 1200, "ask_size": 900,

        # trade fields:
        "price": 5005.0, "size": 100, "aggressor": "buy" | "sell",
    }

`aggressor == "buy"`  -> buyer lifted the offer  -> *hajar kanan*
`aggressor == "sell"` -> seller hit the bid      -> *hajar kiri*
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

# Event type tags
QUOTE = "quote"
TRADE = "trade"

# Aggressor side tags (the trader-slang mapping)
AGGRESSOR_BUY = "buy"     # hajar kanan
AGGRESSOR_SELL = "sell"   # hajar kiri


class FeedAdapter(ABC):
    """A source of canonical market events.

    Concrete adapters: SyntheticAdapter (now), ReplayAdapter (recorded files),
    and later a WebsocketAdapter / RestPollAdapter for the real feed.
    """

    @abstractmethod
    def stream(self) -> Iterator[dict]:
        """Yield canonical event dicts in non-decreasing timestamp order."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources (sockets, files). Default: nothing to do."""
        return None

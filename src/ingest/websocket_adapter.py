"""
Live feed adapter — STUB for Phase 5 (blocked on the Phase 0 data contract).

When the team lead provides the endpoint ("ngehit" a link), this is the ONLY file
that should need real changes: implement `_raw_stream` for the actual protocol and
`normalize` for the actual payload schema. Everything downstream already depends
only on the canonical event shape defined in `base.py`.

Both connected libraries are already installed: `websocket-client` / `websockets`
(for a push websocket) and `requests` (for a REST poll). Pick whichever the feed
uses; delete the other path.

Open questions to confirm before filling this in (Phase 0):
  - Protocol: websocket subscribe, or REST poll? URL + auth (token/header)?
  - Payload schema: field names/units for bid/ask/sizes, trade price/size, and how
    the aggressor side (hajar kanan/kiri) is encoded.
  - Heartbeats / reconnection / sequence numbers for gap detection.
  - How halts and ARA/ARB (auto-reject) states appear in the stream.
"""
from __future__ import annotations

import json
from typing import Iterator

from .base import FeedAdapter, QUOTE, TRADE, AGGRESSOR_BUY, AGGRESSOR_SELL


class WebsocketAdapter(FeedAdapter):
    def __init__(self, url: str, symbol: str, auth: dict | None = None):
        self.url = url
        self.symbol = symbol
        self.auth = auth or {}
        self._conn = None

    def normalize(self, raw: dict) -> dict | None:
        """Map ONE raw feed message to the canonical event shape (see base.py).
        TODO(Phase 0): fill in once the real schema is known. Example skeleton:

            if raw["msgType"] == "ORDERBOOK":
                return {"ts": raw["time"], "type": QUOTE, "symbol": self.symbol,
                        "bid": raw["bidPrice"], "ask": raw["askPrice"],
                        "bid_size": raw["bidVol"], "ask_size": raw["askVol"]}
            if raw["msgType"] == "TRADE":
                side = AGGRESSOR_BUY if raw["side"] == "B" else AGGRESSOR_SELL
                return {"ts": raw["time"], "type": TRADE, "symbol": self.symbol,
                        "price": raw["price"], "size": raw["qty"], "aggressor": side}
        """
        raise NotImplementedError("Implement once the feed schema is confirmed (Phase 0).")

    def _raw_stream(self) -> Iterator[dict]:
        """Yield raw messages from the live source.
        TODO(Phase 0): implement the real connection. Websocket skeleton:

            from websocket import create_connection
            self._conn = create_connection(self.url, header=self.auth)
            self._conn.send(json.dumps({"subscribe": self.symbol}))
            while True:
                yield json.loads(self._conn.recv())
        """
        raise NotImplementedError("Implement once the protocol is confirmed (Phase 0).")

    def stream(self) -> Iterator[dict]:
        for raw in self._raw_stream():
            event = self.normalize(raw)
            if event is not None:
                yield event

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

"""
Recorder: persist a feed to disk as JSONL (one canonical event per line).

This is the bridge that turns a live feed into a backtest dataset. The moment we
get access to the real endpoint, the very first job is to run the recorder so we
start accumulating history — the proprietary data almost certainly has no stored
past we can query.

JSONL is used (not Parquet) so there is no extra dependency, the file is append-
friendly, human-readable, and survives a mid-session crash (every flushed line is
valid).
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import FeedAdapter


def record(adapter: FeedAdapter, path: str | Path, max_events: int | None = None,
           mode: str = "w") -> int:
    """Stream `adapter` to `path` (JSONL). Returns the number of events written.

    mode="w" (default) truncates — correct for the reproducible synthetic demo,
    where each run should regenerate the same file. For LIVE recording pass
    mode="a" (append) or use a dated path per session, otherwise restarting the
    recorder mid-day would silently wipe everything captured so far.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with path.open(mode, encoding="utf-8") as f:
            for event in adapter.stream():
                f.write(json.dumps(event, separators=(",", ":")))
                f.write("\n")
                n += 1
                if max_events is not None and n >= max_events:
                    break
                f.flush()           # every line is durable -> crash-survivable
    finally:
        adapter.close()
    return n

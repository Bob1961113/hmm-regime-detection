"""
Replay: read recorded JSONL back as a FeedAdapter.

This is what makes backtests deterministic and reproducible — develop and test
the model against a fixed recording instead of a live, never-repeating feed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import FeedAdapter


class ReplayAdapter(FeedAdapter):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def stream(self) -> Iterator[dict]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

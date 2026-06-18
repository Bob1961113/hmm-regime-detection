"""
Persist regime outputs for monitoring and later analysis.

CSV + SQLite only (both standard library / pandas built-ins) so there is no extra
dependency. Swap for TimescaleDB/Postgres later only if data volume demands it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config


def save_csv(df: pd.DataFrame, filename: str = "backtest_regimes.csv") -> Path:
    config.ensure_dirs()
    out = config.PROCESSED_DIR / filename
    df.to_csv(out, index_label="bar_start")
    print(f"Saved regimes CSV -> {out}")
    return out


def save_sqlite(df: pd.DataFrame, table: str = "regimes",
                filename: str = "regimes.sqlite") -> Path:
    config.ensure_dirs()
    out = config.PROCESSED_DIR / filename
    with sqlite3.connect(out) as conn:
        df.reset_index().to_sql(table, conn, if_exists="replace", index=False)
    print(f"Saved regimes SQLite -> {out} (table '{table}')")
    return out

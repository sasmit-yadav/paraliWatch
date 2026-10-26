"""Which season's data to serve: the live season (if built) and the 2025 replay.

The replay is the scored, held-out test season shipped with the repo. The live season is
rebuilt from today's satellite data by parali.live; a new Store is created after each rebuild,
so every cache inside the old one is dropped with it.
"""

from __future__ import annotations

import threading
from datetime import date

from parali import config, live
from parali.app.store import Store

REPLAY = Store()
_lock = threading.Lock()
_live: dict = {"built_at": None, "store": None}


def live_store() -> Store | None:
    meta = live.meta()
    if not meta or not (config.LIVE_DIR / f"forecast_{meta['season']}.parquet").exists():
        return None
    with _lock:
        if _live["built_at"] != meta["built_at"]:
            _live["store"] = Store(demo_dir=config.LIVE_DIR, year=meta["season"], live=True)
            _live["built_at"] = meta["built_at"]
        return _live["store"]


def get(season: int | None = None) -> Store:
    """season=None: the live season when available, else the replay."""
    current = live_store()
    if season is None:
        return current or REPLAY
    if season == REPLAY.year:
        return REPLAY
    if current is not None and season == current.year:
        return current
    raise KeyError(season)


def for_date(d: date) -> Store:
    current = live_store()
    if current is not None and d in set(current.dates):
        return current
    return REPLAY


def available() -> list[dict]:
    out = []
    current = live_store()
    if current is not None:
        out.append({"season": current.year, "mode": "live"})
    out.append({"season": REPLAY.year, "mode": "replay"})
    return out

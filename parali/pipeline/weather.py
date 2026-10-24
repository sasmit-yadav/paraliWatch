"""Daily weather per cell from Open-Meteo's ERA5 archive (free, no key).

Cells are snapped to a 0.5 degree point grid (ERA5 itself is 0.25 degrees) and every cell reuses its
point's values. That cuts ~2,500 cells to ~50 locations and stays inside the free API limits.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from parali import config
from parali.pipeline.download import get

log = logging.getLogger(__name__)
BATCH = 50  # locations per request


def snap(values: pd.Series) -> pd.Series:
    step = config.WEATHER_GRID_DEG
    return (np.round(values / step) * step).round(4)


def weather_points(grid: pd.DataFrame) -> pd.DataFrame:
    pts = grid.assign(wlat=snap(grid["lat"]), wlon=snap(grid["lon"]))
    return pts


def fetch_season(points: pd.DataFrame, year: int) -> pd.DataFrame:
    """Daily weather for each unique (wlat, wlon) point over one season."""
    start, end = config.season_bounds(year)
    uniq = points[["wlat", "wlon"]].drop_duplicates().reset_index(drop=True)
    frames = []
    for i in range(0, len(uniq), BATCH):
        batch = uniq.iloc[i : i + BATCH]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in batch["wlat"]),
            "longitude": ",".join(f"{v:.4f}" for v in batch["wlon"]),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(config.WEATHER_DAILY_VARS),
            "timezone": "Asia/Kolkata",
        }
        payload = get(config.OPEN_METEO_ARCHIVE_URL, params=params).json()
        if isinstance(payload, dict):
            payload = [payload]
        for (wlat, wlon), loc in zip(batch[["wlat", "wlon"]].itertuples(index=False), payload, strict=True):
            daily = pd.DataFrame(loc["daily"])
            daily["wlat"], daily["wlon"] = wlat, wlon
            frames.append(daily)
        time.sleep(5.0)  # stay well inside the free tier
    out = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    log.info("Weather %s: %d point-days", year, len(out))
    return out

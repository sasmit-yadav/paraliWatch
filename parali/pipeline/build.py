"""Build every processed dataset from scratch. Each step caches its output, so a rerun only
does missing work. Run with: python -m parali data"""

from __future__ import annotations

import json
import logging

import pandas as pd

from parali import config
from parali.pipeline import firms, weather
from parali.pipeline.boundaries import build_districts
from parali.pipeline.grid import build_grid

log = logging.getLogger(__name__)

GRID = config.PROCESSED_DIR / "grid.parquet"
DISTRICTS = config.DEMO_DIR / "districts.geojson"


def step_grid(force: bool = False) -> pd.DataFrame:
    if GRID.exists() and DISTRICTS.exists() and not force:
        return pd.read_parquet(GRID)
    districts = build_districts()
    DISTRICTS.write_text(json.dumps(districts, separators=(",", ":")), encoding="utf-8")
    grid = build_grid(districts)
    grid.to_parquet(GRID, index=False)
    log.info("Grid: %d cells across %d districts", len(grid), grid["district"].nunique())
    return grid


def fires_path(year: int):
    return config.PROCESSED_DIR / "fires" / f"{year}.parquet"


def step_fires(years=config.YEARS, force: bool = False) -> None:
    cells = set(step_grid()["cell"])
    for year in years:
        out = fires_path(year)
        if out.exists() and not force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        firms.load_archive_year(year, cells).to_parquet(out, index=False)


def weather_path(year: int):
    return config.PROCESSED_DIR / "weather" / f"{year}.parquet"


def step_weather(years=config.YEARS, force: bool = False) -> None:
    points = weather.weather_points(step_grid())
    for year in years:
        out = weather_path(year)
        if out.exists() and not force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        weather.fetch_season(points, year).to_parquet(out, index=False)


def load_fires(years=config.YEARS) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(fires_path(y)) for y in years], ignore_index=True)


def load_weather(years=config.YEARS) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(weather_path(y)) for y in years], ignore_index=True)


def run(force: bool = False) -> None:
    config.ensure_dirs()
    step_grid(force)
    step_fires(force=force)
    step_weather(force=force)

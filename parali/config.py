"""Project-wide settings. Everything tunable lives here so experiments are easy to trace."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("PARALI_CACHE") or DATA_DIR / "raw")
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "eval"

# Study area
STATES = ("Punjab", "Haryana")
# Loose box around both states (west, south, east, north); polygons do the exact cut.
BBOX = (73.8, 27.6, 77.7, 32.6)

# Grid: H3 resolution 6 cells average ~36 km2 (edge ~3.7 km).
H3_RES = 6

# Seasons. Paddy residue burning peaks Oct-Nov; we keep a margin on both sides.
YEARS = list(range(2019, 2026))  # 2019 only seeds "last year" features
SEASON_START = (9, 15)  # month, day
SEASON_END = (12, 15)

# Forecast setup
HORIZON_DAYS = 3  # predict: any fire in the next 1..3 days
TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALID_YEARS = [2024]  # early stopping + calibration
TEST_YEARS = [2025]  # touched once, for the final report

# Data sources (all free)
FIRMS_ARCHIVE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{year}/viirs-snpp_{year}_India.csv"
)
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{bbox}/{days}"
BOUNDARY_URLS = {
    "adm1": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM1/"
    "geoBoundaries-IND-ADM1_simplified.geojson",
    "adm2": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM2/"
    "geoBoundaries-IND-ADM2_simplified.geojson",
}
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# ~50 km. Day-level rain, humidity and wind are regional, and a coarse grid keeps us
# inside Open-Meteo's free limits.
WEATHER_GRID_DEG = 0.5
WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]


def season_bounds(year: int) -> tuple[date, date]:
    return date(year, *SEASON_START), date(year, *SEASON_END)


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, DEMO_DIR, MODELS_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)

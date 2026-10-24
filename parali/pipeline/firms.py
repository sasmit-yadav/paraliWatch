"""NASA FIRMS active-fire points (VIIRS S-NPP, 375 m).

Past seasons come from the yearly country archives (no key needed). Live mode uses the
FIRMS area API, which needs a free MAP_KEY.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date

import h3
import pandas as pd

from parali import config
from parali.pipeline.download import download, get

log = logging.getLogger(__name__)

KEEP_COLS = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight", "type"]


def clean(df: pd.DataFrame, cells: set[str], start: date, end: date) -> pd.DataFrame:
    """Filter raw FIRMS rows to the study area and season and attach H3 cells.

    - confidence: VIIRS uses l/n/h; low-confidence points are dropped.
    - type: archive files flag 0 = presumed vegetation fire, 2 = static land source (e.g. a
      factory), 3 = offshore. Only type 0 is kept. Live NRT files have no type column.
    """
    w, s, e, n = config.BBOX
    df = df[df["longitude"].between(w, e) & df["latitude"].between(s, n)]
    df = df[df["confidence"].astype(str).str.lower().str[0] != "l"]
    if "type" in df.columns:
        df = df[df["type"].fillna(0).astype(int) == 0]
    df = df.assign(acq_date=pd.to_datetime(df["acq_date"]).dt.date)
    df = df[(df["acq_date"] >= start) & (df["acq_date"] <= end)]
    df = df.assign(
        cell=[h3.latlng_to_cell(la, lo, config.H3_RES) for la, lo in zip(df["latitude"], df["longitude"])]
    )
    df = df[df["cell"].isin(cells)]
    cols = [c for c in KEEP_COLS if c in df.columns] + ["cell"]
    return df[cols].reset_index(drop=True)


def load_archive_year(year: int, cells: set[str]) -> pd.DataFrame:
    url = config.FIRMS_ARCHIVE_URL.format(year=year)
    path = download(url, config.RAW_DIR / "firms" / f"viirs-snpp_{year}_India.csv")
    start, end = config.season_bounds(year)
    parts = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in KEEP_COLS, chunksize=500_000):
        parts.append(clean(chunk, cells, start, end))
    out = pd.concat(parts, ignore_index=True)
    log.info("FIRMS %s: %d fire points in study area", year, len(out))
    return out


def load_live(cells: set[str], days: int = 5) -> pd.DataFrame:
    """Recent fires via the FIRMS area API (needs FIRMS_MAP_KEY)."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("Set FIRMS_MAP_KEY in .env to use live mode (free key from NASA FIRMS).")
    bbox = ",".join(str(v) for v in config.BBOX)
    resp = get(config.FIRMS_AREA_URL.format(key=key, bbox=bbox, days=min(days, 5)))
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty:
        return df
    today = date.today()
    return clean(df, cells, date(today.year, 1, 1), today)

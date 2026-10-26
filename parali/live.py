"""Live mode: forecast the season that is happening now, from today's satellite data.

The model is the same one evaluated on 2025; nothing is retrained. Each refresh:

1. Fire points for this season so far from NASA FIRMS (VIIRS S-NPP near-real-time, the same
   sensor the model was trained on), in 5-day requests (the API's maximum). Needs a free key in
   FIRMS_MAP_KEY (.env). Requests for windows older than a week are cached on disk.
2. Daily weather for the same days, plus the next three, from Open-Meteo's forecast API (free).
3. Features for every cell and day of the season so far, built with exactly the same code as
   training (last year's and earlier seasons come from the archive).
4. Probabilities, calibration and top-3 reasons with the saved model, written to data/live/.
5. **Live scoring:** for days whose 3-day outcome window has fully passed, the forecasts are
   scored against the fires that actually happened, so the season gets its own accuracy number
   as it goes, next to the best simple rule.

Usage: python -m parali live            (the dashboard also refreshes in the background)
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from parali import config
from parali.pipeline import build, firms
from parali.pipeline.download import get

log = logging.getLogger(__name__)
WINDOW = 5  # days per FIRMS request (API maximum)


def current_season(today: date | None = None) -> int:
    """The season in progress, or the most recent one outside the burning months."""
    today = today or date.today()
    start, _ = config.season_bounds(today.year)
    return today.year if today >= start else today.year - 1


def windows(start: date, end: date, size: int = WINDOW) -> list[tuple[date, int]]:
    """(first day, number of days) requests covering start..end inclusive."""
    out, d = [], start
    while d <= end:
        days = min(size, (end - d).days + 1)
        out.append((d, days))
        d += timedelta(days=days)
    return out


def available() -> bool:
    return bool(os.environ.get("FIRMS_MAP_KEY"))


def fetch_fires(year: int, cells: set[str], today: date) -> pd.DataFrame:
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("Live mode needs FIRMS_MAP_KEY in .env (free key: NASA FIRMS 'MAP_KEY').")
    start, end = config.season_bounds(year)
    end = min(end, today)
    bbox = ",".join(str(v) for v in config.BBOX)
    cache = config.RAW_DIR / "firms_live"
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for first, days in windows(start, end):
        path = cache / f"{first.isoformat()}_{days}.csv"
        settled = first + timedelta(days=days + 7) < today  # NRT files are final after a few days
        if settled and path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            url = config.FIRMS_AREA_DATED_URL.format(key=key, bbox=bbox, days=days, start=first.isoformat())
            text = get(url).text
            if not text.startswith("latitude"):
                raise RuntimeError(f"FIRMS answered: {text[:200]}")
            if settled:
                path.write_text(text, encoding="utf-8")
        df = pd.read_csv(io.StringIO(text))
        if not df.empty:
            parts.append(firms.clean(df, cells, start, end))
    fires = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=firms.KEEP_COLS + ["cell"])
    # overlapping requests can repeat a detection
    return fires.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"]).reset_index(drop=True)


def fetch_weather(year: int, points: pd.DataFrame, today: date) -> pd.DataFrame:
    """Observed/analysed weather for the season so far plus the forecast for the next 3 days."""
    start, end = config.season_bounds(year)
    end = min(end, today + timedelta(days=config.HORIZON_DAYS))
    uniq = points[["wlat", "wlon"]].drop_duplicates().reset_index(drop=True)
    frames = []
    for i in range(0, len(uniq), 50):
        batch = uniq.iloc[i : i + 50]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in batch["wlat"]),
            "longitude": ",".join(f"{v:.4f}" for v in batch["wlon"]),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(config.WEATHER_DAILY_VARS),
            "timezone": "Asia/Kolkata",
        }
        payload = get(config.OPEN_METEO_FORECAST_URL, params=params).json()
        payload = payload if isinstance(payload, list) else [payload]
        for (wlat, wlon), loc in zip(batch[["wlat", "wlon"]].itertuples(index=False), payload, strict=True):
            daily = pd.DataFrame(loc["daily"])
            daily["wlat"], daily["wlon"] = wlat, wlon
            frames.append(daily)
        time.sleep(1.0)
    out = pd.concat(frames, ignore_index=True).rename(columns={"time": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out


def score(frame: pd.DataFrame, last_complete: date) -> dict:
    """Accuracy on the days whose 3-day outcome is fully known, against the best simple rule."""
    from parali.forecast.metrics import evaluate
    from parali.forecast.train import BASELINES

    done = frame[frame["date"] <= last_complete]
    if done.empty or done["fire_next3"].sum() == 0:
        return {}
    done = done.assign(neighbourhood_3d=BASELINES["neighbourhood_3d"](done).fillna(0))
    return {
        "days_scored": int(done["date"].nunique()),
        "first_day": min(done["date"]).isoformat(),
        "last_day": last_complete.isoformat(),
        "model": evaluate(done, "raw"),
        "best_rule": evaluate(done, "neighbourhood_3d"),
    }


def refresh(today: date | None = None) -> dict:
    """Pull the latest data and rebuild the live season. Returns the live metadata."""
    import joblib
    import lightgbm as lgb

    from parali.forecast import export
    from parali.forecast.features import FEATURES, build_panel
    from parali.forecast.train import CALIBRATOR_PATH, MODEL_PATH
    from parali.pipeline.weather import weather_points

    today = today or date.today()
    year = current_season(today)
    grid = build.step_grid()
    cells = set(grid["cell"])
    t0 = time.time()
    history = [y for y in config.YEARS if y < year]
    missing = [y for y in history if not (build.fires_path(y).exists() and build.weather_path(y).exists())]
    if missing:  # features use earlier seasons (e.g. a cell's fires last year)
        raise RuntimeError(f"Live mode needs seasons {missing}: run `python -m parali data` once first.")

    fires_now = fetch_fires(year, cells, today)
    if fires_now.empty:
        raise RuntimeError(f"No fire data yet for the {year} season.")
    last_data = max(fires_now["acq_date"])  # NRT lags real time by a few hours
    weather_now = fetch_weather(year, weather_points(grid), last_data)
    log.info(
        "live %s: %d fire points up to %s, %d weather rows", year, len(fires_now), last_data, len(weather_now)
    )

    fires = pd.concat([build.load_fires(history), fires_now], ignore_index=True)
    weather = pd.concat([build.load_weather(history), weather_now], ignore_index=True)
    panel = build_panel(fires, weather, grid, years=history + [year], only=year)
    panel = panel[panel["date"] <= pd.Timestamp(last_data)].reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    calibrator = joblib.load(CALIBRATOR_PATH)
    panel["raw"] = booster.predict(panel[FEATURES])
    panel["prob"] = calibrator.predict(panel["raw"])

    # outcomes: complete only where all 3 following days are already in the data
    last_complete = last_data - timedelta(days=config.HORIZON_DAYS)
    metrics = score(panel, last_complete)
    panel.loc[panel["date"] > last_complete, "fires_next3_count"] = np.nan

    config.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    export.write_season(booster, panel, config.LIVE_DIR, year, fires_now)
    for name in ("grid.parquet", "districts.geojson"):
        (config.LIVE_DIR / name).write_bytes((config.DEMO_DIR / name).read_bytes())
    meta = {
        "season": year,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_data_date": last_data.isoformat(),
        "last_complete_outcome": last_complete.isoformat(),
        "fire_points": int(len(fires_now)),
        "source": "NASA FIRMS VIIRS S-NPP NRT + Open-Meteo forecast API",
        "metrics": metrics,
        "build_seconds": round(time.time() - t0, 1),
    }
    (config.LIVE_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("live %s rebuilt in %.0f s (%d days)", year, meta["build_seconds"], panel["date"].nunique())
    return meta


def meta() -> dict | None:
    path = config.LIVE_DIR / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def stale() -> bool:
    m = meta()
    if not m:
        return True
    built = datetime.fromisoformat(m["built_at"])
    return datetime.now(timezone.utc) - built > timedelta(hours=config.LIVE_REFRESH_HOURS)

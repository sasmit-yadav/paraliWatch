"""FastAPI app: JSON API for the forecast plus the map dashboard at /."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from parali import __version__, config
from parali.app.store import Store

STATIC = config.ROOT / "parali" / "app" / "static"

app = FastAPI(
    title="Parali Watch API",
    version=__version__,
    description="Next-3-day crop-residue fire risk for Punjab and Haryana, with reasons.",
)
store = Store()


def _parse_day(value: str | None) -> date:
    if value is None:
        return store.default_date()
    try:
        d = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(400, "date must be YYYY-MM-DD") from exc
    if d not in set(store.dates):
        raise HTTPException(404, f"no forecast for {d}; available {store.dates[0]} to {store.dates[-1]}")
    return d


@lru_cache(maxsize=1)
def _metrics() -> dict:
    path = config.EVAL_DIR / "metrics.json"
    if not path.exists():
        return {}
    m = json.loads(path.read_text(encoding="utf-8"))
    return {"test": m["model_uncalibrated"], "best_rule": m["baselines"]["neighbourhood_3d"]}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/meta")
def meta() -> dict:
    return {
        "mode": "demo",
        "season": store.year,
        "dates": [d.isoformat() for d in store.dates],
        "default_date": store.default_date().isoformat(),
        "horizon_days": config.HORIZON_DAYS,
        "bbox": config.BBOX,
        "metrics": _metrics(),
    }


@app.get("/api/risk")
def risk(date: str | None = None, min_prob: float = Query(0.0, ge=0, le=1)) -> dict:
    return store.risk_geojson(_parse_day(date), min_prob)


@app.get("/api/fires")
def fires(date: str | None = None, days: int = Query(1, ge=1, le=7)) -> dict:
    return store.fires_geojson(_parse_day(date), days)


@app.get("/api/top")
def top(date: str | None = None, n: int = Query(10, ge=1, le=100)) -> list[dict]:
    return store.top(_parse_day(date), n)


@app.get("/api/cell/{cell}")
def cell(cell: str, date: str | None = None) -> dict:
    out = store.cell(cell, _parse_day(date))
    if out is None:
        raise HTTPException(404, "cell not in study area")
    return out


@app.get("/api/districts")
def districts() -> dict:
    return store.districts


@app.get("/api/district-summary")
def district_summary(date: str | None = None) -> list[dict]:
    return store.district_summary(_parse_day(date))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

"""FastAPI app: JSON API for the forecast plus the map dashboard at /."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from parali import __version__, config
from parali.app.store import Store

STATIC = config.ROOT / "parali" / "app" / "static"


def _warm_up() -> None:
    """PARALI_WARMUP=1: load the document index and models in the background at start-up, so the
    first question doesn't wait for a model download (useful on a hosted demo)."""
    import os
    import threading

    if os.environ.get("PARALI_WARMUP", "0") != "1":
        return

    def load() -> None:
        try:
            from parali.advise.answer import answer

            answer("warm-up: crop residue incentive", use_llm=False)
        except Exception:  # the app still works; the first real question will load them
            pass

    threading.Thread(target=load, daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _warm_up()
    yield


app = FastAPI(
    title="Parali Watch API",
    version=__version__,
    description="Next-3-day crop-residue fire risk for Punjab and Haryana, with reasons.",
    lifespan=lifespan,
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


@app.get("/api/season")
def season() -> list[dict]:
    """Per-day totals for the whole season: fires detected, expected fire cells, high-risk cells."""
    return store.season


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


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    user_id: str | None = Field(default=None, max_length=64)


class ProfileIn(BaseModel):
    district: str | None = Field(default=None, max_length=60)
    crop: str | None = Field(default=None, max_length=40)
    land_acres: float | None = Field(default=None, ge=0, le=10_000)
    language: str | None = Field(default=None, pattern="^(en|hi)$")


@app.post("/api/ask")
def ask(q: Question) -> dict:
    """Answer a question about schemes and alternatives to burning, with page citations."""
    from parali.advise import memory
    from parali.advise.answer import answer
    from parali.ops import feedback, trace

    profile = memory.get(q.user_id) if q.user_id else None
    try:
        with trace.start("ask", question=q.question[:200]) as tr:
            out = answer(q.question, profile).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Document index not built yet. Run: python -m parali index") from exc
    out["trace_id"] = tr.trace_id
    try:
        feedback.record_answer(tr.trace_id, out)
    except Exception:  # feedback storage must never break an answer
        pass
    return out


class Feedback(BaseModel):
    trace_id: str = Field(min_length=8, max_length=64)
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=1000)


@app.post("/api/feedback")
def post_feedback(f: Feedback) -> dict:
    """Thumbs up (+1) or down (-1) on an answer, identified by the trace_id /api/ask returned."""
    from parali.ops import feedback

    if f.rating == 0:
        raise HTTPException(422, "rating must be +1 or -1")
    if not feedback.add(f.trace_id, f.rating, f.comment):
        raise HTTPException(404, "Unknown trace_id")
    return {"ok": True}


@app.get("/api/ops/summary")
def ops_summary(last: int = Query(500, ge=1, le=10_000)) -> dict:
    """Latency per answer step (p50/p95) over recent traces, answer modes, and user feedback."""
    from parali.ops import feedback, trace

    return {"tracing": trace.summary(trace.load(last)), "feedback": feedback.summary()}


@app.get("/api/ops/trace/{trace_id}")
def get_trace(trace_id: str) -> dict:
    from parali.ops import trace

    t = trace.find(trace_id)
    if t is None:
        raise HTTPException(404, "Unknown trace_id")
    return t


MAX_AUDIO_BYTES = 5_000_000  # about 5 minutes of compressed speech


@app.post("/api/transcribe")
async def transcribe(request: Request, language: str | None = Query(None, pattern="^(en|hi)$")) -> dict:
    """Speech to text (Hindi or English) for the ask box. Body: raw audio (webm/ogg/wav/mp3)."""
    audio = await request.body()
    if not audio:
        raise HTTPException(400, "Empty audio")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio too long")
    import importlib.util

    if importlib.util.find_spec("faster_whisper") is None:
        raise HTTPException(
            501, "Voice input needs the optional 'voice' dependencies (uv sync --group voice)"
        )
    from parali.voice.asr import transcribe as run

    try:
        return run(audio, language)
    except Exception as exc:  # undecodable audio
        raise HTTPException(400, f"Could not read the audio: {exc}") from exc


@app.get("/api/profile/{user_id}")
def get_profile(user_id: str) -> dict:
    from dataclasses import asdict

    from parali.advise import memory

    return asdict(memory.get(user_id))


@app.put("/api/profile/{user_id}")
def put_profile(user_id: str, p: ProfileIn) -> dict:
    from dataclasses import asdict

    from parali.advise import memory

    return asdict(memory.save(memory.Profile(user_id, **p.model_dump())))


@app.get("/api/bulletin")
def get_bulletin(date: str | None = None) -> dict:
    """The agent-written district bulletin for a date (generated on first request, saved as draft)."""
    from parali.agents import bulletin

    d = _parse_day(date)
    return (bulletin.load(d) or bulletin.generate(d)).model_dump(mode="json")


class Approval(BaseModel):
    official: str = Field(min_length=2, max_length=80)


@app.post("/api/bulletin/approve")
def approve_bulletin(a: Approval, date: str | None = None) -> dict:
    """Human in the loop: an official approves the draft before it is shared."""
    from parali.agents import bulletin

    return bulletin.approve(_parse_day(date), a.official).model_dump(mode="json")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

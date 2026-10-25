"""Tracing: where does the time go in one answer, and what did each step see?

A trace is one request; spans are its steps (guard, search, rerank, LLM, checks), nested like a
call tree. Field names follow OpenTelemetry (trace_id, span_id, parent_id, name, start, duration,
attributes), so the same records could be sent to Jaeger, Tempo or Langfuse later. Here they are
appended to data/ops/traces.jsonl, one JSON line per finished trace: no extra service to run.

    with trace.start("ask", question=q) as t:      # root span; t.trace_id is returned to the user
        with trace.span("retrieve"):
            ...
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field

from parali import config

TRACE_FILE = config.OPS_DIR / "traces.jsonl"
ENABLED = os.environ.get("PARALI_TRACE", "1") == "1"
_lock = threading.Lock()


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    start: float
    duration_ms: float = 0.0
    attributes: dict = field(default_factory=dict)
    status: str = "ok"


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)

    def to_dict(self) -> dict:
        root = self.spans[0] if self.spans else None
        return {
            "trace_id": self.trace_id,
            "name": root.name if root else "",
            "start": root.start if root else 0,
            "duration_ms": root.duration_ms if root else 0,
            "spans": [asdict(s) for s in self.spans],
        }


_current_trace: ContextVar[Trace | None] = ContextVar("trace", default=None)
_current_span: ContextVar[Span | None] = ContextVar("span", default=None)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def span(name: str, **attributes):
    """A step inside the current trace. Outside a trace it does nothing (no cost, no file)."""
    tr = _current_trace.get()
    if tr is None:
        yield None
        return
    parent = _current_span.get()
    s = Span(name, _new_id(), parent.span_id if parent else None, time.time(), attributes=dict(attributes))
    tr.spans.append(s)
    token = _current_span.set(s)
    t0 = time.perf_counter()
    try:
        yield s
    except Exception as exc:
        s.status = f"error: {type(exc).__name__}"
        raise
    finally:
        s.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        _current_span.reset(token)


def annotate(**attributes) -> None:
    """Add attributes to the span that is currently open."""
    s = _current_span.get()
    if s is not None:
        s.attributes.update(attributes)


@contextmanager
def start(name: str, **attributes):
    """Root span of a new trace; the finished trace is written to the trace file."""
    tr = Trace(uuid.uuid4().hex)
    token = _current_trace.set(tr)
    try:
        with span(name, **attributes):
            yield tr
    finally:
        _current_trace.reset(token)
        if ENABLED:
            write(tr)


def write(tr: Trace) -> None:
    try:
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(tr.to_dict(), ensure_ascii=False, default=str)
        with _lock, open(TRACE_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:  # tracing must never break an answer
        pass


def load(limit: int | None = None) -> list[dict]:
    if not TRACE_FILE.exists():
        return []
    lines = TRACE_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in (lines[-limit:] if limit else lines) if x.strip()]


def find(trace_id: str) -> dict | None:
    for t in reversed(load()):
        if t["trace_id"] == trace_id:
            return t
    return None


def summary(traces: list[dict]) -> dict:
    """Latency per step (p50 / p95 in ms) and how often each answer mode was used."""
    import numpy as np

    by_step: dict[str, list[float]] = {}
    modes: dict[str, int] = {}
    for t in traces:
        for s in t["spans"]:
            by_step.setdefault(s["name"], []).append(s["duration_ms"])
            if s["name"] == "ask" and "mode" in s["attributes"]:
                modes[s["attributes"]["mode"]] = modes.get(s["attributes"]["mode"], 0) + 1
    steps = {
        name: {
            "count": len(v),
            "p50_ms": round(float(np.percentile(v, 50)), 1),
            "p95_ms": round(float(np.percentile(v, 95)), 1),
        }
        for name, v in sorted(by_step.items())
    }
    return {"traces": len(traces), "steps": steps, "answer_modes": modes}

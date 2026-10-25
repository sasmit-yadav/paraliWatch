"""User feedback loop: thumbs up / down on answers, turned into new evaluation questions.

1. Every answer served by /api/ask is stored with its trace id (question, answer, mode, sources).
2. The user can rate it (/api/feedback): +1 or -1, with an optional comment.
3. `python -m parali.ops.feedback export` writes the down-voted answers to
   eval/rag/feedback_candidates.jsonl in the same format as eval/rag/questions.jsonl, with the
   gold page and facts left empty. A person fills those in (what *should* the answer cite?) and
   moves the reviewed lines into questions.jsonl, so every real failure becomes a test case
   that guards against regressions.

Storage: SQLite at data/ops/feedback.db (local, not committed: it can hold user text).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone

from parali import config

DB = config.OPS_DIR / "feedback.db"
CANDIDATES = config.EVAL_DIR / "rag" / "feedback_candidates.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    trace_id TEXT PRIMARY KEY, at TEXT, question TEXT, answer TEXT, mode TEXT, language TEXT,
    citations TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, trace_id TEXT, rating INTEGER, comment TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_answer(trace_id: str, answer: dict) -> None:
    cites = [{"n": c["n"], "doc_id": c["doc_id"], "page": c["page"]} for c in answer.get("citations", [])]
    with closing(_connect()) as con, con:
        con.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace_id,
                _now(),
                answer["question"],
                answer["text"],
                answer["mode"],
                answer["language"],
                json.dumps(cites),
            ),
        )


def add(trace_id: str, rating: int, comment: str | None = None) -> bool:
    """Store a rating; False if the trace id is unknown (nothing to attach it to)."""
    with closing(_connect()) as con, con:
        if con.execute("SELECT 1 FROM answers WHERE trace_id = ?", (trace_id,)).fetchone() is None:
            return False
        con.execute(
            "INSERT INTO feedback (at, trace_id, rating, comment) VALUES (?, ?, ?, ?)",
            (_now(), trace_id, rating, (comment or "")[:1000]),
        )
    return True


def summary() -> dict:
    with closing(_connect()) as con:
        answers = con.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        rows = con.execute(
            "SELECT a.mode, f.rating, COUNT(*) FROM feedback f JOIN answers a USING (trace_id) GROUP BY 1, 2"
        ).fetchall()
    by_mode: dict[str, dict] = {}
    for mode, rating, n in rows:
        by_mode.setdefault(mode, {"up": 0, "down": 0})["up" if rating > 0 else "down"] += n
    up = sum(m["up"] for m in by_mode.values())
    down = sum(m["down"] for m in by_mode.values())
    return {
        "answers": answers,
        "rated": up + down,
        "up": up,
        "down": down,
        "satisfaction": round(up / (up + down), 3) if up + down else None,
        "by_mode": by_mode,
    }


def export_candidates() -> int:
    """Down-voted answers -> evaluation question candidates for human review."""
    with closing(_connect()) as con:
        rows = con.execute(
            "SELECT a.trace_id, a.question, a.language, a.answer, a.citations, f.comment FROM feedback f "
            "JOIN answers a USING (trace_id) WHERE f.rating < 0 ORDER BY f.id"
        ).fetchall()
    existing = set()
    if CANDIDATES.exists():
        existing = {
            json.loads(x)["from_trace"] for x in CANDIDATES.read_text(encoding="utf-8").splitlines() if x
        }
    new = 0
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES, "a", encoding="utf-8") as fh:
        for trace_id, question, lang, answer, cites, comment in rows:
            if trace_id in existing:
                continue
            fh.write(
                json.dumps(
                    {
                        "id": f"fb-{trace_id[:8]}",
                        "question": question,
                        "lang": lang,
                        "answerable": None,
                        "gold": [],
                        "facts": [],
                        "from_trace": trace_id,
                        "user_comment": comment,
                        "bad_answer": answer[:500],
                        "bad_citations": json.loads(cites),
                        "review": "TODO: set answerable, gold pages and facts, then move to questions.jsonl",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            existing.add(trace_id)
            new += 1
    return new


if __name__ == "__main__":
    if sys.argv[1:] == ["export"]:
        print(f"{export_candidates()} new candidates written to {CANDIDATES}")
    else:
        print(json.dumps(summary(), indent=2))

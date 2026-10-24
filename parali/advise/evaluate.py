"""Evaluate the assistant on a hand-written question set.

Question file (eval/rag/questions.jsonl), one JSON object per line:
    {"id": "q01", "question": "...", "lang": "en" | "hi", "answerable": true,
     "gold": [{"doc_id": "file.pdf", "page": 3}], "facts": ["1000", "per acre"]}

Retrieval: for each chunking strategy x embedding model x search mode, the share of questions whose
gold page is in the top k (recall@k) and the mean reciprocal rank (MRR).
Answers (needs an LLM): citation on a gold page, key facts present, numbers not in the sources,
and correct "not found" on unanswerable questions.

    python -m parali.advise.evaluate retrieval
    python -m parali.advise.evaluate answers
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

from parali import config
from parali.advise import index
from parali.advise.ingest import load_corpus
from parali.advise.retrieve import MODES, Retriever
from parali.advise.text import normalize, numbers

EVAL_DIR = config.EVAL_DIR / "rag"
QUESTIONS = EVAL_DIR / "questions.jsonl"
EMBED_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-large",
]
RERANK_MODELS = ["BAAI/bge-reranker-v2-m3-int8", "Xenova/ms-marco-MiniLM-L-6-v2"]


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _is_gold(chunk, gold: list[dict]) -> bool:
    return any(
        chunk.doc_id == g["doc_id"] and (g.get("page") is None or chunk.page == g["page"]) for g in gold
    )


def per_question(retriever: Retriever, questions: list[dict], mode: str, k: int = 5) -> list[dict]:
    """Search each question once; record the gold rank, top score and latency."""
    out = []
    for q in questions:
        start = time.perf_counter()
        hits = retriever.search(q["question"], k=k, mode=mode)
        rank = next((i for i, h in enumerate(hits, start=1) if _is_gold(h.chunk, q["gold"])), None)
        out.append(
            {
                "id": q["id"],
                "lang": q["lang"],
                "answerable": q["answerable"],
                "rank": rank,
                "top_score": hits[0].score if hits else None,
                "seconds": time.perf_counter() - start,
            }
        )
    return out


def summarise(rows: list[dict], ks=(1, 3, 5)) -> dict:
    ans = [r for r in rows if r["answerable"]]
    n = max(1, len(ans))
    out = {f"recall@{k}": sum(r["rank"] is not None and r["rank"] <= k for r in ans) / n for k in ks}
    out["mrr"] = sum(1.0 / r["rank"] for r in ans if r["rank"]) / n
    out["questions"] = len(ans)
    out["median_latency_ms"] = (
        round(1000 * float(np.median([r["seconds"] for r in rows])), 1) if rows else None
    )
    return out


def run_retrieval(
    strategies=("fixed", "structure", "semantic"), models=EMBED_MODELS[:2], rerankers=RERANK_MODELS
) -> list[dict]:
    questions = load_questions()
    pages = load_corpus()
    rows = []
    for strategy, model in product(strategies, models):
        index.build(pages, strategy=strategy, model=model)
        for reranker in rerankers:
            r = Retriever(strategy=strategy, model=model, rerank_model=reranker)
            for mode in MODES:
                if mode != "hybrid_rerank" and reranker != rerankers[0]:
                    continue  # the reranker only changes the last mode
                per_q = per_question(r, questions, mode)
                for lang in ("all", "en", "hi"):
                    sub = [x for x in per_q if lang == "all" or x["lang"] == lang]
                    row = {
                        "strategy": strategy,
                        "embed_model": model,
                        "mode": mode,
                        "reranker": reranker if mode == "hybrid_rerank" else None,
                        "lang": lang,
                        **summarise(sub),
                    }
                    if lang == "all" and mode == "hybrid_rerank":
                        row["top_scores"] = {
                            "answerable": [x["top_score"] for x in per_q if x["answerable"]],
                            "unanswerable": [x["top_score"] for x in per_q if not x["answerable"]],
                        }
                    rows.append(row)
                    print({k: v for k, v in row.items() if k != "top_scores"}, flush=True)
                EVAL_DIR.mkdir(parents=True, exist_ok=True)
                (EVAL_DIR / "retrieval_results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def best_threshold(rows: list[dict], strategy=None, model=None, reranker=None) -> dict:
    """Pick the reranker score that best separates answerable from unanswerable questions."""
    row = next(
        r
        for r in rows
        if r["mode"] == "hybrid_rerank"
        and r["lang"] == "all"
        and r.get("top_scores")
        and strategy in (None, r["strategy"])
        and model in (None, r["embed_model"])
        and reranker in (None, r["reranker"])
    )
    a, u = np.array(row["top_scores"]["answerable"]), np.array(row["top_scores"]["unanswerable"])
    best = (0.0, None)
    for t in np.unique(np.concatenate([a, u])):
        acc = ((a >= t).sum() + (u < t).sum()) / (len(a) + len(u))
        if acc > best[0]:
            best = (float(acc), float(t))
    return {"threshold": best[1], "accuracy": best[0], "answerable": len(a), "unanswerable": len(u)}


def _has_fact(answer: str, fact: str) -> bool:
    a, f = normalize(answer).lower(), normalize(fact).lower()
    if f in a:
        return True
    nums = numbers(f)
    return bool(nums) and nums <= numbers(a)


HITS_CACHE = EVAL_DIR / "retrieved_contexts.json"


def retrieve_all(questions: list[dict]) -> None:
    """Stage 1: retrieval for every question (loads the embedding and reranker models)."""
    from parali.advise.answer import retrieve

    out = {}
    for q in questions:
        hits, trace = retrieve(q["question"], use_llm=False)
        out[q["id"]] = {
            "trace": trace,
            "hits": [{"chunk": h.chunk.to_dict(), "score": h.score, "detail": h.detail} for h in hits],
        }
    HITS_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def run_answers(limit: int | None = None, stage: str = "both") -> dict | None:
    """Answer quality. Stages can run in separate processes, so a local LLM server and the reranker
    don't have to fit in memory at the same time: `answers retrieve`, then `answers generate`."""
    from parali.advise.answer import compose
    from parali.advise.chunking import Chunk
    from parali.advise.retrieve import Hit

    questions = load_questions()[:limit]
    if stage in ("both", "retrieve"):
        retrieve_all(questions)
        if stage == "retrieve":
            return None
    cached = json.loads(HITS_CACHE.read_text(encoding="utf-8"))
    records = []
    for q in questions:
        start = time.perf_counter()
        c = cached[q["id"]]
        hits = [Hit(Chunk(**h["chunk"]), h["score"], h["detail"]) for h in c["hits"]]
        a = compose(q["question"], hits, {"injection_suspected": False, **c["trace"]})
        cited_gold = any(_is_gold(cit, q.get("gold", [])) for cit in a.citations)
        records.append(
            {
                "id": q["id"],
                "lang": q["lang"],
                "answerable": q["answerable"],
                "mode": a.mode,
                "seconds": round(time.perf_counter() - start, 1),
                "cited_gold_page": cited_gold,
                "facts_found": [f for f in q.get("facts", []) if _has_fact(a.text, f)],
                "facts_total": len(q.get("facts", [])),
                "unsupported_numbers": a.checks.get("unsupported_numbers", []),
                "has_citation": a.checks.get("has_citation"),
                "rejected_llm_text": a.checks.get("rejected_llm_text"),
                "llm_error": a.checks.get("llm_error"),
                "answer": a.text,
            }
        )
        print(
            records[-1]["id"],
            records[-1]["mode"],
            records[-1]["cited_gold_page"],
            records[-1]["seconds"],
            flush=True,
        )
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    llm_tried = [r for r in ans if r["mode"] == "llm" or r["rejected_llm_text"]]
    summary = {
        "questions": len(records),
        "llm_model": config.LLM_MODEL or None,
        "llm_attempts": len(llm_tried),
        "llm_answers_accepted": sum(r["mode"] == "llm" for r in ans),
        "llm_rejected_no_citation": sum(r["has_citation"] is False for r in ans),
        "llm_rejected_unsupported_numbers": sum(bool(r["unsupported_numbers"]) for r in ans),
        "llm_errors": sum(bool(r["llm_error"]) for r in records),
        "fallback_to_quotes": sum(r["mode"] == "extractive" for r in ans),
        "cited_gold_page": sum(r["cited_gold_page"] for r in ans) / max(1, len(ans)),
        "fact_recall": sum(len(r["facts_found"]) for r in ans) / max(1, sum(r["facts_total"] for r in ans)),
        "false_answer_on_unanswerable": sum(r["mode"] != "not_found" for r in una),
        "unanswerable": len(una),
        "wrongly_refused_answerable": sum(r["mode"] == "not_found" for r in ans),
        "median_seconds_generation": float(np.median([r["seconds"] for r in records])) if records else None,
    }
    name = "answer_results_" + (
        config.LLM_MODEL.replace("/", "_").replace(":", "_") if config.LLM_MODEL else "no_llm"
    )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"{name}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    if what == "retrieval":
        res = run_retrieval()
        print(best_threshold(res))
    elif what == "answers":
        run_answers(stage=sys.argv[2] if len(sys.argv) > 2 else "both")

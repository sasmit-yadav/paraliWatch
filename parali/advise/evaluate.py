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


def retrieval_metrics(retriever: Retriever, questions: list[dict], mode: str, ks=(1, 3, 5)) -> dict:
    hits_at = {k: 0 for k in ks}
    rr, n, top_scores, latency = 0.0, 0, {"answerable": [], "unanswerable": []}, []
    for q in questions:
        start = time.perf_counter()
        hits = retriever.search(q["question"], k=max(ks), mode=mode)
        latency.append(time.perf_counter() - start)
        if hits and mode == "hybrid_rerank":
            top_scores["answerable" if q["answerable"] else "unanswerable"].append(hits[0].score)
        if not q["answerable"]:
            continue
        n += 1
        rank = next((i for i, h in enumerate(hits, start=1) if _is_gold(h.chunk, q["gold"])), None)
        for k in ks:
            hits_at[k] += int(rank is not None and rank <= k)
        rr += 1.0 / rank if rank else 0.0
    out = {f"recall@{k}": hits_at[k] / max(1, n) for k in ks}
    out.update(mrr=rr / max(1, n), questions=n, median_latency_ms=round(1000 * float(np.median(latency)), 1))
    if mode == "hybrid_rerank":
        out["top_scores"] = top_scores
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
                for lang in ("all", "en", "hi"):
                    qs = [q for q in questions if lang == "all" or q["lang"] == lang]
                    if not qs:
                        continue
                    m = retrieval_metrics(r, qs, mode)
                    rows.append(
                        {
                            "strategy": strategy,
                            "embed_model": model,
                            "mode": mode,
                            "reranker": reranker if mode == "hybrid_rerank" else None,
                            "lang": lang,
                            **m,
                        }
                    )
                    print({k: v for k, v in rows[-1].items() if k != "top_scores"})
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "retrieval_results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def best_threshold(rows: list[dict]) -> dict:
    """Pick the reranker score that best separates answerable from unanswerable questions."""
    row = next(r for r in rows if r["mode"] == "hybrid_rerank" and r["lang"] == "all" and r.get("top_scores"))
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


def run_answers(limit: int | None = None) -> dict:
    from parali.advise.answer import answer

    questions = load_questions()[:limit]
    records = []
    for q in questions:
        start = time.perf_counter()
        a = answer(q["question"])
        cited_gold = any(_is_gold(c, q.get("gold", [])) for c in a.citations)
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
                "answer": a.text,
            }
        )
        print(records[-1]["id"], records[-1]["mode"], records[-1]["cited_gold_page"], records[-1]["seconds"])
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    summary = {
        "questions": len(records),
        "llm_answers": sum(r["mode"] == "llm" for r in ans),
        "fallback_to_quotes": sum(r["mode"] == "extractive" for r in ans),
        "cited_gold_page": sum(r["cited_gold_page"] for r in ans) / max(1, len(ans)),
        "fact_recall": sum(len(r["facts_found"]) for r in ans) / max(1, sum(r["facts_total"] for r in ans)),
        "answers_with_unsupported_numbers": sum(bool(r["unsupported_numbers"]) for r in ans),
        "false_answer_on_unanswerable": sum(r["mode"] != "not_found" for r in una),
        "unanswerable": len(una),
        "wrongly_refused_answerable": sum(r["mode"] == "not_found" for r in ans),
        "median_seconds": float(np.median([r["seconds"] for r in records])) if records else None,
        "llm_model": config.LLM_MODEL or None,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "answer_results.json").write_text(
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
        run_answers()

"""Build fine-tuning data that teaches a small LLM to answer *only* from the given sources, with
[n] citations, and to say "not found" when the sources do not answer the question.

Steps (no human labelling, no paid API):
 1. Pick document chunks that are NOT on any evaluation gold page (and not near-duplicates of
    one), so the evaluation questions stay unseen.
 2. doc2query: a local LLM writes questions that the chunk answers (English, some Hindi).
 3. Round-trip filter: keep a question only if the real retriever puts its chunk in the top 3.
 4. Target answer = the grounded extractive answer (the sentences closest to the question,
    each with its source number), so every target passes the guardrails by construction.
 5. Negatives: the same question paired with passages the reranker scores as irrelevant -> "not found".

Prompts are built with answer.build_messages, the exact prompt used at inference time.

Usage (three stages, so the LLM server and the retrieval models never share memory):
  python -m parali.tune.data select
  PARALI_LLM_BASE_URL=... PARALI_LLM_MODEL=... python -m parali.tune.data questions [per_chunk]
  python -m parali.tune.data build
"""

from __future__ import annotations

import json
import random
import re
import sys

import numpy as np

from parali import config
from parali.advise import llm
from parali.advise.answer import NOT_FOUND, _extractive, build_messages, retriever
from parali.advise.embed import embed
from parali.advise.evaluate import load_questions
from parali.advise.retrieve import rerank

OUT = config.DATA_DIR / "tune"
SEED = 0


def candidate_chunks() -> list:
    chunks = retriever().chunks
    gold = {(g["doc_id"], g["page"]) for q in load_questions() for g in q.get("gold", [])}
    gold_chunks = [c for c in chunks if (c.doc_id, c.page) in gold]
    free = [c for c in chunks if (c.doc_id, c.page) not in gold and len(c.text.split()) > 60]
    if gold_chunks and free:
        g = embed([c.text for c in gold_chunks], kind="passage")
        f = embed([c.text for c in free], kind="passage")
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        free = [c for c, s in zip(free, (f @ g.T).max(1), strict=True) if s < 0.9]
    return free


def generate_questions(text: str, n: int) -> list[str]:
    prompt = (
        f"Read this passage from an Indian government document on crop residue (paddy straw) management.\n"
        f"Write {n} different questions a farmer or official could ask that this passage answers. "
        "Each question must be answerable from the passage alone. One question per line, no numbering.\n\n"
        f"Passage:\n{text}"
    )
    out = llm.chat(
        [{"role": "user", "content": prompt}], temperature=0.7, max_tokens=200, purpose="doc2query"
    )
    qs = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip() for line in out.splitlines()]
    return [q for q in qs if q.endswith("?") and 6 <= len(q.split()) <= 30][:n]


def to_hindi(question: str) -> str | None:
    out = llm.chat(
        [
            {
                "role": "user",
                "content": f"Translate this question into simple Hindi (Devanagari). Output only the "
                f"translation.\n{question}",
            }
        ],
        temperature=0.2,
        max_tokens=120,
        purpose="translate",
    )
    return out.strip() if re.search(r"[ऀ-ॿ]", out) else None


def blocks_for(hits) -> list[str]:
    return [
        f"[{i}] {h.chunk.title}, page {h.chunk.page}:\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    ]


def select() -> None:
    """Stage 1 (no LLM): pick the chunks questions will be written from."""
    chunks = candidate_chunks()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunks.json").write_text(json.dumps([c.chunk_id for c in chunks], indent=1))
    print(f"{len(chunks)} candidate chunks", flush=True)


def questions(per_chunk: int = 5) -> None:
    """Stage 2 (LLM only): doc2query, and translate about 30% of questions into Hindi."""
    from parali.advise.chunking import Chunk

    rng = random.Random(SEED)
    by_id = {}
    from parali.advise.index import table_name

    path = config.INDEX_DIR / f"{table_name(config.CHUNK_STRATEGY, config.EMBED_MODEL)}.chunks.jsonl"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            c = Chunk(**json.loads(line))
            by_id[c.chunk_id] = c
    out = []
    for cid in json.loads((OUT / "chunks.json").read_text()):
        try:
            qs = generate_questions(by_id[cid].text, per_chunk)
        except llm.LLMUnavailable as exc:
            raise SystemExit(f"doc2query needs an LLM endpoint: {exc}") from exc
        for q in qs:
            item = {"chunk_id": cid, "question": q, "english": q}
            if rng.random() < 0.3:  # a share of Hindi questions, like real users
                hq = to_hindi(q)
                if hq:
                    item["question"] = hq
            out.append(item)
        print(cid, len(out), flush=True)
    (OUT / "questions.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n")


def build() -> None:
    """Stage 3 (no LLM): round-trip filter, grounded targets, verified negatives."""
    rng = random.Random(SEED)
    r = retriever()
    items = [json.loads(line) for line in open(OUT / "questions.jsonl", encoding="utf-8") if line.strip()]
    chunk_ids = json.loads((OUT / "chunks.json").read_text())
    rows = []
    stats = {
        "generated": len(items),
        "hindi": sum(i["question"] != i["english"] for i in items),
        "kept_round_trip": 0,
        "negatives": 0,
    }
    for item in items:
        q, cid = item["question"], item["chunk_id"]
        doc_id = r.by_id[cid].doc_id
        hits = [h for h in r.search(q) if h.score >= config.MIN_RERANK_SCORE]
        if cid not in [h.chunk.chunk_id for h in hits[:3]]:
            continue
        stats["kept_round_trip"] += 1
        lang = "hi" if re.search(r"[ऀ-ॿ]", q) else "en"
        text, _ = _extractive(q, hits, max_sentences=3)
        if lang == "hi":
            text = "दस्तावेज़ों से संबंधित अंश (अंग्रेज़ी में):\n" + text
        rows.append(
            {
                "kind": "answer",
                "lang": lang,
                "source_chunk": cid,
                "messages": build_messages(q, blocks_for(hits), lang)
                + [{"role": "assistant", "content": text}],
            }
        )
        # negative: same question, passages from another topic that the reranker judges irrelevant
        if rng.random() < 0.35:
            other = r.by_id[rng.choice(chunk_ids)]
            others = [h for h in r.search(other.text[:300]) if h.chunk.doc_id != doc_id][:3]
            if (
                others
                and max(rerank(q, [h.chunk.text for h in others], r.rerank_model)) < config.MIN_RERANK_SCORE
            ):
                rows.append(
                    {
                        "kind": "not_found",
                        "lang": lang,
                        "source_chunk": cid,
                        "messages": build_messages(q, blocks_for(others), lang)
                        + [
                            {"role": "assistant", "content": NOT_FOUND[lang].format(helpline=config.HELPLINE)}
                        ],
                    }
                )
                stats["negatives"] += 1
    rng.shuffle(rows)
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["rows"] = len(rows)
    stats["candidate_chunks"] = len(chunk_ids)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(stats)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("select", "all"):
        select()
    if stage in ("questions", "all"):
        questions(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    if stage in ("build", "all"):
        build()

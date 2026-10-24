"""Search: keyword (BM25), meaning (vectors), both fused (reciprocal rank fusion), then reranked.

Each mode is evaluated separately in eval/rag so we can show what each step adds.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parali import config
from parali.advise.chunking import Chunk
from parali.advise.embed import embed, rerank
from parali.advise.index import table_name
from parali.advise.text import expand_bilingual, tokens

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    detail: dict = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        strategy: str = config.CHUNK_STRATEGY,
        model: str = config.EMBED_MODEL,
        rerank_model: str = config.RERANK_MODEL,
        index_dir: Path = config.INDEX_DIR,
    ):
        import lancedb

        self.model, self.rerank_model = model, rerank_model
        name = table_name(strategy, model)
        path = index_dir / f"{name}.chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No index for {name}. Run: python -m parali index")
        with open(path, encoding="utf-8") as fh:
            self.chunks = [Chunk(**json.loads(line)) for line in fh]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        with open(index_dir / f"{name}.bm25.pkl", "rb") as fh:
            data = pickle.load(fh)
        self.bm25, self.bm25_ids = data["bm25"], data["ids"]
        self.table = lancedb.connect(str(index_dir / "lancedb")).open_table(name)

    def keyword(self, query: str, k: int = 20) -> list[Hit]:
        """BM25+ (not plain BM25): its term weights never drop to zero, which matters in small
        corpora where a word appears in half the chunks."""
        q = tokens(expand_bilingual(query))
        scores = self.bm25.get_scores(q)
        has_term = np.array([any(t in freqs for t in q) for freqs in self.bm25.doc_freqs])
        scores = np.where(has_term, scores, -np.inf)  # a chunk sharing no word is not a keyword hit
        order = np.argsort(-scores)[:k]
        return [Hit(self.by_id[self.bm25_ids[i]], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def dense(self, query: str, k: int = 20) -> list[Hit]:
        vec = embed([query], self.model, kind="query")[0]
        rows = self.table.search(vec).metric("cosine").limit(k).to_list()
        return [Hit(self.by_id[r["chunk_id"]], 1.0 - float(r["_distance"])) for r in rows]

    def hybrid(self, query: str, k: int = 20, rrf_k: int = 60) -> list[Hit]:
        """Reciprocal rank fusion: score = sum of 1 / (60 + rank) over both lists. Needs no score
        calibration between BM25 and cosine, which live on different scales."""
        fused: dict[str, dict] = {}
        for source, hits in (("bm25", self.keyword(query, k)), ("dense", self.dense(query, k))):
            for rank, hit in enumerate(hits, start=1):
                entry = fused.setdefault(hit.chunk.chunk_id, {"score": 0.0})
                entry["score"] += 1.0 / (rrf_k + rank)
                entry[f"{source}_rank"] = rank
        order = sorted(fused.items(), key=lambda kv: -kv[1]["score"])[:k]
        return [Hit(self.by_id[cid], d["score"], d) for cid, d in order]

    def search(
        self, query: str, k: int = config.TOP_K, mode: str = "hybrid_rerank", pool: int = 20
    ) -> list[Hit]:
        if mode == "bm25":
            return self.keyword(query, k)
        if mode == "dense":
            return self.dense(query, k)
        candidates = self.hybrid(query, pool)
        if mode == "hybrid":
            return candidates[:k]
        scores = rerank(query, [h.chunk.text for h in candidates], self.rerank_model)
        for hit, s in zip(candidates, scores, strict=True):
            hit.detail["rerank"] = float(s)
            hit.score = float(s)
        return sorted(candidates, key=lambda h: -h.score)[:k]

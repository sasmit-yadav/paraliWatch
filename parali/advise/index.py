"""Build the search indexes: a LanceDB vector table and a BM25 keyword index per chunking strategy.

python -m parali index            # reads data/corpus, writes data/index
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Plus

from parali import config
from parali.advise import chunking
from parali.advise.chunking import Chunk
from parali.advise.embed import embed
from parali.advise.ingest import Page, load_corpus
from parali.advise.text import expand_bilingual, tokens

log = logging.getLogger(__name__)


def make_chunks(pages: list[Page], strategy: str, model: str = config.EMBED_MODEL) -> list[Chunk]:
    usable = [p for p in pages if p.quality == "ok"]
    if strategy == "fixed":
        return chunking.fixed(usable)
    if strategy == "structure":
        return chunking.structure(usable)
    if strategy == "semantic":
        return chunking.semantic(usable, lambda xs: embed(xs, model))
    raise ValueError(f"unknown chunking strategy {strategy}")


def table_name(strategy: str, model: str) -> str:
    return f"{strategy}__{model.split('/')[-1].lower().replace('-', '_').replace('.', '_')}"


def build(
    pages: list[Page] | None = None,
    strategy: str = config.CHUNK_STRATEGY,
    model: str = config.EMBED_MODEL,
    index_dir: Path = config.INDEX_DIR,
) -> list[Chunk]:
    import lancedb

    if pages is None:
        pages = load_corpus()
        if index_dir == config.INDEX_DIR:
            from parali.advise.ingest import save_snapshot

            save_snapshot(pages)
    chunks = make_chunks(pages, strategy, model)
    if not chunks:
        raise RuntimeError("No usable text found in the corpus. Add documents to data/corpus/.")
    index_dir.mkdir(parents=True, exist_ok=True)
    name = table_name(strategy, model)

    vectors = embed([c.text for c in chunks], model, kind="passage")
    db = lancedb.connect(str(index_dir / "lancedb"))
    rows = [{**c.to_dict(), "vector": v.tolist()} for c, v in zip(chunks, vectors, strict=True)]
    db.create_table(name, data=rows, mode="overwrite")

    corpus_tokens = [tokens(expand_bilingual(c.text)) for c in chunks]
    with open(index_dir / f"{name}.bm25.pkl", "wb") as fh:
        pickle.dump({"bm25": BM25Plus(corpus_tokens), "ids": [c.chunk_id for c in chunks]}, fh)
    with open(index_dir / f"{name}.chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    log.info("Indexed %d chunks (%s, %s)", len(chunks), strategy, model)
    return chunks

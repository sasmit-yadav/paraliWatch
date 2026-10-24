"""Embedding and reranking models (ONNX via fastembed: no PyTorch needed, runs on a laptop CPU).

Models download once (to PARALI_MODEL_CACHE, default ~/.cache/parali/models) and are reused."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from parali import config

CACHE = os.environ.get("PARALI_MODEL_CACHE") or str(Path.home() / ".cache" / "parali" / "models")


@lru_cache(maxsize=4)
def _embedder(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name, cache_dir=CACHE)


def embed(texts: list[str], model: str = config.EMBED_MODEL, kind: str = "passage") -> np.ndarray:
    """E5 models expect 'query: ' / 'passage: ' prefixes; other models take raw text."""
    if "e5" in model.lower():
        texts = [f"{kind}: {t}" for t in texts]
    return np.asarray(list(_embedder(model).embed(texts, batch_size=32)), dtype=np.float32)


@lru_cache(maxsize=2)
def _reranker(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=name, cache_dir=CACHE)


def rerank(query: str, texts: list[str], model: str = config.RERANK_MODEL) -> np.ndarray:
    """Cross-encoder relevance score for each (query, text) pair; higher is more relevant."""
    if not texts:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(list(_reranker(model).rerank(query, texts)), dtype=np.float32)

"""Embedders and the SQLite-backed embedding cache.

``HashEmbedder`` is a deterministic feature-hashing fake for unit tests.
``LocalEmbedder`` wraps fastembed with the model cached in ``data/models/`` and
never downloads at runtime (``local_files_only=True``).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from coderag.lexical import tokenize
from coderag.store import Store

DEFAULT_MODEL = os.environ.get("CODERAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_MODEL_DIR = Path(
    os.environ.get("CODERAG_MODEL_DIR", Path(__file__).resolve().parents[2] / "data" / "models")
)


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to L2-normalized float32 vectors."""

    @property
    def name(self) -> str:
        """Stable identifier used as the embedding cache key."""
        ...

    @property
    def dim(self) -> int:
        """Vector dimension."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Normalized vectors for documents, shape (n, dim)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Normalized vector for a query, shape (dim,)."""
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return np.where(norms > 0, m / np.maximum(norms, 1e-12), m).astype(np.float32)


class HashEmbedder:
    """Deterministic signed feature hashing over identifier-aware tokens."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        """Cache key for this embedder's vectors."""
        return f"hash-{self._dim}"

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in tokenize(text):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            v[idx] += 1.0 if h[4] & 1 else -1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Vectors for documents, shape (n, dim)."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        """Vector for a query, shape (dim,)."""
        return _normalize(self._vec(text))


class LocalEmbedder:
    """fastembed model loaded strictly from the local cache directory."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path = DEFAULT_MODEL_DIR,
        batch_size: int = 32,
    ) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._batch = batch_size
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir), local_files_only=True
        )
        self._dim = int(self.embed_query("dimension probe").shape[0])

    @property
    def name(self) -> str:
        """Cache key for this embedder's vectors."""
        return f"fastembed:{self._model_name}"

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Passage vectors, shape (n, dim)."""
        if not texts:
            return np.zeros((0, getattr(self, "_dim", 0)), dtype=np.float32)
        vecs = list(self._model.passage_embed(list(texts), batch_size=self._batch))
        return _normalize(np.stack(vecs))

    def embed_query(self, text: str) -> np.ndarray:
        """Query vector (uses the model's query instruction, if any), shape (dim,)."""
        return _normalize(next(iter(self._model.query_embed([text]))))


def embed_chunks(
    store: Store, embedder: Embedder, batch_size: int = 64
) -> tuple[list[int], np.ndarray, dict]:
    """Vectors for every chunk, using and filling the cache keyed by content hash.

    Returns ``(rowids, matrix, {"hits", "misses"})``. If the embedder yields no
    vectors (unavailable model), returns an empty index rather than raising.
    """
    rows = store.conn.execute(
        "SELECT rowid, content_hash, text FROM chunks ORDER BY rowid"
    ).fetchall()
    model = embedder.name
    cached: dict[str, np.ndarray] = {}
    for h, dim, blob in store.conn.execute(
        "SELECT content_hash, dim, vector FROM embeddings WHERE model = ?", (model,)
    ):
        cached[h] = np.frombuffer(blob, dtype=np.float32, count=dim)
    hits = sum(1 for _, h, _ in rows if h in cached)
    missing: dict[str, str] = {}
    for _, h, text in rows:
        if h not in cached:
            missing.setdefault(h, text)
    stats = {"hits": hits, "misses": len(rows) - hits}
    items = list(missing.items())
    with store.conn:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            vecs = embedder.embed_documents([t for _, t in batch])
            if vecs.shape[0] != len(batch):
                return [], np.zeros((0, embedder.dim), dtype=np.float32), stats
            for (h, _), v in zip(batch, vecs, strict=True):
                v = np.ascontiguousarray(v, dtype=np.float32)
                cached[h] = v
                store.conn.execute(
                    "INSERT OR REPLACE INTO embeddings(content_hash, model, dim, vector) "
                    "VALUES (?, ?, ?, ?)",
                    (h, model, int(v.shape[0]), v.tobytes()),
                )
    if not rows:
        return [], np.zeros((0, embedder.dim), dtype=np.float32), stats
    matrix = np.stack([cached[h] for _, h, _ in rows]).astype(np.float32)
    return [r[0] for r in rows], matrix, stats

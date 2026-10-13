"""Brute-force vector search and reciprocal rank fusion with BM25."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from coderag.embeddings import Embedder, embed_chunks
from coderag.lexical import search_lexical
from coderag.store import Hit, Store

RRF_K = 60


def rrf(
    rankings: Sequence[Sequence[Hashable]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[Hashable, float]]:
    """Reciprocal rank fusion: score(d) = sum_i w_i / (k + rank_i(d)), ranks from 1.

    Ties keep the order in which items were first seen (ranking by ranking).
    """
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    scores: dict[Hashable, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    order = {item: i for i, item in enumerate(scores)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


class VectorIndex:
    """In-memory matrix of chunk vectors, aligned with chunk rowids."""

    def __init__(self, store: Store, embedder: Embedder) -> None:
        self.embedder = embedder
        self.rowids, self.matrix, self.cache_stats = embed_chunks(store, embedder)

    def __len__(self) -> int:
        return len(self.rowids)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-``k`` (rowid, cosine) pairs."""
        if not len(self) or k <= 0:
            return []
        q = self.embedder.embed_query(query)
        sims = self.matrix @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.lexsort((top, -sims[top]))]
        return [(self.rowids[i], float(sims[i])) for i in top]


def vector_search(
    store: Store, embedder: Embedder, query: str, k: int, index: VectorIndex | None = None
) -> list[Hit]:
    """Top-``k`` chunks by cosine similarity."""
    index = index or VectorIndex(store, embedder)
    return store.hits_for(index.search(query, k))


class HybridSearcher:
    """BM25 + vector rankings fused with weighted RRF."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        rrf_k: int = RRF_K,
        w_lex: float = 1.0,
        w_vec: float = 0.5,
        depth: int = 50,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.w_lex = w_lex
        self.w_vec = w_vec
        self.depth = depth
        self._index: VectorIndex | None = None

    @property
    def index(self) -> VectorIndex:
        """Lazily built vector index."""
        if self._index is None:
            self._index = VectorIndex(self.store, self.embedder)
        return self._index

    def search_vector(self, query: str, k: int) -> list[Hit]:
        """Vector-only ranking."""
        return self.store.hits_for(self.index.search(query, k))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Fused top-``k``; falls back to BM25 alone when there are no vectors."""
        depth = max(self.depth, k)
        lex = [h.rowid for h in search_lexical(self.store, query, depth)]
        vec = [rid for rid, _ in self.index.search(query, depth)] if self.w_vec else []
        fused = rrf([lex, vec], weights=[self.w_lex, self.w_vec], k=self.rrf_k)
        return self.store.hits_for([(rid, s) for rid, s in fused[:k] if s > 0])

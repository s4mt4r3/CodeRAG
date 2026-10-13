"""M5: embeddings, embedding cache, RRF, and hybrid search (offline, HashEmbedder)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from coderag.embeddings import HashEmbedder, embed_chunks
from coderag.hybrid import HybridSearcher, rrf, vector_search
from coderag.lexical import search_lexical
from coderag.store import Store, index_repo
from conftest import FIXTURE_ROOT


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=64)
        self.calls = 0
        self.texts = 0

    def embed_documents(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return super().embed_documents(texts)


class EmptyEmbedder(HashEmbedder):
    """An embedder whose index is always empty (e.g. model unavailable)."""

    name = "empty"

    def embed_documents(self, texts):
        return np.zeros((0, self.dim), dtype=np.float32)


# ------------------------------------------------------------------ RRF


def test_rrf_hand_computed():
    lex = ["a", "b", "c"]
    vec = ["c", "a", "d"]
    out = dict(rrf([lex, vec], k=60))
    assert out["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert out["b"] == pytest.approx(1 / 62)
    assert out["d"] == pytest.approx(1 / 63)
    assert [x for x, _ in rrf([lex, vec], k=60)] == ["a", "c", "b", "d"]


def test_rrf_weights_and_empty():
    out = dict(rrf([["a"], ["b"]], weights=[2.0, 1.0], k=0))
    assert out == {"a": 2.0, "b": 1.0}
    assert rrf([[], []]) == []
    with pytest.raises(ValueError):
        rrf([["a"]], weights=[1.0, 2.0])


def test_rrf_tie_break_is_deterministic():
    a = rrf([["x", "y"], ["y", "x"]], k=60)
    b = rrf([["x", "y"], ["y", "x"]], k=60)
    assert a == b
    assert a[0][1] == a[1][1]


# ------------------------------------------------------------------ embedders


def test_hash_embedder_deterministic_and_normalized():
    e1, e2 = HashEmbedder(dim=128), HashEmbedder(dim=128)
    v1 = e1.embed_documents(["def parse_config(path): return load(path)", "x"])
    v2 = e2.embed_documents(["def parse_config(path): return load(path)", "x"])
    assert v1.shape == (2, 128) and v1.dtype == np.float32
    assert np.array_equal(v1, v2)
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)
    q = e1.embed_query("parse config")
    assert q.shape == (128,)
    assert float(v1[0] @ q) > float(v1[1] @ q)


def test_hash_embedder_empty_text():
    v = HashEmbedder(dim=32).embed_documents([""])
    assert v.shape == (1, 32)
    assert not np.isnan(v).any()


# ------------------------------------------------------------------ cache


@pytest.fixture
def repo_db(tmp_path: Path):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, repo, ignore=shutil.ignore_patterns("__pycache__"))
    db = tmp_path / "i.db"
    index_repo(repo, db)
    return repo, db


def test_embedding_cache_hit_and_miss(repo_db):
    repo, db = repo_db
    store = Store(db)
    emb = CountingEmbedder()
    rowids, mat, stats = embed_chunks(store, emb)
    n = len(store.all_chunks())
    assert len(rowids) == n and mat.shape == (n, 64)
    assert stats == {"hits": 0, "misses": n}
    _, mat2, stats2 = embed_chunks(store, emb)
    assert stats2 == {"hits": n, "misses": 0}
    assert np.array_equal(mat, mat2)

    target = repo / "shopcore" / "utils.py"
    target.write_text(target.read_text() + "\n\ndef shout(t):\n    return t.upper()\n")
    store.close()
    index_repo(repo, db)
    store = Store(db)
    _, _, stats3 = embed_chunks(store, emb)
    assert stats3["misses"] >= 1
    assert stats3["misses"] <= 2  # only the new function chunk (and maybe the module chunk)
    assert stats3["hits"] == len(store.all_chunks()) - stats3["misses"]


def test_cache_is_keyed_by_model(repo_db):
    _, db = repo_db
    store = Store(db)
    embed_chunks(store, HashEmbedder(dim=64))
    other = HashEmbedder(dim=32)
    _, mat, stats = embed_chunks(store, other)
    assert stats["hits"] == 0 and mat.shape[1] == 32


# ------------------------------------------------------------------ search


def test_vector_search_ranks_by_cosine(repo_db):
    _, db = repo_db
    store = Store(db)
    hits = vector_search(store, HashEmbedder(dim=256), "slugify text lowercase dash", k=5)
    assert len(hits) == 5
    assert hits[0].symbol == "shopcore.utils.slugify"
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_hybrid_returns_fused_ranking(repo_db):
    _, db = repo_db
    store = Store(db)
    hs = HybridSearcher(store, HashEmbedder(dim=256))
    hits = hs.search("normalize sku uppercase", k=5)
    assert hits[0].symbol == "shopcore.utils.normalize_sku"
    assert len({h.chunk_id for h in hits}) == len(hits)


def test_hybrid_degrades_gracefully_without_vectors(repo_db):
    _, db = repo_db
    store = Store(db)
    hs = HybridSearcher(store, EmptyEmbedder(dim=16))
    q = "render receipt"
    got = [h.chunk_id for h in hs.search(q, k=5)]
    lex = [h.chunk_id for h in search_lexical(store, q, k=5)]
    assert got == lex


def test_hybrid_weights_zero_vector_equals_lexical_order(repo_db):
    _, db = repo_db
    store = Store(db)
    hs = HybridSearcher(store, HashEmbedder(dim=64), w_lex=1.0, w_vec=0.0)
    q = "tax rate region"
    assert [h.chunk_id for h in hs.search(q, k=5)] == [
        h.chunk_id for h in search_lexical(store, q, k=5)
    ]

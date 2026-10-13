"""M3: identifier-aware tokenizer and BM25 search over FTS5."""

from __future__ import annotations

import pytest

from coderag.lexical import search_lexical, split_identifier, tokenize
from coderag.store import Store, index_repo
from conftest import FIXTURE_ROOT


def test_snake_camel_pascal_all_yield_parts():
    for word in ["parseConfig", "parse_config", "ParseConfig"]:
        toks = tokenize(word)
        assert "parse" in toks and "config" in toks, word
        assert word.lower() in toks, "original identifier is kept"


def test_acronyms():
    assert split_identifier("HTTPServer") == ["http", "server"]
    assert split_identifier("parseHTTPResponse") == ["parse", "http", "response"]
    assert split_identifier("XMLHttpRequest") == ["xml", "http", "request"]
    assert split_identifier("ID") == ["id"]


def test_digits():
    assert split_identifier("sha256Hash") == ["sha", "256", "hash"]
    assert split_identifier("utf8_decode") == ["utf", "8", "decode"]
    assert split_identifier("getHTTP2Response") == ["get", "http", "2", "response"]
    assert split_identifier("v2") == ["v", "2"]


def test_snake_and_dunder():
    assert split_identifier("__init__") == ["init"]
    assert split_identifier("_private_helper") == ["private", "helper"]
    assert split_identifier("SCREAMING_CASE_NAME") == ["screaming", "case", "name"]


def test_tokenize_text_and_unicode():
    toks = tokenize("def größe_bucket(x): return x.normalizeSku()")
    assert "größe_bucket" in toks and "größe" in toks and "bucket" in toks
    assert "normalizesku" in toks and "normalize" in toks and "sku" in toks


def test_query_stopwords_are_dropped():
    toks = tokenize("Where is the parse config defined?", query=True)
    assert "parse" in toks and "config" in toks
    assert "where" not in toks and "the" not in toks and "is" not in toks


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Store:
    db = tmp_path_factory.mktemp("idx") / "index.db"
    index_repo(FIXTURE_ROOT, db)
    return Store(db)


QUERIES = [
    ("slugify text", "shopcore.utils.slugify"),
    ("normalizeSku", "shopcore.utils.normalize_sku"),
    ("OrderRepository save", "storage.repository.OrderRepository.save"),
    ("loyalty discount tier", "shopcore.pricing.loyalty_discount"),
    ("render receipt", "service.notifications.Notifier.render_receipt"),
    ("sales tax rate for a region", "shopcore.pricing.tax_rate"),
    ("retry async job attempts", "service.workers.retry"),
    ("rebuild order from dict", "storage.serializers.order_from_dict"),
    ("allocate id", "service.orders.OrderService._allocate_id"),
    ("broadcast message to customers concurrently", "service.notifications.broadcast"),
    ("ErrorResponse status message", "service.api_handlers.error_response"),
]


@pytest.mark.parametrize(("query", "symbol"), QUERIES)
def test_hand_written_queries_find_the_right_chunk(store, query, symbol):
    hits = search_lexical(store, query, k=3)
    assert symbol in [h.symbol for h in hits], [h.symbol for h in hits]


def test_hits_are_ranked_with_scores(store):
    hits = search_lexical(store, "order", k=10)
    assert len(hits) == 10
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(h.path and h.start_line >= 1 and h.segments for h in hits)


def test_empty_and_punctuation_queries(store):
    assert search_lexical(store, "", k=5) == []
    assert search_lexical(store, "?? !! ()", k=5) == []
    assert search_lexical(store, 'quote"in AND OR NOT * near', k=5) is not None

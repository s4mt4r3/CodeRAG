# Retrieval comparison (recall@k / MRR)

Embedder: `fastembed:BAAI/bge-small-en-v1.5`, RRF k=60, w_lex=1.0, w_vec=0.5.

## Real repo (pallets/click src, seed 0)

| type | n | system | recall_at_1 | recall_at_5 | recall_at_10 | mrr |
|---|---|---|---|---|---|---|
| callers | 40 | bm25 | 0.1675 | 0.7358 | 0.7850 | 0.4662 |
| callers | 40 | vector | 0.0350 | 0.4308 | 0.5467 | 0.2948 |
| callers | 40 | hybrid | 0.1425 | 0.5658 | 0.6392 | 0.4029 |
| definition | 80 | bm25 | 0.7000 | 0.9125 | 0.9250 | 0.7854 |
| definition | 80 | vector | 0.7625 | 0.9375 | 0.9750 | 0.8471 |
| definition | 80 | hybrid | 0.7375 | 0.9250 | 0.9625 | 0.8275 |
| docstring_masked | 80 | bm25 | 0.8375 | 0.9375 | 0.9500 | 0.8810 |
| docstring_masked | 80 | vector | 0.5375 | 0.7750 | 0.8375 | 0.6388 |
| docstring_masked | 80 | hybrid | 0.7875 | 0.9000 | 0.9125 | 0.8385 |
| overall | 200 | bm25 | 0.6485 | 0.8872 | 0.9070 | 0.7598 |
| overall | 200 | vector | 0.5270 | 0.7712 | 0.8343 | 0.6533 |
| overall | 200 | hybrid | 0.6385 | 0.8432 | 0.8778 | 0.7470 |

## Fixture repo

| type | n | system | recall_at_1 | recall_at_5 | recall_at_10 | mrr |
|---|---|---|---|---|---|---|
| callers | 31 | bm25 | 0.1935 | 1.0000 | 1.0000 | 0.6022 |
| callers | 31 | vector | 0.0968 | 0.8468 | 0.9194 | 0.4917 |
| callers | 31 | hybrid | 0.1935 | 1.0000 | 1.0000 | 0.5941 |
| definition | 55 | bm25 | 0.8727 | 0.9818 | 1.0000 | 0.9261 |
| definition | 55 | vector | 0.9636 | 1.0000 | 1.0000 | 0.9818 |
| definition | 55 | hybrid | 0.8727 | 1.0000 | 1.0000 | 0.9309 |
| docstring_masked | 38 | bm25 | 0.9737 | 1.0000 | 1.0000 | 0.9868 |
| docstring_masked | 38 | vector | 0.8947 | 1.0000 | 1.0000 | 0.9430 |
| docstring_masked | 38 | hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| overall | 124 | bm25 | 0.7339 | 0.9919 | 1.0000 | 0.8637 |
| overall | 124 | vector | 0.7258 | 0.9617 | 0.9798 | 0.8474 |
| overall | 124 | hybrid | 0.7419 | 1.0000 | 1.0000 | 0.8679 |

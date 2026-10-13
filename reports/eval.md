# coderag evaluation

Repo: pallets/click (sha 8b19813f2bfca99f1018a587a8cf54fc959f2e5d), 200 auto-generated questions, seed 0. Generated 2026-10-12T18:28:38+00:00.

## Retrieval

| type | system | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| callers | bm25 | 40 | 0.1675 | 0.7358 | 0.7850 | 0.4662 |
| callers | vector | 40 | 0.0350 | 0.4308 | 0.5467 | 0.2948 |
| callers | hybrid | 40 | 0.1425 | 0.5658 | 0.6392 | 0.4029 |
| definition | bm25 | 80 | 0.7000 | 0.9125 | 0.9250 | 0.7854 |
| definition | vector | 80 | 0.7625 | 0.9375 | 0.9750 | 0.8471 |
| definition | hybrid | 80 | 0.7375 | 0.9250 | 0.9625 | 0.8275 |
| docstring_masked | bm25 | 80 | 0.8375 | 0.9375 | 0.9500 | 0.8810 |
| docstring_masked | vector | 80 | 0.5375 | 0.7750 | 0.8375 | 0.6388 |
| docstring_masked | hybrid | 80 | 0.7875 | 0.9000 | 0.9125 | 0.8385 |
| overall | bm25 | 200 | 0.6485 | 0.8872 | 0.9070 | 0.7598 |
| overall | vector | 200 | 0.5270 | 0.7712 | 0.8343 | 0.6533 |
| overall | hybrid | 200 | 0.6385 | 0.8432 | 0.8778 | 0.7470 |

## Agent

Mode: **scripted** (scripted). Pipeline check with a scripted LLM (search -> read top hit -> cite it). Measures tool dispatch, citation verification and grounding plumbing, NOT model answer quality. Token counts are synthetic scripted values.

| metric | value |
|---|---|
| questions | 30 |
| citation accuracy (verified / cited) | 1.0000 |
| answer grounded rate | 0.7000 |
| unsupported rate | 0.0000 |
| failure rate | 0.0000 |
| latency p50 (ms) | 21.0 |
| latency p95 (ms) | 45.0 |
| tokens in / out | 900 / 450 |
| estimated cost (USD) | n/a |

## Chunking ablation (AST vs fixed-size, same retriever)

Fixed-size chunks: 40 lines, 10 overlap. A retrieved chunk hits when it covers at least half of the truth span, so larger chunks hit more easily; compare the lines-per-chunk column.

| chunker | retriever | chunks | lines per top-5 chunk | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|---|
| ast | bm25 | 694 | 22.5 | 0.6485 | 0.8872 | 0.9070 | 0.7598 |
| ast | hybrid | 694 | 22.5 | 0.6385 | 0.8432 | 0.8778 | 0.7470 |
| fixed | bm25 | 426 | 39.4 | 0.4970 | 0.8232 | 0.8537 | 0.6463 |
| fixed | hybrid | 426 | 39.4 | 0.4770 | 0.7823 | 0.8390 | 0.6231 |

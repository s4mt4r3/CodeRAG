# Chunking ablation

## Chunking ablation (AST vs fixed-size, same retriever)

Fixed-size chunks: 40 lines, 10 overlap. A retrieved chunk hits when it covers at least half of the truth span, so larger chunks hit more easily; compare the lines-per-chunk column.

| chunker | retriever | chunks | lines per top-5 chunk | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|---|
| ast | bm25 | 694 | 22.5 | 0.6485 | 0.8872 | 0.9070 | 0.7598 |
| ast | hybrid | 694 | 22.5 | 0.6385 | 0.8432 | 0.8778 | 0.7470 |
| fixed | bm25 | 426 | 39.4 | 0.4970 | 0.8232 | 0.8537 | 0.6463 |
| fixed | hybrid | 426 | 39.4 | 0.4770 | 0.7823 | 0.8390 | 0.6231 |

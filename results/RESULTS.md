# Retrieval experiment results

The configuration was selected only on the development split. The held-out test
results below were reported after selection.

## Selected configuration

- Embedding model: `embeddinggemma`
- Chunk size: 200 words
- Overlap: 50 words
- Development Recall@5: 94.3%
- Development MRR@5: 0.805

## Held-out result

| k | Questions | Evidence Recall@k (95% CI) | Page Recall@k | MRR@k |
|---:|---:|---:|---:|---:|
| 1 | 15 | 66.7% (41.7%–84.8%) | 86.7% | 0.667 |
| 3 | 15 | 93.3% (70.2%–98.8%) | 100.0% | 0.789 |
| 5 | 15 | 100.0% (79.6%–100.0%) | 100.0% | 0.802 |
| 10 | 15 | 100.0% (79.6%–100.0%) | 100.0% | 0.802 |

## Best development setting within each model

| Model | Chunk words | Overlap words | Development Recall@5 |
|---|---:|---:|---:|
| `all-minilm` | 400 | 50 | 85.7% |
| `embeddinggemma` | 200 | 50 | 94.3% |
| `mxbai-embed-large` | 200 | 50 | 91.4% |
| `nomic-embed-text` | 200 | 50 | 91.4% |

## Interpretation limits

- Recall tests retrieval of reviewed evidence, not full generated-answer correctness.
- The 50 questions come from one engineering textbook, so results may not transfer unchanged to another corpus.
- Models with a 512-token context can truncate some 400- and 800-word chunks; the result reflects deployed behavior.
- Page recall is diagnostic only. Large chunks can touch a labeled page without carrying the answer, so evidence recall is the selection metric.

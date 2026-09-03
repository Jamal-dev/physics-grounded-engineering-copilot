# Retrieval study protocol

## Research question

> How do embedding model, word-based chunk size, overlap, and retrieval depth
> influence the probability that an engineering RAG system receives the
> answer-bearing evidence?

The study treats successful evidence retrieval as a necessary condition for a
reliable grounded answer. It does not equate retrieval success with complete
answer correctness.

## Corpus and questions

The private corpus contains Docling output from pages 298–483 of Reint de Boer's
*Theory of Porous Media*. This 186-page scope covers poroelastic formulations,
constitutive theory, transport, and applications. The source text is never
committed.

The public benchmark has 50 manually reviewed questions. Every label includes:

- one or more source pages;
- a paraphrased reference answer;
- exact evidence terms from the private passage; and
- a SHA-256 hash of that passage.

The fixed split contains 35 development and 15 test questions. The test items
are distributed through the page range rather than held out as one contiguous
chapter.

## Experimental factors

The full factorial grid contains 36 retrieval configurations and 144 metric
points:

| Factor | Values |
|---|---|
| Embedding | `all-minilm`, `nomic-embed-text`, `embeddinggemma`, `mxbai-embed-large` |
| Chunk size | 200, 400, 800 words |
| Overlap | 0, 50, 100 words |
| k | 1, 3, 5, 10 |

Embeddings are normalized and ranked with cosine similarity. Each model uses
its recommended asymmetric retrieval prompt when required:

| Model | Document prefix | Query prefix | Declared context |
|---|---|---|---:|
| `all-minilm` | none | none | 512 tokens |
| `nomic-embed-text` | `search_document:` | `search_query:` | 2,048 tokens |
| `embeddinggemma` | `title: … | text:` | `task: search result | query:` | 2,048 tokens |
| `mxbai-embed-large` | none | `Represent this sentence for searching relevant passages:` | 512 tokens |

Prompt and context choices follow the
[Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5),
[EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card),
and [Mixedbread model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).
The deployed Ollama tags are documented in
[`model_inventory.csv`](../results/model_inventory.csv).

## Metrics

For question \(q\), a chunk is relevant when it overlaps a labeled page and
contains every normalized evidence term. The primary metric is:

\[
\operatorname{Recall@k}=\frac{1}{N}\sum_{q=1}^{N}
\mathbb{1}[\text{a relevant chunk for }q\text{ appears in the top }k].
\]

Mean reciprocal rank at k captures whether the evidence appears early:

\[
\operatorname{MRR@k}=\frac{1}{N}\sum_{q=1}^{N}
\begin{cases}
1/r_q, & r_q \le k\\
0, & r_q > k.
\end{cases}
\]

Page Recall@k is a secondary diagnostic. It is intentionally not used for
selection because a long chunk can touch the labeled page without containing
the answer. Recall proportions include Wilson 95% confidence intervals.

## Selection rule

The chosen configuration maximizes development evidence Recall@5. Ties are
broken, in order, by:

1. higher development MRR@5;
2. lower measured retrieval latency;
3. smaller chunks;
4. lower overlap; and
5. model name, for deterministic resolution.

The held-out test split does not participate in this decision. Its Recall@1,
Recall@3, Recall@5, Recall@10, page recall, MRR, and confidence intervals are
reported after selection.

## Reproduce

From the activated `local-rag` environment:

```bash
python -m experiments.export_corpus --source theory_of_elasticity.pdf
python -m experiments.curate_questions
python -m experiments.validate_benchmark
python -m experiments.run_retrieval_benchmark
python -m experiments.analyze_results
python -m experiments.build_notebook
python -m experiments.qa_results
```

Model matrices are cached under `RAG_DATA_DIR/evaluation/cache`. Cache entries
include corpus, configuration, model-prompt, and local model-digest
fingerprints. The published result is in [`results/RESULTS.md`](../results/RESULTS.md),
and the executable analysis is in
[`notebooks/retrieval_choices.ipynb`](../notebooks/retrieval_choices.ipynb).

## Threats to validity

- Fifteen held-out questions produce wide uncertainty intervals.
- The questions and evidence terms were reviewed by one curator.
- Lexical evidence terms are conservative and may reject a semantically correct
  chunk that uses different notation.
- One textbook and one technical page range do not represent every engineering
  corpus.
- Word counts are not token counts. The 512-token models can truncate some
  400- and 800-word chunks; this is retained as part of deployed behavior and
  flagged in the results.
- Retrieval metrics do not measure final response completeness, numerical
  validity, or citation entailment. Those require a separate answer-level study.

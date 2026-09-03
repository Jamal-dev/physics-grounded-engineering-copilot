# Engineering retrieval benchmark

`questions.jsonl` contains 50 technical questions manually reviewed against
Reint de Boer's *Theory of Porous Media*. The book is not included in this
repository. Each row points to a page in a legally obtained local copy without
reproducing the source passage.

The retrieval experiment is scoped to the technical section on pages 298–483.
All benchmark answers lie inside that range.

## Fields

- `id`: stable question identifier.
- `question`: paraphrased retrieval query.
- `answer_summary`: concise, paraphrased reference answer.
- `source`: portable source filename, never a machine-specific path.
- `expected_pages`: document pages containing the reviewed answer.
- `evidence_terms`: exact terms checked in the private evidence passage.
- `evidence_sha256`: hash of that private Docling passage for auditability.
- `split`: `development` or held-out `test`.

The 35 development questions are used to select an embedding/chunking
configuration. The 15 test questions are used only for the final estimate. The
split is deterministic and distributed across the source-page range rather than
formed as one contiguous block.

## Relevance rule

A retrieved chunk is relevant only when it:

1. overlaps at least one `expected_pages` value; and
2. contains every normalized `evidence_terms` value.

The second condition prevents large chunks from receiving credit merely because
they span the correct page. `experiments.validate_benchmark` checks the page,
hash, evidence terms, unique IDs, unique questions, and split values against the
private exported corpus before any experiment runs.

This benchmark measures retrieval coverage. It does not by itself measure the
factual correctness, completeness, or citation faithfulness of generated prose.

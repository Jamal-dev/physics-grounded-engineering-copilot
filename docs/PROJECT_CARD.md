# Project card: Physics-Grounded Engineering Copilot

## Problem

Engineering assistants can sound convincing while retrieving the wrong source,
following instructions embedded in a document, inventing a citation, or sending
unsafe parameters to a numerical solver. This project makes those failure modes
visible and testable instead of hiding them behind a chat interface.

## Implemented system

1. Docling ingests engineering documents with source/page provenance.
2. Chroma separates document sets by evidence scope.
3. Retrieval rejects low-relevance and instruction-like passages.
4. The answer model must cite only passages actually supplied; a failed citation
   check triggers one constrained repair and otherwise an abstention.
5. A local LLM router chooses only between scoped evidence answering and one
   allowlisted thermal-diffusion tool.
6. Planner arguments not stated by the user are removed, then physical-domain,
   explicit-stability, and resource limits are applied.
7. The finite-difference result is checked against a Fourier-series solution,
   its discrete PDE residual, and the maximum principle.

## Verified evidence

| Evidence | Result |
|---|---:|
| Unit tests | 32 passing |
| Static checks | Ruff passing |
| Packaged entry point | Editable wheel installed and executed |
| Reference thermal case | Validation passed |
| Relative L2 error | 0.0045% |
| Maximum absolute error | 0.0072 K |
| Maximum-principle violation | 0 K |
| Live local agent route | `simulate_thermal_diffusion` selected and executed |
| Retrieval evaluation | 50 reviewed questions; 35 development / 15 held-out test |
| Retrieval design points | 144 |
| Selected retrieval configuration | `embeddinggemma`, 200 words, 50-word overlap |
| Development evidence Recall@5 | 94.3% |
| Held-out evidence Recall@5 | 100.0% (95% CI 79.6%–100.0%; n=15) |
| Held-out MRR@5 | 0.802 |
| Fine-tuning data validation | 136 train / 48 validation / 48 held-out test |
| Base model token F1 | 9.6% |
| Base model + RAG token F1 | 76.1% |
| Base model + fine-tuning token F1 | 28.7% |
| Base model + fine-tuning + RAG token F1 | 87.1% |

Machine-readable evidence, aggregate retrieval results, confidence intervals,
and plots are stored in `results/`; the executed analysis notebook is stored in
`notebooks/`.

## Fine-tuning boundary

The repository implements conversational-data validation, LoRA training,
adapter on/off inference, and a measured four-condition evaluation: base,
base+RAG, fine-tuned, and fine-tuned+RAG. The untouched 48-question test split
shows the highest token F1 for the combined system, but structural metrics expose
a failure mode: 34/48 fine-tuned-only outputs and 27/48 combined outputs reached
the 384-token cap, reducing valid JSON to 2.1% and 18.8%, respectively. This is
evidence to tune termination and output-format behavior before deployment, not a
claim that one condition is best on every reliability dimension.

## Five-minute review path

1. Run `make test` and inspect the security and numerical tests.
2. Run `python physics_cli.py` and compare the numerical and analytical metrics.
3. Start `./run.sh`, open **Agent workflow**, and execute the reference request.
4. Open **Retrieval study** to inspect held-out model/chunking trade-offs.
5. Read `docs/FINE_TUNING.md` for the leakage-aware 2x2 experiment design.

## Current limitations

- Retrieval evidence comes from one poromechanics textbook and should not be
  assumed to transfer unchanged to another engineering domain.
- The thermal tool is a transparent reference model, not a general industrial
  heat-transfer solver.
- Citation-number validation does not by itself prove semantic entailment.
- The LoRA result is one training run on one task collection; repeat seeds and
  expert review are needed for broader conclusions.

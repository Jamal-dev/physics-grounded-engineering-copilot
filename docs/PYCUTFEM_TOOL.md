# Adding pycutfem as a model tool

It is useful to add pycutfem when the assistant must verify or run a numerical
claim. Keep the roles separate:

- RAG supplies current documentation and theory.
- Fine-tuning teaches response structure and when a computation is appropriate.
- A pycutfem tool performs an actual, auditable computation.

Do not expose arbitrary Python, shell access, or the complete pycutfem API to
the model. Start with a few typed, allowlisted operations, for example:

```text
validate_problem(problem_spec) -> validation report
run_named_case(case_name, parameters, resolution) -> result id + metrics
inspect_result(result_id, fields) -> selected scalar/array summaries
```

## Recommended boundary

pycutfem and this RAG application use different Conda environments. Keep the
solver in its tested environment and communicate through JSON:

```text
model tool call
  -> document-rag tool registry validates JSON Schema
  -> bounded subprocess/service in the pycutfem environment
  -> pycutfem validates an allowlisted case and parameters
  -> JSON result, logs, provenance, and artifact paths
  -> model writes the user-facing explanation
```

The solver-side command can follow this shape:

```bash
conda run --no-capture-output -n fenicsx \
  python -m pycutfem.tools.run --request request.json --response response.json
```

Implement that entry point inside the pycutfem repository, where its own tests
and environment rules apply. The request must select a registered driver rather
than provide source code. Enforce schema validation, wall-time and mesh-size
limits, a dedicated output directory, deterministic seeds where relevant, and
captured solver/version provenance.

## Integration sequence

1. Define one read-only or cheap pycutfem operation and its JSON Schema.
2. Implement and test the solver-side dispatcher in the `fenicsx` environment.
3. Register the same schema in the model runtime and add an explicit tool loop:
   generate call, validate, execute, append the tool result, generate final answer.
4. Add training examples only for stable tool contracts. Do not fine-tune the
   model to memorize solver outputs.
5. Evaluate tool-call precision/recall, argument validity, execution success,
   numerical correctness, and final-answer correctness in addition to the four
   RAG/fine-tuning variants.

Begin with validation or named examples, not unrestricted simulations. That
gives the model useful numerical grounding while keeping execution predictable,
reviewable, and safe.

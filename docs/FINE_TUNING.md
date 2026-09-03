# Fine-tuning and four-way evaluation

This extension trains a small LoRA adapter while keeping retrieval and model
adaptation independent. It supports any Hugging Face causal chat model whose
tokenizer has a chat template; the included Qwen/FEM file is an example
configuration, not a hard-coded backend.

## Architecture

```text
fine_tuning/
  config.py       YAML loading and path resolution
  data.py         JSON/JSONL validation and held-out example extraction
  train.py        TRL SFTTrainer + PEFT LoRA
  inference.py    one base checkpoint with an adapter on/off switch
  comparison.py   shared retrieval and the four experimental conditions
  metrics.py      generic/reference/structured metrics and effect sizes
  evaluate.py     resumable single-query and held-out evaluation CLI
  configs/        model/task-specific examples only
```

The comparison is a 2x2 experiment:

| Variant | Retrieval | LoRA adapter |
|---|---:|---:|
| Base model | off | off |
| Base model + RAG | on | off |
| Base model + fine-tuning | off | on |
| Base model + RAG + fine-tuning | on | on |

One model object is loaded. For base runs, PEFT temporarily disables the
adapter. One retrieval result is also reused for both RAG variants. This avoids
accidentally comparing different checkpoints or different evidence.

## Prepare the FEM starter data

Training data is local and excluded from version control. From the repository
root:

```bash
mkdir -p .local_data/fem_weakform
cp /path/to/fem_qwen_finetune_starter/fem_weakform_sft_{train,val,test}.jsonl \
  .local_data/fem_weakform/
```

Validate the files and all training options without downloading a model:

```bash
conda run --no-capture-output -n local-rag \
  python -m fine_tuning.train \
  --config fine_tuning/configs/fem_weakform_qwen3.yaml \
  --dry-run
```

The dry run checks message roles, tool fields, duplicate IDs, configuration
keys, split hashes, precision selection, and adapter targets.

## Train

```bash
conda run --no-capture-output -n local-rag \
  python -m fine_tuning.train \
  --config fine_tuning/configs/fem_weakform_qwen3.yaml
```

The final adapter and a reproducibility manifest are written below the
configured `output_dir`. CPU training works, but it is expected to be slow.
With another model, copy the YAML and change `model.name_or_path`; keep
`target_modules: all-linear` unless that architecture needs an explicit module
list.

`assistant_only_loss: true` requires a compatible conversational chat template.
TRL patches known templates such as Qwen3. For an unknown model, supply a TRL
chat template with assistant-generation masks or set the option to `false` and
document why training on the complete conversation is acceptable. Values under
`chat_template` are passed to both training and inference, so model-specific
switches such as `enable_thinking` stay consistent.

## Evaluate a query in all four conditions

Index the relevant documents first, then run:

```bash
conda run --no-capture-output -n local-rag \
  python -m fine_tuning.evaluate \
  --config fine_tuning/configs/fem_weakform_qwen3.yaml \
  --query "Derive the weak form of the Poisson equation with mixed boundary conditions."
```

This is useful for inspection, but a single answer does not establish an
improvement.

## Evaluate a held-out split

```bash
conda run --no-capture-output -n local-rag \
  python -m fine_tuning.evaluate \
  --config fine_tuning/configs/fem_weakform_qwen3.yaml \
  --top-k 4 --resume
```

Each completed example is flushed to JSONL, so `--resume` can continue an
interrupted CPU run. The summary contains exact match, token F1, JSON validity,
configured JSON-field accuracy, tool-decision accuracy, and latency for every
variant. It also reports:

- fine-tuning lift without RAG;
- fine-tuning lift with RAG;
- RAG lift on the base model;
- RAG lift on the fine-tuned model; and
- the RAG/fine-tuning interaction.

Use the untouched test split only once the YAML, prompts, decoding, retrieval
settings, and field metrics have been selected on training/validation data.
Inspect failures as well as aggregate means; text overlap cannot by itself
establish mathematical correctness or citation faithfulness.

## Dataset contract

Input is a JSON array or JSONL with one object per example:

```json
{
  "id": "unique-id",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": []
}
```

Extra metadata is preserved. OpenAI-style `tools`, assistant `tool_calls`, and
`tool` messages are accepted, which keeps the starter's symbolic examples
usable. Tool schemas teach call selection and argument formatting; they do not
by themselves execute a tool at inference time.

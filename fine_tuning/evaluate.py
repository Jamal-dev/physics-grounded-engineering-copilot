"""Run base/RAG/fine-tuned/RAG+fine-tuned comparisons."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rag import retrieve

from .comparison import ComparisonRunner
from .config import load_config
from .data import evaluation_examples, load_and_validate
from .inference import AdapterChatModel
from .metrics import score_records


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name"):
            names.append(str(function["name"]))
    return names


def _read_completed(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_dataset(
    runner: ComparisonRunner,
    test_file: Path,
    output_file: Path,
    *,
    top_k: int,
    limit: int | None,
    resume: bool,
    json_fields: list[str],
) -> dict[str, Any]:
    rows, _ = load_and_validate(test_file)
    examples = evaluation_examples(rows)
    if limit is not None:
        examples = examples[:limit]
    records = _read_completed(output_file) if resume else []
    completed = {str(record["id"]) for record in records}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_file.exists() else "w"
    with output_file.open(mode, encoding="utf-8") as handle:
        for position, example in enumerate(examples, start=1):
            if example.id in completed:
                continue
            result = runner.run(
                example.query,
                prompt_messages=example.prompt_messages,
                tools=example.tools,
                top_k=top_k,
            )
            record = {
                "id": example.id,
                "query": example.query,
                "reference": example.reference,
                "expects_tool_call": example.expects_tool_call,
                "tool_names": _tool_names(example.tools),
                "metadata": example.metadata,
                **result.as_dict(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            print(f"[{position}/{len(examples)}] {example.id}", flush=True)

    # Ignore stale records when --resume is used with a smaller --limit.
    requested = {example.id for example in examples}
    selected = [record for record in records if str(record["id"]) in requested]
    summary = score_records(selected, json_fields=json_fields)
    summary.update(
        {
            "test_file": str(test_file),
            "predictions_file": str(output_file),
            "json_fields": json_fields,
        }
    )
    summary_path = output_file.with_name(f"{output_file.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", help="Adapter directory; defaults to config output/final")
    parser.add_argument("--query", help="Run one qualitative four-way comparison")
    parser.add_argument("--test-file", help="Held-out JSON/JSONL conversational dataset")
    parser.add_argument("--output", help="Prediction JSONL path")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--scope", help="RAG evidence scope; defaults to .env")
    parser.add_argument("--min-relevance-score", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model = AdapterChatModel.from_config(config, adapter_path=args.adapter)
    runner = ComparisonRunner(
        model,
        lambda query, k: retrieve(
            query,
            k=k,
            scope=args.scope,
            min_relevance_score=args.min_relevance_score,
        ),
    )
    if args.query:
        result = runner.run(args.query, top_k=args.top_k)
        payload = result.as_dict()
        if args.output:
            destination = Path(args.output).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    test_file = (
        Path(args.test_file).expanduser().resolve() if args.test_file else config.test_file
    )
    if test_file is None:
        parser.error("Provide --query, --test-file, or data.test_file in the config.")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else config.output_dir / "evaluation" / "predictions.jsonl"
    )
    fields = [str(field) for field in config.evaluation.get("json_fields", [])]
    summary = run_dataset(
        runner,
        test_file,
        output,
        top_k=args.top_k,
        limit=args.limit,
        resume=args.resume,
        json_fields=fields,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

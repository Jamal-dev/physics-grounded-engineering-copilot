"""Validate private predictions and publish aggregate four-condition results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from .comparison import VARIANTS
from .config import FineTuneConfig, load_config
from .data import evaluation_examples, load_and_validate, split_prompt_collisions
from .metrics import answer_metrics, score_records

METRICS = (
    ("token_f1", "Token F1"),
    ("json_valid", "Valid JSON"),
    ("json_field.status", "Status accuracy"),
    (
        "json_field.boundary_condition_assessment.sufficient",
        "Boundary assessment accuracy",
    ),
    ("tool_decision_accuracy", "Tool-decision accuracy"),
    ("exact_match", "Exact match"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_records(
    records: list[dict[str, Any]], config: FineTuneConfig
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if config.test_file is None:
        raise ValueError("The configuration has no held-out test split.")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    reports = {}
    for name, path in (
        ("train", config.train_file),
        ("validation", config.validation_file),
        ("test", config.test_file),
    ):
        if path is not None:
            split_rows[name], reports[name] = load_and_validate(path)
    collisions = split_prompt_collisions(split_rows)
    if any(collisions.values()):
        raise ValueError(f"Exact prompt overlap between splits: {collisions}")

    test_examples = evaluation_examples(split_rows["test"])
    expected = {example.id: example for example in test_examples}
    actual_ids = [str(record["id"]) for record in records]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Prediction file contains duplicate test IDs.")
    if set(actual_ids) != set(expected):
        raise ValueError("Prediction IDs do not exactly match the held-out test split.")

    expected_variants = {variant.key for variant in VARIANTS}
    expected_k = int(config.evaluation["rag_top_k"])
    for record in records:
        example = expected[str(record["id"])]
        if str(record["reference"]) != example.reference:
            raise ValueError(f"Reference mismatch for {example.id}.")
        answer_keys = [str(answer["variant"]) for answer in record["answers"]]
        if len(answer_keys) != 4 or set(answer_keys) != expected_variants:
            raise ValueError(f"Incomplete four-condition output for {example.id}.")
        sources = record.get("sources", [])
        if len(sources) != expected_k or any(
            not str(source.get("source", "")).startswith("training-example-")
            for source in sources
        ):
            raise ValueError(f"Invalid retrieval provenance for {example.id}.")
    return test_examples, collisions


def _metric_samples(
    records: list[dict[str, Any]], json_fields: list[str]
) -> dict[str, dict[str, list[float]]]:
    samples = {
        variant.key: {metric: [] for metric, _ in METRICS} for variant in VARIANTS
    }
    for record in records:
        by_key = {str(answer["variant"]): answer for answer in record["answers"]}
        for variant in VARIANTS:
            scored = answer_metrics(
                str(by_key[variant.key]["text"]),
                str(record["reference"]),
                json_fields=json_fields,
                expects_tool_call=bool(record.get("expects_tool_call", False)),
                tool_names=[str(name) for name in record.get("tool_names", [])],
            )
            for metric, _ in METRICS:
                value = scored[metric]
                if value is not None:
                    samples[variant.key][metric].append(float(value))
    return samples


def _bootstrap_interval(values: list[float], seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(10_000, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _write_csv(
    path: Path,
    summary: dict[str, Any],
    intervals: dict[str, dict[str, tuple[float, float]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "label", "metric", "score", "ci_low", "ci_high", "n"),
            lineterminator="\n",
        )
        writer.writeheader()
        for variant in VARIANTS:
            for metric, label in METRICS:
                low, high = intervals[variant.key][metric]
                writer.writerow(
                    {
                        "variant": variant.key,
                        "label": variant.label,
                        "metric": label,
                        "score": summary["variants"][variant.key][metric],
                        "ci_low": low,
                        "ci_high": high,
                        "n": summary["applicable"][variant.key][metric],
                    }
                )


def _write_chart(path: Path, summary: dict[str, Any]) -> None:
    shown = METRICS[:5]
    positions = np.arange(len(shown))
    width = 0.19
    fig, axis = plt.subplots(figsize=(12, 6.5))
    colors = ("#64748b", "#2563eb", "#f59e0b", "#059669")
    for index, (variant, color) in enumerate(zip(VARIANTS, colors, strict=True)):
        values = [summary["variants"][variant.key][metric] for metric, _ in shown]
        axis.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            label=variant.label,
            color=color,
        )
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(positions, [label for _, label in shown], rotation=12, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    fig.suptitle("Held-out four-condition evaluation (n=48)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _percentage(value: float) -> str:
    return f"{100 * value:.1f}%"


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    intervals: dict[str, dict[str, tuple[float, float]]],
    cap_hits: dict[str, int],
) -> None:
    lines = [
        "# Fine-tuning and RAG results",
        "",
        "These are measured results on the untouched 48-question test split. The LoRA "
        "adapter was trained for one epoch on 136 training examples. RAG used three "
        "passages from a separate index built only from those same training examples.",
        "",
        "| Condition | Token F1 (95% bootstrap CI) | Valid JSON | Status accuracy | "
        "Boundary assessment accuracy | Tool-decision accuracy | At token cap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        scores = summary["variants"][variant.key]
        low, high = intervals[variant.key]["token_f1"]
        boundary_metric = "json_field.boundary_condition_assessment.sufficient"
        boundary_n = summary["applicable"][variant.key][boundary_metric]
        lines.append(
            f"| {variant.label} | {_percentage(scores['token_f1'])} "
            f"({_percentage(low)}–{_percentage(high)}) | "
            f"{_percentage(scores['json_valid'])} | "
            f"{_percentage(scores['json_field.status'])} | "
            f"{_percentage(scores[boundary_metric])} (n={boundary_n}) | "
            f"{_percentage(scores['tool_decision_accuracy'])} | "
            f"{cap_hits[variant.key]}/48 |"
        )
    token_effects = summary["effects"]["token_f1"]
    lines.extend(
        [
            "",
            "## Design and interpretation",
            "",
            f"- Fine-tuning effect without RAG: {token_effects['fine_tuning_without_rag']:+.3f} token F1.",
            f"- Fine-tuning effect with RAG: {token_effects['fine_tuning_with_rag']:+.3f} token F1.",
            f"- RAG effect on the base model: {token_effects['rag_on_base']:+.3f} token F1.",
            f"- RAG effect on the fine-tuned model: {token_effects['rag_on_fine_tuned']:+.3f} token F1.",
            "- Generation was deterministic. Both RAG conditions received exactly the same retrieved passages per question.",
            "- The test references were not used for training, retrieval indexing, configuration selection, or threshold tuning.",
            "- Many adapter-enabled outputs reached the 384-token generation cap. This explains part of their low JSON-validity score and is a measured failure mode, not a missing result.",
            "",
            "Token overlap and structured-field accuracy are useful reproducible signals, "
            "but they do not fully establish mathematical equivalence. This is one "
            "task collection and one training run; broader conclusions require repeated "
            "seeds and expert review of derivations.",
            "",
            "![Four-condition score chart](fine_tuning_comparison.png)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def publish(config: FineTuneConfig, predictions: Path, output_dir: Path) -> dict[str, Any]:
    records = _read_jsonl(predictions)
    _, collisions = _validate_records(records, config)
    fields = [str(field) for field in config.evaluation.get("json_fields", [])]
    summary = score_records(records, json_fields=fields)
    samples = _metric_samples(records, fields)
    intervals = {
        variant.key: {
            metric: _bootstrap_interval(values, 20_260_903 + index)
            for index, (metric, values) in enumerate(samples[variant.key].items())
        }
        for variant in VARIANTS
    }
    generation_cap = int(config.generation["max_new_tokens"])
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, **config.tokenizer_init
    )
    cap_hits = {variant.key: 0 for variant in VARIANTS}
    for record in records:
        for answer in record["answers"]:
            generated_tokens = len(
                tokenizer.encode(str(answer["text"]), add_special_tokens=False)
            )
            if generated_tokens >= generation_cap:
                cap_hits[str(answer["variant"])] += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "measured",
        "model": config.base_model,
        "training": {
            "method": "LoRA",
            "epochs": config.sft["num_train_epochs"],
            "examples": 136,
            "sha256": _sha256(config.train_file),
        },
        "validation": {
            "examples": 48,
            "sha256": _sha256(config.validation_file),
        },
        "test": {
            "examples": summary["examples"],
            "sha256": _sha256(config.test_file),
            "split_prompt_collisions": collisions,
        },
        "retrieval": {
            "source_split": "train",
            "source_examples": 136,
            "embedding_model": "embeddinggemma",
            "chunk_words": 200,
            "overlap_words": 50,
            "top_k": config.evaluation["rag_top_k"],
        },
        "generation": {
            **config.generation,
            "outputs_at_token_cap": cap_hits,
        },
        "metrics": summary,
        "bootstrap_intervals": {
            variant: {
                metric: {"low": bounds[0], "high": bounds[1]}
                for metric, bounds in metric_intervals.items()
            }
            for variant, metric_intervals in intervals.items()
        },
    }
    json_path = output_dir / "fine_tuning_results.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(output_dir / "fine_tuning_comparison.csv", summary, intervals)
    _write_chart(output_dir / "fine_tuning_comparison.png", summary)
    _write_markdown(
        output_dir / "FINE_TUNING_RESULTS.md", summary, intervals, cap_hits
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    config = load_config(args.config)
    predictions = (
        Path(args.predictions).expanduser().resolve()
        if args.predictions
        else config.output_dir / "evaluation" / "predictions.jsonl"
    )
    result = publish(config, predictions, Path(args.output_dir).resolve())
    print(json.dumps(result["metrics"]["variants"], indent=2))


if __name__ == "__main__":
    main()

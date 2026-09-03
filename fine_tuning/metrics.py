"""Reference-based metrics and 2x2 treatment-effect summaries."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .comparison import VARIANTS


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected or overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def parse_json_answer(text: str) -> Any | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _nested(value: Any, field: str) -> Any:
    current = value
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _mentions_tool(text: str, tool_names: Sequence[str]) -> bool:
    lowered = text.lower()
    return "<tool_call" in lowered or any(name.lower() in lowered for name in tool_names)


def answer_metrics(
    prediction: str,
    reference: str,
    *,
    json_fields: Sequence[str] = (),
    expects_tool_call: bool = False,
    tool_names: Sequence[str] = (),
) -> dict[str, float | None]:
    predicted_json = parse_json_answer(prediction)
    reference_json = parse_json_answer(reference)
    result = {
        "exact_match": float(_normalise(prediction) == _normalise(reference)),
        "token_f1": token_f1(prediction, reference),
        "json_valid": float(predicted_json is not None),
        "tool_decision_accuracy": float(
            _mentions_tool(prediction, tool_names) == expects_tool_call
        ),
    }
    for field in json_fields:
        reference_value = _nested(reference_json, field)
        result[f"json_field.{field}"] = (
            None
            if reference_json is None or reference_value is None
            else float(
                predicted_json is not None
                and _nested(predicted_json, field) == reference_value
            )
        )
    return result


def _mean(values: Iterable[float | None]) -> float:
    items = [value for value in values if value is not None and not math.isnan(value)]
    return sum(items) / len(items) if items else math.nan


def _effects(by_variant: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    keys = set.intersection(*(set(metrics) for metrics in by_variant.values()))
    effects: dict[str, Any] = {}
    for metric in sorted(keys):
        base = by_variant["base"][metric]
        base_rag = by_variant["base_rag"][metric]
        tuned = by_variant["fine_tuned"][metric]
        tuned_rag = by_variant["fine_tuned_rag"][metric]
        effects[metric] = {
            "fine_tuning_without_rag": tuned - base,
            "fine_tuning_with_rag": tuned_rag - base_rag,
            "rag_on_base": base_rag - base,
            "rag_on_fine_tuned": tuned_rag - tuned,
            "rag_fine_tuning_interaction": (
                tuned_rag - tuned - (base_rag - base)
            ),
        }
    return effects


def score_records(
    records: Sequence[Mapping[str, Any]], *, json_fields: Sequence[str] = ()
) -> dict[str, Any]:
    collected: dict[str, list[dict[str, float | None]]] = {
        variant.key: [] for variant in VARIANTS
    }
    latencies: dict[str, list[float]] = {variant.key: [] for variant in VARIANTS}
    for record in records:
        reference = str(record["reference"])
        expected_tool = bool(record.get("expects_tool_call", False))
        tool_names = [str(name) for name in record.get("tool_names", [])]
        for answer in record["answers"]:
            key = str(answer["variant"])
            collected[key].append(
                answer_metrics(
                    str(answer["text"]),
                    reference,
                    json_fields=json_fields,
                    expects_tool_call=expected_tool,
                    tool_names=tool_names,
                )
            )
            latencies[key].append(float(answer.get("latency_seconds", 0.0)))

    by_variant: dict[str, dict[str, float]] = {}
    applicable: dict[str, dict[str, int]] = {}
    for variant in VARIANTS:
        rows = collected[variant.key]
        names = rows[0].keys() if rows else ()
        by_variant[variant.key] = {
            name: _mean(row[name] for row in rows) for name in names
        }
        by_variant[variant.key]["latency_seconds"] = _mean(latencies[variant.key])
        applicable[variant.key] = {
            name: sum(row[name] is not None for row in rows) for name in names
        }
        applicable[variant.key]["latency_seconds"] = len(latencies[variant.key])
    return {
        "examples": len(records),
        "variants": by_variant,
        "applicable": applicable,
        "effects": _effects(by_variant) if records else {},
    }

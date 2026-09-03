"""Conversational dataset validation and held-out example extraction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class DatasetReport:
    path: Path
    examples: int
    assistant_messages: int
    tool_call_examples: int
    sha256: str
    duplicate_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "examples": self.examples,
            "assistant_messages": self.assistant_messages,
            "tool_call_examples": self.tool_call_examples,
            "sha256": self.sha256,
            "duplicate_ids": list(self.duplicate_ids),
        }


@dataclass(frozen=True)
class EvaluationExample:
    id: str
    query: str
    prompt_messages: list[dict[str, Any]]
    reference: str
    tools: list[dict[str, Any]]
    expects_tool_call: bool
    metadata: dict[str, Any]


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if source.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_number} is not an object.")
                rows.append(value)
        return rows
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"{source} must contain an array of objects.")
        return value
    raise ValueError("Fine-tuning data must be .json or .jsonl.")


def _validate_messages(messages: Any, row_label: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{row_label}: messages must be a non-empty list.")
    checked: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{row_label}: messages[{index}] must be an object.")
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{row_label}: unsupported role {role!r}.")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError(f"{row_label}: messages[{index}].content must be text.")
        if role == "assistant" and not content and not message.get("tool_calls"):
            raise ValueError(
                f"{row_label}: assistant message {index} has no content or tool call."
            )
        checked.append(dict(message))
    if not any(message["role"] == "user" for message in checked):
        raise ValueError(f"{row_label}: at least one user message is required.")
    if not any(message["role"] == "assistant" for message in checked):
        raise ValueError(f"{row_label}: at least one assistant message is required.")
    return checked


def validate_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> DatasetReport:
    source = Path(path).expanduser().resolve()
    ids: list[str] = []
    assistant_messages = 0
    tool_call_examples = 0
    for index, row in enumerate(rows):
        row_id = str(row.get("id", index))
        messages = _validate_messages(row.get("messages"), row_id)
        ids.append(row_id)
        assistant_messages += sum(m["role"] == "assistant" for m in messages)
        tool_call_examples += int(any(m.get("tool_calls") for m in messages))
        tools = row.get("tools", [])
        if tools is not None and not isinstance(tools, list):
            raise ValueError(f"{row_id}: tools must be a list when present.")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for row_id in ids:
        if row_id in seen:
            duplicates.add(row_id)
        seen.add(row_id)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return DatasetReport(
        path=source,
        examples=len(rows),
        assistant_messages=assistant_messages,
        tool_call_examples=tool_call_examples,
        sha256=digest,
        duplicate_ids=tuple(sorted(duplicates)),
    )


def load_and_validate(path: str | Path) -> tuple[list[dict[str, Any]], DatasetReport]:
    rows = read_rows(path)
    report = validate_rows(rows, path)
    if report.duplicate_ids:
        joined = ", ".join(report.duplicate_ids[:5])
        raise ValueError(f"Duplicate dataset ids: {joined}")
    return rows, report


def evaluation_examples(rows: Iterable[Mapping[str, Any]]) -> list[EvaluationExample]:
    """Convert complete conversations to prompts plus held-out final answers."""
    examples: list[EvaluationExample] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id", index))
        messages = _validate_messages(row.get("messages"), row_id)
        final_index = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i]["role"] == "assistant" and messages[i].get("content")
            ),
            None,
        )
        if final_index is None:
            raise ValueError(f"{row_id}: no final assistant text to evaluate.")
        prompt_messages = messages[:final_index]
        # Held-out evaluation starts before the first model response. Tool loops
        # can be scored separately through expects_tool_call.
        first_assistant = next(
            (i for i, message in enumerate(prompt_messages) if message["role"] == "assistant"),
            len(prompt_messages),
        )
        prompt_messages = prompt_messages[:first_assistant]
        users = [m for m in prompt_messages if m["role"] == "user"]
        if not users:
            raise ValueError(f"{row_id}: no user query before the answer.")
        ignored = {"id", "messages", "tools"}
        examples.append(
            EvaluationExample(
                id=row_id,
                query=str(users[-1].get("content", "")),
                prompt_messages=prompt_messages,
                reference=str(messages[final_index]["content"]),
                tools=list(row.get("tools") or []),
                expects_tool_call=any(m.get("tool_calls") for m in messages),
                metadata={key: value for key, value in row.items() if key not in ignored},
            )
        )
    return examples


def split_prompt_collisions(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    """Count exact conversational user-prompt overlap between named splits."""

    prompt_sets: dict[str, set[str]] = {}
    for name, rows in splits.items():
        prompts: set[str] = set()
        for index, row in enumerate(rows):
            messages = _validate_messages(row.get("messages"), str(row.get("id", index)))
            user_turns = [
                str(message.get("content", "")).strip()
                for message in messages
                if message["role"] == "user"
            ]
            prompts.add("\n\n".join(user_turns))
        prompt_sets[name] = prompts
    return {
        f"{left}_{right}": len(prompt_sets[left] & prompt_sets[right])
        for left, right in combinations(prompt_sets, 2)
    }

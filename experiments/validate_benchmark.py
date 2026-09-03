"""Validate question labels against the private corpus before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from experiments.common import DEFAULT_CORPUS, DEFAULT_QUESTIONS, read_jsonl


def normalize(text: str) -> str:
    return re.sub(
        r"\s+", " ", text.lower().replace("\u00ad ", "").replace("\u00ad", "")
    ).strip()


def validate(corpus_path: Path, questions_path: Path) -> dict[str, int]:
    corpus = read_jsonl(corpus_path)
    questions = read_jsonl(questions_path)
    by_hash = {
        hashlib.sha256(row["text"].encode("utf-8")).hexdigest(): row for row in corpus
    }
    errors = []
    ids = [row["id"] for row in questions]
    if len(ids) != len(set(ids)):
        errors.append("Question IDs are not unique")
    if len({normalize(row["question"]) for row in questions}) != len(questions):
        errors.append("Question texts are not unique")

    for question in questions:
        evidence = by_hash.get(question["evidence_sha256"])
        if evidence is None:
            errors.append(f"{question['id']}: evidence hash not found")
            continue
        if sorted(evidence["pages"]) != sorted(question["expected_pages"]):
            errors.append(f"{question['id']}: expected pages do not match evidence")
        evidence_text = normalize(evidence["text"])
        for term in question["evidence_terms"]:
            if normalize(term) not in evidence_text:
                errors.append(f"{question['id']}: evidence term absent: {term}")
        if not question["question"].strip().endswith("?"):
            errors.append(f"{question['id']}: question lacks question mark")
        if question["split"] not in {"development", "test"}:
            errors.append(f"{question['id']}: invalid split")

    if errors:
        raise RuntimeError("\n".join(errors))
    return {
        "questions": len(questions),
        "development": sum(row["split"] == "development" for row in questions),
        "test": sum(row["split"] == "test" for row in questions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args()
    counts = validate(args.corpus.expanduser().resolve(), args.questions.expanduser().resolve())
    print(f"Benchmark valid: {counts}")


if __name__ == "__main__":
    main()

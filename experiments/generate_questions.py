"""Generate a page-grounded benchmark from private corpus passages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import requests

from experiments.common import DEFAULT_CORPUS, DEFAULT_QUESTIONS, read_jsonl, write_jsonl

EXCLUDED_HEADINGS = ("references", "author index", "subject index", "contents")


def _heading(text: str) -> str:
    return text.splitlines()[0].strip()[:160]


def select_evidence(
    records: list[dict[str, Any]], count: int, first_page: int, last_page: int
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        pages = record["pages"]
        text = record["text"]
        heading = _heading(text).lower()
        if not pages or pages[0] < first_page or pages[-1] > last_page:
            continue
        if len(text.split()) < 90 or "formula-not-decoded" in text:
            continue
        if re.search(r"\(\d{4}[–-]\d{4}\)", _heading(text)):
            continue
        if any(value in heading for value in EXCLUDED_HEADINGS):
            continue
        candidates.append(record)
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} suitable evidence passages found")

    selected = []
    width = (last_page - first_page + 1) / count
    for index in range(count):
        low = first_page + index * width
        high = first_page + (index + 1) * width
        bucket = [
            row
            for row in candidates
            if low <= sum(row["pages"]) / len(row["pages"]) < high
            and row not in selected
        ]
        if not bucket:
            target = (low + high) / 2
            bucket = sorted(
                (row for row in candidates if row not in selected),
                key=lambda row: abs(sum(row["pages"]) / len(row["pages"]) - target),
            )[:8]
        selected.append(max(bucket, key=lambda row: len(row["text"].split())))
    return selected


def _generate_batch(
    rows: list[dict[str, Any]], model: str, base_url: str
) -> list[dict[str, Any]]:
    evidence = []
    for row in rows:
        evidence.append(
            {
                "id": row["candidate_id"],
                "page": row["pages"],
                "passage": row["text"][:2600],
            }
        )
    prompt = f"""Create exactly one engineering or porous-media question per passage.

Rules:
- The answer must be explicitly supported by that passage, not merely by the chapter heading.
- Ask a self-contained technical question suitable for evaluating document retrieval.
- Paraphrase rather than copying a sentence.
- Avoid questions about biographies, dates, correspondence, or the book itself.
- Give a concise paraphrased answer summary, 8-30 words.
- Give 2-5 lowercase evidence terms that actually occur in the passage.
- Preserve each id exactly. Return only valid JSON with an `items` array.

Passages:
{json.dumps(evidence, ensure_ascii=False)}
"""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "answer_summary": {"type": "string"},
                        "evidence_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["id", "question", "answer_summary", "evidence_terms"],
                },
            }
        },
        "required": ["items"],
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1800},
        },
        timeout=600,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])["items"]


def generate_questions(
    corpus_path: Path,
    output: Path,
    count: int,
    first_page: int,
    last_page: int,
    model: str,
    base_url: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    selected = select_evidence(
        read_jsonl(corpus_path), count, first_page, last_page
    )
    candidates = []
    for index, row in enumerate(selected, start=1):
        candidates.append(
            {
                **row,
                "candidate_id": f"Q{index:03d}",
                "evidence_sha256": hashlib.sha256(
                    row["text"].encode("utf-8")
                ).hexdigest(),
            }
        )

    generated: dict[str, dict[str, Any]] = {}
    batches = math.ceil(len(candidates) / batch_size)
    for batch_number, start in enumerate(range(0, len(candidates), batch_size), start=1):
        batch = candidates[start : start + batch_size]
        print(f"Generating question batch {batch_number}/{batches}...")
        for item in _generate_batch(batch, model, base_url):
            generated[item["id"]] = item

    rows = []
    for index, candidate in enumerate(candidates, start=1):
        item = generated.get(candidate["candidate_id"])
        if not item:
            raise RuntimeError(f"No generated question for {candidate['candidate_id']}")
        question = re.sub(r"\s+", " ", item["question"]).strip()
        if not question.endswith("?"):
            question += "?"
        passage_lower = candidate["text"].lower()
        evidence_terms = [
            str(term).strip().lower()
            for term in item["evidence_terms"]
            if str(term).strip().lower() in passage_lower
        ]
        if len(evidence_terms) < 2:
            raise RuntimeError(
                f"Insufficient verifiable evidence terms for {candidate['candidate_id']}"
            )
        rows.append(
            {
                "id": candidate["candidate_id"],
                "question": question,
                "answer_summary": re.sub(
                    r"\s+", " ", item["answer_summary"]
                ).strip(),
                "source": candidate["source"],
                "expected_pages": candidate["pages"],
                "evidence_terms": evidence_terms,
                "evidence_sha256": candidate["evidence_sha256"],
                "split": "test" if index % 10 in {0, 3, 7} else "development",
            }
        )
    write_jsonl(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--first-page", type=int, default=298)
    parser.add_argument("--last-page", type=int, default=483)
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    rows = generate_questions(
        args.corpus.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.count,
        args.first_page,
        args.last_page,
        args.model,
        args.base_url,
        args.batch_size,
    )
    splits = {name: sum(row["split"] == name for row in rows) for name in ("development", "test")}
    print(f"Wrote {len(rows)} questions to {args.output}: {splits}")


if __name__ == "__main__":
    main()

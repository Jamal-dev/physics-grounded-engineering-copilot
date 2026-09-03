"""Shared corpus, chunking, and embedding helpers for retrieval experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
PRIVATE_DATA_DIR = Path(os.getenv("RAG_DATA_DIR", PROJECT_DIR / ".data")).expanduser()
if not PRIVATE_DATA_DIR.is_absolute():
    PRIVATE_DATA_DIR = (PROJECT_DIR / PRIVATE_DATA_DIR).resolve()
DEFAULT_CORPUS = PRIVATE_DATA_DIR / "evaluation" / "corpus.jsonl"
DEFAULT_QUESTIONS = PROJECT_DIR / "benchmark" / "questions.jsonl"
DEFAULT_RESULTS = PROJECT_DIR / "results"
DEFAULT_CACHE = PRIVATE_DATA_DIR / "evaluation" / "cache"

CHUNK_SIZES = (200, 400, 800)
OVERLAPS = (0, 50, 100)
TOP_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    document_prefix: str = ""
    query_prefix: str = ""
    context_tokens: int | None = None


EMBEDDING_MODELS = (
    EmbeddingModel(
        name="all-minilm",
        context_tokens=512,
    ),
    EmbeddingModel(
        name="nomic-embed-text",
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        context_tokens=2048,
    ),
    EmbeddingModel(
        name="embeddinggemma",
        document_prefix="title: Theory of Porous Media | text: ",
        query_prefix="task: search result | query: ",
        context_tokens=2048,
    ),
    EmbeddingModel(
        name="mxbai-embed-large",
        query_prefix="Represent this sentence for searching relevant passages: ",
        context_tokens=512,
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_pages(value: Any) -> list[int]:
    if isinstance(value, list):
        return sorted({int(page) for page in value})
    return sorted({int(page) for page in re.findall(r"\d+", str(value or ""))})


def batched(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def rechunk(
    records: Sequence[dict[str, Any]], chunk_size: int, overlap: int
) -> list[dict[str, Any]]:
    """Create deterministic word windows while preserving source-page provenance."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words: list[str] = []
    word_pages: list[tuple[int, ...]] = []
    for record in sorted(records, key=lambda row: int(row["sequence"])):
        pages = tuple(parse_pages(record["pages"]))
        record_words = str(record["text"]).split()
        words.extend(record_words)
        word_pages.extend([pages] * len(record_words))

    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        if end <= start:
            break
        pages = sorted({page for group in word_pages[start:end] for page in group})
        text = " ".join(words[start:end])
        chunks.append(
            {
                "chunk_id": f"w{chunk_size}-o{overlap}-{start:07d}",
                "start_word": start,
                "end_word": end,
                "word_count": end - start,
                "pages": pages,
                "text": text,
            }
        )
        if end == len(words):
            break
    return chunks


class OllamaEmbedder:
    """Small Ollama embedding client with timings and bounded batches."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        batch_size: int = 32,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(
        self, model: EmbeddingModel, texts: Sequence[str], *, kind: str
    ) -> tuple[np.ndarray, float]:
        prefix = model.query_prefix if kind == "query" else model.document_prefix
        prepared = [prefix + text for text in texts]
        arrays: list[np.ndarray] = []
        started = time.perf_counter()
        for batch in batched(prepared, self.batch_size):
            payload: dict[str, Any] = {
                "model": model.name,
                "input": batch,
                "truncate": True,
            }
            if model.context_tokens is not None:
                payload["options"] = {"num_ctx": model.context_tokens}
            response = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            arrays.append(np.asarray(response.json()["embeddings"], dtype=np.float32))
        elapsed = time.perf_counter() - started
        matrix = np.vstack(arrays)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12), elapsed


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")

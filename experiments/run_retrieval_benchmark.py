"""Run the embedding, chunk-size, overlap, and top-k retrieval experiment."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import requests

from experiments.common import (
    CHUNK_SIZES,
    DEFAULT_CACHE,
    DEFAULT_CORPUS,
    DEFAULT_QUESTIONS,
    DEFAULT_RESULTS,
    EMBEDDING_MODELS,
    OVERLAPS,
    TOP_K_VALUES,
    EmbeddingModel,
    OllamaEmbedder,
    model_slug,
    read_jsonl,
    rechunk,
    stable_hash,
)
from experiments.validate_benchmark import normalize, validate


@contextmanager
def _exclusive_run_lock(path: Path):
    """Prevent concurrent benchmark writers from corrupting caches or results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another retrieval benchmark is already using {path.parent}"
            ) from exc
        handle.write(str(time.time_ns()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def _cached_embeddings(
    cache_path: Path,
    texts: list[str],
    model: EmbeddingModel,
    kind: str,
    embedder: OllamaEmbedder,
    fingerprint: str,
) -> tuple[np.ndarray, float, bool]:
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["fingerprint"].item()) == fingerprint:
            return cached["embeddings"], float(cached["elapsed_seconds"].item()), True
    embeddings, elapsed = embedder.embed(model, texts, kind=kind)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        elapsed_seconds=np.asarray(elapsed),
        fingerprint=np.asarray(fingerprint),
    )
    return embeddings, elapsed, False


def _evidence_relevant(chunk: dict[str, Any], question: dict[str, Any]) -> bool:
    if not set(chunk["pages"]).intersection(question["expected_pages"]):
        return False
    text = normalize(chunk["text"])
    return all(normalize(term) in text for term in question["evidence_terms"])


def _page_relevant(chunk: dict[str, Any], question: dict[str, Any]) -> bool:
    return bool(set(chunk["pages"]).intersection(question["expected_pages"]))


def _first_rank(
    ranked_indices: Iterable[int],
    chunks: list[dict[str, Any]],
    question: dict[str, Any],
    predicate,
) -> int | None:
    for rank, index in enumerate(ranked_indices, start=1):
        if predicate(chunks[int(index)], question):
            return rank
    return None


def _model_inventory(base_url: str) -> dict[str, dict[str, Any]]:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()
    result = {}
    for model in response.json().get("models", []):
        result[model["name"].split(":")[0]] = model
    return result


def _summarize(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "model",
        "chunk_words",
        "overlap_words",
        "k",
        "split",
        "embedding_dimension",
        "model_context_tokens",
        "index_chunks",
        "document_embedding_seconds",
        "query_embedding_seconds",
        "retrieval_ms_per_question",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in details:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    summaries = []
    for values, rows in grouped.items():
        summary = dict(zip(keys, values, strict=False))
        hits = sum(int(row["evidence_hit"]) for row in rows)
        proportion = hits / len(rows)
        z = 1.96
        denominator = 1 + z * z / len(rows)
        center = (proportion + z * z / (2 * len(rows))) / denominator
        margin = (
            z
            * math.sqrt(
                proportion * (1 - proportion) / len(rows)
                + z * z / (4 * len(rows) ** 2)
            )
            / denominator
        )
        summary.update(
            {
                "questions": len(rows),
                "recall_at_k": proportion,
                "recall_ci_low": max(0.0, center - margin),
                "recall_ci_high": min(1.0, center + margin),
                "page_recall_at_k": sum(int(row["page_hit"]) for row in rows) / len(rows),
                "mrr_at_k": sum(float(row["reciprocal_rank"]) for row in rows) / len(rows),
                "context_truncation_risk": bool(
                    int(summary["chunk_words"])
                    > 0.75 * int(summary["model_context_tokens"])
                ),
            }
        )
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            row["split"],
            row["model"],
            row["chunk_words"],
            row["overlap_words"],
            row["k"],
        ),
    )


def _select_configuration(summary: list[dict[str, Any]]) -> dict[str, Any]:
    development = [
        row for row in summary if row["split"] == "development" and row["k"] == 5
    ]
    best = sorted(
        development,
        key=lambda row: (
            -row["recall_at_k"],
            -row["mrr_at_k"],
            row["retrieval_ms_per_question"],
            row["chunk_words"],
            row["overlap_words"],
            row["model"],
        ),
    )[0]
    test = [
        row
        for row in summary
        if row["split"] == "test"
        and row["model"] == best["model"]
        and row["chunk_words"] == best["chunk_words"]
        and row["overlap_words"] == best["overlap_words"]
    ]
    return {
        "selection_rule": (
            "Highest development evidence Recall@5; ties broken by development "
            "MRR@5, retrieval latency, smaller chunks, lower overlap, then model name."
        ),
        "selected_on": "development",
        "selected_configuration": {
            "model": best["model"],
            "chunk_words": best["chunk_words"],
            "overlap_words": best["overlap_words"],
            "development_recall_at_5": best["recall_at_k"],
            "development_mrr_at_5": best["mrr_at_k"],
        },
        "held_out_test": [
            {
                "k": row["k"],
                "questions": row["questions"],
                "recall_at_k": row["recall_at_k"],
                "recall_ci_low": row["recall_ci_low"],
                "recall_ci_high": row["recall_ci_high"],
                "page_recall_at_k": row["page_recall_at_k"],
                "mrr_at_k": row["mrr_at_k"],
            }
            for row in sorted(test, key=lambda row: row["k"])
        ],
    }


def run(args: argparse.Namespace) -> None:
    validation = validate(args.corpus, args.questions)
    full_corpus = read_jsonl(args.corpus)
    corpus = [
        row
        for row in full_corpus
        if row["pages"]
        and row["pages"][-1] >= args.first_page
        and row["pages"][0] <= args.last_page
    ]
    if not corpus:
        raise RuntimeError("The requested corpus page range contains no records")
    questions = read_jsonl(args.questions)
    corpus_hash = stable_hash(corpus)
    question_hash = stable_hash(questions)
    all_models = {model.name: model for model in EMBEDDING_MODELS}
    requested = args.models or list(all_models)
    unknown = sorted(set(requested) - set(all_models))
    if unknown:
        raise ValueError(f"Unknown embedding model(s): {', '.join(unknown)}")
    models = [all_models[name] for name in requested]

    inventory = _model_inventory(args.base_url)
    missing = [model.name for model in models if model.name not in inventory]
    if missing:
        raise RuntimeError(
            "Pull missing Ollama models first: "
            + ", ".join(f"ollama pull {name}" for name in missing)
        )
    embedder = OllamaEmbedder(
        base_url=args.base_url, batch_size=args.batch_size, timeout=args.timeout
    )
    details: list[dict[str, Any]] = []
    inventory_rows = []

    for model_index, model in enumerate(models, start=1):
        model_meta = inventory[model.name]
        digest = model_meta.get("digest", "unknown")
        query_fingerprint = stable_hash(
            {
                "questions": question_hash,
                "model": model.__dict__,
                "digest": digest,
                "kind": "query",
            }
        )
        query_cache = args.cache / f"queries-{model_slug(model.name)}.npz"
        query_embeddings, query_seconds, query_cached = _cached_embeddings(
            query_cache,
            [row["question"] for row in questions],
            model,
            "query",
            embedder,
            query_fingerprint,
        )
        print(
            f"[{model_index}/{len(models)}] {model.name}: "
            f"{query_embeddings.shape[1]} dimensions; queries "
            f"{'loaded from cache' if query_cached else f'embedded in {query_seconds:.1f}s'}"
        )
        inventory_rows.append(
            {
                "model": model.name,
                "digest": digest,
                "size_bytes": model_meta.get("size", ""),
                "embedding_dimension": query_embeddings.shape[1],
                "context_tokens": model.context_tokens,
                "document_prefix": model.document_prefix,
                "query_prefix": model.query_prefix,
            }
        )

        for chunk_words in CHUNK_SIZES:
            for overlap_words in OVERLAPS:
                chunks = rechunk(corpus, chunk_words, overlap_words)
                fingerprint = stable_hash(
                    {
                        "corpus": corpus_hash,
                        "model": model.__dict__,
                        "digest": digest,
                        "chunk_words": chunk_words,
                        "overlap_words": overlap_words,
                    }
                )
                cache_path = args.cache / (
                    f"documents-{model_slug(model.name)}-w{chunk_words}-o{overlap_words}.npz"
                )
                document_embeddings, document_seconds, document_cached = _cached_embeddings(
                    cache_path,
                    [row["text"] for row in chunks],
                    model,
                    "document",
                    embedder,
                    fingerprint,
                )
                started = time.perf_counter()
                similarities = query_embeddings @ document_embeddings.T
                ranked = np.argsort(-similarities, axis=1)[:, : max(TOP_K_VALUES)]
                retrieval_ms = (
                    (time.perf_counter() - started) * 1000 / len(questions)
                )
                print(
                    f"  w={chunk_words}, overlap={overlap_words}: {len(chunks)} chunks; "
                    f"{'cache' if document_cached else f'{document_seconds:.1f}s'}"
                )

                for query_index, question in enumerate(questions):
                    evidence_rank = _first_rank(
                        ranked[query_index], chunks, question, _evidence_relevant
                    )
                    page_rank = _first_rank(
                        ranked[query_index], chunks, question, _page_relevant
                    )
                    for k in TOP_K_VALUES:
                        details.append(
                            {
                                "model": model.name,
                                "chunk_words": chunk_words,
                                "overlap_words": overlap_words,
                                "k": k,
                                "split": question["split"],
                                "question_id": question["id"],
                                "question": question["question"],
                                "expected_pages": ",".join(
                                    str(page) for page in question["expected_pages"]
                                ),
                                "evidence_hit": evidence_rank is not None
                                and evidence_rank <= k,
                                "page_hit": page_rank is not None and page_rank <= k,
                                "first_evidence_rank": evidence_rank or "",
                                "first_page_rank": page_rank or "",
                                "reciprocal_rank": (
                                    1.0 / evidence_rank
                                    if evidence_rank is not None and evidence_rank <= k
                                    else 0.0
                                ),
                                "embedding_dimension": query_embeddings.shape[1],
                                "model_context_tokens": model.context_tokens,
                                "index_chunks": len(chunks),
                                "document_embedding_seconds": round(document_seconds, 6),
                                "query_embedding_seconds": round(query_seconds, 6),
                                "retrieval_ms_per_question": round(retrieval_ms, 6),
                            }
                        )

    # Add aggregate rows without peeking at them during model selection.
    aggregate_details = details + [
        {**row, "split": "all"} for row in details
    ]
    summary = _summarize(aggregate_details)
    selection = _select_configuration(summary)
    args.results.mkdir(parents=True, exist_ok=True)
    _write_csv(args.results / "retrieval_details.csv", details)
    _write_csv(args.results / "retrieval_summary.csv", summary)
    _write_csv(args.results / "model_inventory.csv", inventory_rows)
    (args.results / "selected_configuration.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "benchmark": validation,
        "corpus_sha256": corpus_hash,
        "questions_sha256": question_hash,
        "chunk_sizes_words": list(CHUNK_SIZES),
        "overlap_words": list(OVERLAPS),
        "top_k": list(TOP_K_VALUES),
        "models": [model.name for model in models],
        "embedding_options": {
            "truncate": True,
            "num_ctx": "Each model's declared context limit",
            "similarity": "cosine over L2-normalized embeddings",
        },
        "corpus_scope": {
            "first_page": args.first_page,
            "last_page": args.last_page,
            "docling_records": len(corpus),
            "words": sum(len(row["text"].split()) for row in corpus),
        },
        "metric": (
            "Recall@k requires a retrieved chunk to overlap the labeled page(s) "
            "and contain every reviewed evidence term."
        ),
    }
    (args.results / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--models", nargs="*", help="Subset of configured model names")
    parser.add_argument("--first-page", type=int, default=298)
    parser.add_argument("--last-page", type=int, default=483)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    for field in ("corpus", "questions", "results", "cache"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    with _exclusive_run_lock(args.cache / ".benchmark.lock"):
        run(args)


if __name__ == "__main__":
    main()

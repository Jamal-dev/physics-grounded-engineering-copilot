"""Build an operational Chroma index from the selected retrieval configuration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from langchain_core.documents import Document

from config import settings
from experiments.common import (
    DEFAULT_CORPUS,
    DEFAULT_RESULTS,
    EMBEDDING_MODELS,
    read_jsonl,
    stable_hash,
)
from rag import _document_ids, normalize_scope, rechunk_documents, vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_RESULTS / "selected_configuration.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RESULTS / "experiment_manifest.json",
    )
    parser.add_argument("--collection-name")
    parser.add_argument("--scope", default=settings.default_scope)
    parser.add_argument(
        "--document-title",
        default="Theory of Porous Media",
        help="EmbeddingGemma title prompt used for this benchmark corpus.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    selection_report = json.loads(
        args.selection.expanduser().resolve().read_text(encoding="utf-8")
    )
    manifest = json.loads(
        args.manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    selection = selection_report["selected_configuration"]
    corpus_scope = manifest["corpus_scope"]
    first_page = int(corpus_scope["first_page"])
    last_page = int(corpus_scope["last_page"])
    collection_name = args.collection_name or (
        f"document_rag_{selection['model']}_w{selection['chunk_words']}"
        f"_o{selection['overlap_words']}_p{first_page}_{last_page}"
    )
    selected_model = next(
        model for model in EMBEDDING_MODELS if model.name == selection["model"]
    )
    config = replace(
        settings,
        embedding_model=str(selection["model"]),
        embedding_context_tokens=int(selected_model.context_tokens or 2048),
        embedding_document_title=args.document_title,
        collection_name=collection_name,
        chunk_words=int(selection["chunk_words"]),
        chunk_overlap_words=int(selection["overlap_words"]),
    )

    full_corpus = read_jsonl(args.corpus.expanduser().resolve())
    records = sorted(
        (
            record
            for record in full_corpus
            if record["pages"]
            and record["pages"][-1] >= first_page
            and record["pages"][0] <= last_page
        ),
        key=lambda row: row["sequence"],
    )
    if len(records) != int(corpus_scope["docling_records"]):
        raise RuntimeError(
            "The deployment corpus does not reproduce the benchmark page scope."
        )
    if stable_hash(records) != manifest["corpus_sha256"]:
        raise RuntimeError(
            "The deployment corpus hash does not match the benchmark manifest."
        )
    scope = normalize_scope(args.scope)
    documents = [
        Document(
            page_content=str(record["text"]),
            metadata={
                "source": Path(str(record["source"])).name,
                "scope": scope,
                "pages": ", ".join(str(page) for page in record["pages"]),
            },
        )
        for record in records
    ]
    chunks = rechunk_documents(
        documents,
        chunk_words=config.chunk_words,
        overlap_words=config.chunk_overlap_words,
    )
    ids = _document_ids(chunks)
    store = vector_store(config)
    for start in range(0, len(chunks), args.batch_size):
        end = min(start + args.batch_size, len(chunks))
        store.add_documents(chunks[start:end], ids=ids[start:end])
        print(f"Indexed {end}/{len(chunks)} chunks", flush=True)

    if store._collection.count() != len(chunks):
        raise RuntimeError(
            "The target collection contains records outside this deployment. "
            "Use a fresh collection name."
        )
    print(
        json.dumps(
            {
                "collection": config.collection_name,
                "embedding_model": config.embedding_model,
                "chunk_words": config.chunk_words,
                "overlap_words": config.chunk_overlap_words,
                "chunks": len(chunks),
                "scope": scope,
                "first_page": first_page,
                "last_page": last_page,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Build the fine-tuning comparison RAG corpus from training examples only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace

from langchain_core.documents import Document

from config import settings
from rag import normalize_scope, rechunk_documents, vector_store

from .config import FineTuneConfig, load_config
from .data import evaluation_examples, load_and_validate


def _document_id(document: Document) -> str:
    value = "\0".join(
        (
            str(document.metadata["scope"]),
            str(document.metadata["source"]),
            str(document.metadata["start_word"]),
            document.page_content,
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()


def build_training_corpus(config: FineTuneConfig) -> dict[str, object]:
    evaluation = config.evaluation
    collection_name = str(evaluation["rag_collection_name"])
    scope = normalize_scope(str(evaluation["rag_scope"]))
    rag_config = replace(settings, collection_name=collection_name)
    rows, report = load_and_validate(config.train_file)
    examples = evaluation_examples(rows)
    documents = [
        Document(
            page_content=(
                f"Engineering question:\n{example.query}\n\n"
                f"Verified response:\n{example.reference}"
            ),
            metadata={
                "source": f"training-example-{example.id}",
                "scope": scope,
                "family": str(example.metadata.get("family", "unspecified")),
                "task": str(example.metadata.get("task", "unspecified")),
            },
        )
        for example in examples
    ]
    chunks = rechunk_documents(
        documents,
        chunk_words=rag_config.chunk_words,
        overlap_words=rag_config.chunk_overlap_words,
    )
    store = vector_store(rag_config)
    ids = [_document_id(chunk) for chunk in chunks]
    for start in range(0, len(chunks), 32):
        store.add_documents(chunks[start : start + 32], ids=ids[start : start + 32])

    count = int(store._collection.count())
    if count != len(chunks):
        raise RuntimeError(
            f"Collection contains {count} chunks; expected exactly {len(chunks)}. "
            "Use a fresh evaluation.rag_collection_name."
        )
    manifest = {
        "collection_name": collection_name,
        "scope": scope,
        "source_split": "train",
        "source_examples": len(examples),
        "source_sha256": report.sha256,
        "indexed_chunks": count,
        "embedding_model": rag_config.embedding_model,
        "chunk_words": rag_config.chunk_words,
        "chunk_overlap_words": rag_config.chunk_overlap_words,
    }
    destination = config.output_dir / "retrieval_corpus_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(build_training_corpus(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()

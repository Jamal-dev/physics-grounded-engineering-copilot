"""Command-line entry point for document-grounded answer comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import settings
from rag import ask, ask_without_rag, indexed_chunk_count, ingest, validate_local_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Document-grounded RAG tool")
    parser.add_argument("--ingest", nargs="*", metavar="FILE", help="Documents to index")
    parser.add_argument("--question", help="Question to answer from indexed documents")
    parser.add_argument(
        "--provider", choices=("Ollama", "OpenAI"), default="Ollama"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--scope", help="Evidence scope; defaults to RAG_DEFAULT_SCOPE")
    parser.add_argument(
        "--min-relevance-score",
        type=float,
        help="Retrieval gate; defaults to RAG_MIN_RELEVANCE_SCORE",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Explicit persistent-data path; overrides ambient Conda configuration",
    )
    parser.add_argument("--collection-name", help="Explicit Chroma collection name")
    args = parser.parse_args()

    config = settings
    if args.data_dir:
        config = replace(config, data_dir=args.data_dir.expanduser().resolve())
    if args.collection_name:
        config = replace(config, collection_name=args.collection_name)

    problems = validate_local_models(config)
    if problems and args.provider == "Ollama":
        raise SystemExit("\n".join(problems))

    if args.ingest:
        count = ingest(args.ingest, config=config, scope=args.scope)
        print(f"Indexed {count} chunks. Total: {indexed_chunk_count(config)}")

    if args.question:
        baseline = ask_without_rag(args.question, provider=args.provider, config=config)
        grounded = ask(
            args.question,
            provider=args.provider,
            k=args.top_k,
            retrieval_hint=baseline,
            scope=args.scope,
            min_relevance_score=args.min_relevance_score,
            config=config,
        )
        print(f"\nBEFORE RAG\n{baseline}\n")
        print(f"AFTER RAG\n{grounded.text}\n")
        if grounded.warnings:
            print("RETRIEVAL AND CITATION GATES:")
            for warning in grounded.warnings:
                print(f"  - {warning}")
        print("EVIDENCE:")
        for source in grounded.sources:
            pages = f", page(s) {source.pages}" if source.pages else ""
            print(
                f"  [{source.number}] {source.source}{pages} "
                f"(relevance {source.score:.3f})"
            )

    if not args.ingest and not args.question:
        print(f"Indexed chunks: {indexed_chunk_count(config)}")


if __name__ == "__main__":
    main()

"""Export Docling-derived Chroma records to a private, page-aware corpus file."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb

from experiments.common import DEFAULT_CORPUS, parse_pages, stable_hash, write_jsonl


def _sequence(metadata: dict[str, Any], fallback: int) -> int:
    try:
        parsed = json.loads(str(metadata.get("dl_meta", "{}")))
    except json.JSONDecodeError:
        return fallback
    references = []
    for item in parsed.get("doc_items", []):
        match = re.search(r"/(\d+)$", str(item.get("self_ref", "")))
        if match:
            references.append(int(match.group(1)))
    return min(references, default=fallback)


def export_corpus(
    chroma_dir: Path,
    collection_name: str,
    output: Path,
    source: str | None = None,
) -> list[dict[str, Any]]:
    collection = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(
        collection_name
    )
    result = collection.get(include=["documents", "metadatas"])
    records = []
    for fallback, (document, metadata) in enumerate(
        zip(result["documents"], result["metadatas"], strict=False)
    ):
        if source and Path(str(metadata.get("source", ""))).name != Path(source).name:
            continue
        pages = parse_pages(metadata.get("pages"))
        if not document or not pages:
            continue
        records.append(
            {
                "sequence": _sequence(metadata, fallback + 10_000_000),
                "source": Path(str(metadata.get("source", "document"))).name,
                "pages": pages,
                "text": document.strip(),
            }
        )

    records.sort(key=lambda row: (row["sequence"], row["pages"][0]))
    for index, record in enumerate(records):
        record["sequence"] = index
    write_jsonl(output, records)
    manifest = {
        "collection": collection_name,
        "source": source,
        "record_count": len(records),
        "corpus_sha256": stable_hash(records),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return records


def main() -> None:
    default_data = Path(os.getenv("RAG_DATA_DIR", ".data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-dir", type=Path, default=default_data / "chroma")
    parser.add_argument("--collection", default="document_rag")
    parser.add_argument("--source", default="theory_of_elasticity.pdf")
    parser.add_argument("--output", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    records = export_corpus(
        args.chroma_dir.expanduser().resolve(),
        args.collection,
        args.output.expanduser().resolve(),
        args.source,
    )
    print(f"Exported {len(records)} private corpus records to {args.output}")


if __name__ == "__main__":
    main()

"""Configuration for the document-grounded RAG tool."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def _project_path(variable: str, default: str) -> Path:
    value = Path(os.getenv(variable, default)).expanduser()
    if not value.is_absolute():
        value = PROJECT_DIR / value
    return value.resolve()


def _environment_flag(variable: str, default: bool = False) -> bool:
    value = os.getenv(variable)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_project_path(variable: str) -> Path | None:
    value = os.getenv(variable)
    return _project_path(variable, value) if value else None


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _project_path("RAG_DATA_DIR", ".data")
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
    )
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    embedding_model: str = os.getenv(
        "OLLAMA_EMBED_MODEL", "nomic-embed-text"
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    collection_name: str = os.getenv(
        "RAG_COLLECTION_NAME", "document_rag"
    )
    default_scope: str = os.getenv("RAG_DEFAULT_SCOPE", "default")
    min_relevance_score: float = float(
        os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.45")
    )
    enable_ocr: bool = _environment_flag("RAG_ENABLE_OCR")
    docling_threads: int = int(os.getenv("DOCLING_THREADS", "8"))
    fine_tune_config: Path | None = _optional_project_path("FINE_TUNE_CONFIG")
    fine_tune_adapter: Path | None = _optional_project_path(
        "FINE_TUNE_ADAPTER_PATH"
    )

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def example_document(self) -> Path | None:
        configured = os.getenv("RAG_EXAMPLE_DOCUMENT")
        if configured:
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = PROJECT_DIR / candidate
        else:
            candidate = PROJECT_DIR / ".local_documents" / "theory_of_elasticity.pdf"
        candidate = candidate.resolve()
        return candidate if candidate.is_file() else None

    def ensure_directories(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

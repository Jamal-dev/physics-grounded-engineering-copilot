"""Configuration loading for fine-tuning, inference, and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


@dataclass(frozen=True)
class FineTuneConfig:
    """Validated, path-aware view of a YAML fine-tuning configuration."""

    source: Path
    base_model: str
    train_file: Path
    validation_file: Path | None
    test_file: Path | None
    output_dir: Path
    model_init: dict[str, Any]
    tokenizer_init: dict[str, Any]
    lora: dict[str, Any]
    sft: dict[str, Any]
    generation: dict[str, Any]
    chat_template: dict[str, Any]
    evaluation: dict[str, Any]

    @property
    def adapter_dir(self) -> Path:
        configured = self.evaluation.get("adapter_path")
        if configured:
            return _resolve(self.source.parent, str(configured))
        return self.output_dir / "final"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": {
                "name_or_path": self.base_model,
                "init_kwargs": deepcopy(self.model_init),
                "tokenizer_kwargs": deepcopy(self.tokenizer_init),
            },
            "data": {
                "train_file": str(self.train_file),
                "validation_file": (
                    str(self.validation_file) if self.validation_file else None
                ),
                "test_file": str(self.test_file) if self.test_file else None,
            },
            "output_dir": str(self.output_dir),
            "lora": deepcopy(self.lora),
            "sft": deepcopy(self.sft),
            "generation": deepcopy(self.generation),
            "chat_template": deepcopy(self.chat_template),
            "evaluation": deepcopy(self.evaluation),
        }


def load_config(path: str | Path) -> FineTuneConfig:
    """Load a YAML config and resolve all local paths relative to that file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fine-tuning config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("The fine-tuning config must contain a YAML mapping.")

    model = _mapping(raw.get("model"), "model")
    data = _mapping(raw.get("data"), "data")
    base_model = str(model.get("name_or_path", "")).strip()
    train_value = data.get("train_file")
    if not base_model:
        raise ValueError("model.name_or_path is required.")
    if not train_value:
        raise ValueError("data.train_file is required.")

    base_dir = source.parent
    validation_value = data.get("validation_file")
    test_value = data.get("test_file")
    output_value = raw.get("output_dir", ".data/fine_tuning/run")
    return FineTuneConfig(
        source=source,
        base_model=base_model,
        train_file=_resolve(base_dir, str(train_value)),
        validation_file=(
            _resolve(base_dir, str(validation_value)) if validation_value else None
        ),
        test_file=_resolve(base_dir, str(test_value)) if test_value else None,
        output_dir=_resolve(base_dir, str(output_value)),
        model_init=_mapping(model.get("init_kwargs"), "model.init_kwargs"),
        tokenizer_init=_mapping(
            model.get("tokenizer_kwargs"), "model.tokenizer_kwargs"
        ),
        lora=_mapping(raw.get("lora"), "lora"),
        sft=_mapping(raw.get("sft"), "sft"),
        generation=_mapping(raw.get("generation"), "generation"),
        chat_template=_mapping(raw.get("chat_template"), "chat_template"),
        evaluation=_mapping(raw.get("evaluation"), "evaluation"),
    )

"""CLI for model-agnostic conversational LoRA supervised fine-tuning."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from .config import FineTuneConfig, load_config
from .data import DatasetReport, load_and_validate, split_prompt_collisions


def _automatic_bool(value: Any, automatic: bool) -> bool:
    if isinstance(value, str) and value.lower() == "auto":
        return automatic
    return bool(value)


def build_sft_config(config: FineTuneConfig) -> SFTConfig:
    values = dict(config.sft)
    values["output_dir"] = str(config.output_dir)
    cuda = torch.cuda.is_available()
    values["use_cpu"] = _automatic_bool(values.get("use_cpu", "auto"), not cuda)
    values["bf16"] = _automatic_bool(
        values.get("bf16", "auto"),
        cuda and torch.cuda.is_bf16_supported(),
    )
    values["fp16"] = _automatic_bool(
        values.get("fp16", "auto"), cuda and not values["bf16"]
    )

    model_init = dict(config.model_init)
    if "dtype" not in model_init:
        model_init["dtype"] = "auto" if cuda else "float32"
    values["model_init_kwargs"] = model_init
    values.setdefault("report_to", "none")
    if config.validation_file:
        values.setdefault("eval_strategy", "epoch")
    else:
        values.setdefault("eval_strategy", "no")
    return SFTConfig(**values)


def build_lora_config(config: FineTuneConfig) -> LoraConfig:
    values = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
        "bias": "none",
    }
    values.update(config.lora)
    return LoraConfig(**values)


def _dataset(
    path: Path, chat_template: dict[str, Any]
) -> tuple[Dataset, DatasetReport, list[dict[str, Any]]]:
    rows, report = load_and_validate(path)
    if chat_template:
        for row in rows:
            per_example = dict(row.get("chat_template_kwargs") or {})
            row["chat_template_kwargs"] = {**chat_template, **per_example}
    return Dataset.from_list(rows, on_mixed_types="use_json"), report, rows


def _manifest(
    config: FineTuneConfig,
    reports: list[DatasetReport],
    *,
    completed: bool,
    collisions: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "completed": completed,
        "config_source": str(config.source),
        "config": config.as_dict(),
        "datasets": [report.as_dict() for report in reports],
        "split_prompt_collisions": collisions or {},
        "libraries": {
            name: version(name)
            for name in ("torch", "transformers", "datasets", "peft", "trl")
        },
    }


def train(config: FineTuneConfig, *, resume: str | None = None) -> Path:
    train_dataset, train_report, train_rows = _dataset(
        config.train_file, config.chat_template
    )
    reports = [train_report]
    split_rows = {"train": train_rows}
    eval_dataset = None
    if config.validation_file:
        eval_dataset, validation_report, validation_rows = _dataset(
            config.validation_file, config.chat_template
        )
        reports.append(validation_report)
        split_rows["validation"] = validation_rows
    collisions = split_prompt_collisions(split_rows)
    if any(collisions.values()):
        raise ValueError(f"Exact user-prompt leakage between splits: {collisions}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(config, reports, completed=False, collisions=collisions),
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer = SFTTrainer(
        model=config.base_model,
        args=build_sft_config(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(config),
    )
    trainer.train(resume_from_checkpoint=resume)
    adapter_dir = config.output_dir / "final"
    trainer.save_model(str(adapter_dir))
    manifest_path.write_text(
        json.dumps(
            _manifest(config, reports, completed=True, collisions=collisions),
            indent=2,
        ),
        encoding="utf-8",
    )
    return adapter_dir


def dry_run(config: FineTuneConfig) -> dict[str, Any]:
    reports: list[DatasetReport] = []
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for name, path in (
        ("train", config.train_file),
        ("validation", config.validation_file),
        ("test", config.test_file),
    ):
        if path:
            rows, report = load_and_validate(path)
            reports.append(report)
            split_rows[name] = rows
    collisions = split_prompt_collisions(split_rows)
    if any(collisions.values()):
        raise ValueError(f"Exact user-prompt leakage between splits: {collisions}")
    # Construct both configs so misspelled or version-incompatible options fail
    # before a model download or expensive run begins.
    sft = build_sft_config(config)
    lora = build_lora_config(config)
    result = _manifest(config, reports, completed=False, collisions=collisions)
    result["resolved"] = {
        "use_cpu": sft.use_cpu,
        "bf16": sft.bf16,
        "fp16": sft.fp16,
        "target_modules": lora.target_modules,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        help="Resume from the latest or named Trainer checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and options without loading the base model",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dry_run(config), indent=2))
        return
    resume = args.resume
    if resume is True:
        resume = True
    adapter_dir = train(config, resume=resume)
    print(f"Saved adapter: {adapter_dir}")


if __name__ == "__main__":
    main()

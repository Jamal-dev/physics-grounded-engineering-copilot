"""Shared base/LoRA-adapter inference for Hugging Face chat models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import FineTuneConfig


class AdapterChatModel:
    """Load one base model and toggle a PEFT adapter for fair comparisons."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str | Path,
        *,
        model_init: Mapping[str, Any] | None = None,
        tokenizer_init: Mapping[str, Any] | None = None,
        generation: Mapping[str, Any] | None = None,
        chat_template: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_model_name = base_model
        self.adapter_path = str(adapter_path)
        tokenizer_options = dict(tokenizer_init or {})
        model_options = {"dtype": "auto", "device_map": "auto"}
        model_options.update(model_init or {})
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_options)
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()
        self.generation = {"max_new_tokens": 512, "do_sample": False}
        self.generation.update(generation or {})
        self.chat_template = dict(chat_template or {})

    @classmethod
    def from_config(
        cls,
        config: FineTuneConfig,
        *,
        adapter_path: str | Path | None = None,
    ) -> AdapterChatModel:
        return cls(
            config.base_model,
            adapter_path or config.adapter_dir,
            model_init=config.model_init,
            tokenizer_init=config.tokenizer_init,
            generation=config.generation,
            chat_template=config.chat_template,
        )

    def _model_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tuned: bool,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        template_options = dict(self.chat_template)
        if tools:
            template_options["tools"] = list(tools)
        encoded = self.tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_options,
        )
        encoded = {
            name: tensor.to(self._model_device())
            for name, tensor in encoded.items()
        }
        generation_options = dict(self.generation)
        generation_options.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        adapter_context = nullcontext() if tuned else self.model.disable_adapter()
        with torch.inference_mode(), adapter_context:
            output = self.model.generate(**encoded, **generation_options)
        prompt_tokens = encoded["input_ids"].shape[-1]
        return self.tokenizer.decode(
            output[0, prompt_tokens:], skip_special_tokens=True
        ).strip()

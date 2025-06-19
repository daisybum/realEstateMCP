"""Base model loader for LGAI-EXAONE/EXAONE-Deep-2.4B.

This module centralises all logic to load, configure and optionally fine-tune
the base model with memory-efficient settings that fit within a 48 GB VRAM
budget.

Usage::

    from src.model.base_model import BaseModelLoader
    loader = BaseModelLoader()
    model, tokenizer = loader.load_model()

The loader automatically:
1. Downloads & caches the model/tokenizer from HuggingFace.
2. Loads in FP16 (torch.float16) by default.
3. Utilises `device_map="auto"` so the model is placed on the available GPU(s)
   or falls back to CPU.
4. Optionally enables gradient checkpointing.
5. If ``peft`` is installed and ``use_lora`` is True, prepares the model with
   LoRA adapters for parameter-efficient fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "`torch` and `transformers` are required for model loading. "
        "Please install them first (e.g. `pip install torch transformers`)."
    ) from e


@dataclass
class LoaderConfig:
    """Configuration parameters for :class:`BaseModelLoader`."""

    model_name: str = "LGAI-EXAONE/EXAONE-Deep-2.4B"
    torch_dtype: str = "float16"  # one of {"float16", "bfloat16", "float32"}
    device_map: str | Dict[str, Any] = "auto"  # or explicit dict
    enable_gradient_checkpointing: bool = True
    use_lora: bool = False  # prepare LoRA adapters via `peft`
    lora_r: int = 16  # rank
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def torch_dtype_obj(self) -> torch.dtype:  # type: ignore[name-defined]
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping[self.torch_dtype]


class BaseModelLoader:
    """Loads and configures the EXAONE-Deep-2.4B model."""

    def __init__(self, config: LoaderConfig | None = None) -> None:
        self.config = config or LoaderConfig()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def load_model(self) -> Tuple["torch.nn.Module", "AutoTokenizer"]:  # type: ignore[name-defined]
        """Download and load model + tokenizer with the provided config."""
        logger.info("Loading tokenizer: %s", self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        logger.info(
            "Loading model: %s (dtype=%s)…", self.config.model_name, self.config.torch_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype_obj(),
            device_map=self.config.device_map,
            trust_remote_code=True,
        )

        if self.config.enable_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing…")
            model.gradient_checkpointing_enable()

        if self.config.use_lora:
            model = self._prepare_lora(model)

        return model, tokenizer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _prepare_lora(self, model: "torch.nn.Module") -> "torch.nn.Module":  # type: ignore[name-defined]
        """Inject LoRA adapters using *peft* (if available)."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "`peft` is not installed. Install with `pip install peft` to use LoRA."
            ) from exc

        logger.info("Adding LoRA adapters (r=%d, α=%d)…", self.config.lora_r, self.config.lora_alpha)
        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()  # type: ignore[attr-defined]
        return model


__all__ = ["BaseModelLoader", "LoaderConfig"]

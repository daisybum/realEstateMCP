"""Tokenizer wrapper around HuggingFace *transformers* AutoTokenizer.

Handles model loading, text cleaning (optional), and provides helper methods
for tokenisation & encoding.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import List, Sequence, Dict, Iterable

from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore

from .cleaning import clean_text, TextCleaner

LOGGER = logging.getLogger(__name__)


class HFTokenizer:
    """Thin wrapper around *transformers* ``AutoTokenizer``.

    Parameters
    ----------
    model_name : str
        Pre-trained model name or path.
    max_length : int, default 512
        Truncation length for ``encode``.
    add_special_tokens : bool, default True
        Whether to add BOS/EOS etc.
    do_clean : bool, default True
        If ``True``, apply `clean_text` before tokenising.
    """

    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-Deep-2.4B",
        *,
        max_length: int = 512,
        add_special_tokens: bool = True,
        do_clean: bool = True,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.do_clean = do_clean
        self.cleaner = TextCleaner()
        LOGGER.info("Loaded tokenizer '%s'", model_name)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """Return list of tokens."""
        if self.do_clean:
            text = clean_text(text)
        return self.tokenizer.tokenize(text)

    def encode(self, text: str) -> List[int]:
        """Return token IDs with truncation/padding as configured."""
        if self.do_clean:
            text = clean_text(text)
        return self.tokenizer.encode(
            text,
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            max_length=self.max_length,
        )

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text (skipping special tokens)."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_count(self, text: str) -> int:
        """Return the number of tokens for *text* after cleaning."""
        return len(self.tokenize(text))

    def analyze_corpus(self, texts: Iterable[str]) -> Dict[str, float]:
        """Compute basic statistics for a corpus.

        Returns a dict with keys:
        - avg_tokens : float
        - max_tokens : int
        - pct_unk    : percentage of <unk> tokens across corpus
        """
        sizes: List[int] = []
        unk_id = self.tokenizer.unk_token_id
        unk_total = 0
        total_tokens = 0
        for t in texts:
            ids = self.encode(t)
            sizes.append(len(ids))
            total_tokens += len(ids)
            unk_total += sum(1 for i in ids if i == unk_id)
        if not sizes:
            return {"avg_tokens": 0.0, "max_tokens": 0, "pct_unk": 0.0}
        return {
            "avg_tokens": sum(sizes) / len(sizes),
            "max_tokens": max(sizes),
            "pct_unk": (unk_total / total_tokens) * 100 if total_tokens else 0.0,
        }

    def batch_encode(self, texts: Sequence[str]):
        """Encode batch and return *dict* suitable for model input."""
        if self.do_clean:
            texts = [clean_text(t) for t in texts]
        return self.tokenizer(
            list(texts),
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            # Return plain lists to avoid requiring torch
            return_tensors=None,
        )


__all__ = ["HFTokenizer"]

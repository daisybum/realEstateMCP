"""DatasetBuilder – convert pre-processed corpus into HuggingFace DatasetDict.

Implements **Task 1.4**: format conversion + train/validation/test splitting.

The builder consumes records from ::

    WeolbuPostsLoader -> Preprocessor -> HFTokenizer

and outputs a ``datasets.DatasetDict`` with ``train``, ``validation`` and
``test`` splits containing *input_ids* and *attention_mask* suitable for model
training.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Sequence

import datasets  # type: ignore
from datasets import Dataset, DatasetDict

from .loader import WeolbuPostsLoader
from .preprocess import Preprocessor
from .tokenizer import HFTokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetBuilder:
    jsonl_path: Path | str
    tokenizer: HFTokenizer
    preprocessor: Preprocessor
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_texts(self) -> Generator[Dict[str, str], None, None]:
        loader = WeolbuPostsLoader(self.jsonl_path)
        for rec in self.preprocessor.process_records(loader.extract_body_and_parsed()):
            # Concatenate processed body + parsed_content
            text_parts: List[str] = []
            if rec["body"]["processed"]:
                text_parts.append(rec["body"]["processed"])
            if rec["parsed_content"]["processed"]:
                text_parts.append(rec["parsed_content"]["processed"])
            if not text_parts:
                continue  # skip empty
            yield {"text": "\n\n".join(text_parts)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, save_dir: Path | str | None = None) -> DatasetDict:
        """Return a tokenised DatasetDict; optionally save to *save_dir*."""
        LOGGER.info("Building dataset from %s", self.jsonl_path)
        raw_ds: Dataset = Dataset.from_generator(self._iter_texts)
        LOGGER.info("Loaded %d raw examples", raw_ds.num_rows)

        # Split
        split_ds: DatasetDict = raw_ds.train_test_split(
            test_size=self.test_ratio, seed=self.seed
        )
        # Create validation from train subset
        val_portion = self.val_ratio / (1 - self.test_ratio)
        split_ds["train"], split_ds["validation"] = split_ds["train"].train_test_split(
            test_size=val_portion, seed=self.seed
        ).values()

        # Tokenise
        def _tokenise(batch: Dict[str, Sequence[str]]):
            enc = self.tokenizer.batch_encode(batch["text"])
            return enc

        tokenised = split_ds.map(
            _tokenise,
            batched=True,
            remove_columns=["text"],
            desc="Tokenising corpus",
        )

        if save_dir is not None:
            save_path = Path(save_dir)
            LOGGER.info("Saving dataset to %s", save_path)
            tokenised.save_to_disk(str(save_path))

        return tokenised


__all__ = ["DatasetBuilder"]

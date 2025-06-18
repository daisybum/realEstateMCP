"""Data Quality Validation utilities (Task **1.5**).

This module analyses a *tokenised* HuggingFace ``datasets.Dataset`` (or
``DatasetDict``) and produces a comprehensive JSON report covering:

* dataset sizes (per split)
* token count statistics (mean/median/std/min/max)
* percentage of ``<unk>`` tokens
* vocabulary coverage (unique token IDs vs. model vocab)
* duplicate detection (identical *input_ids*)

The resulting report can be version-controlled with *DVC* (Data Version
Control).  Example CLI usage is provided at the bottom of this file.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List, Sequence

from datasets import Dataset, DatasetDict

from .tokenizer import HFTokenizer

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _flatten_input_ids(dataset: Dataset) -> List[List[int]]:
    """Return list-of-list of *input_ids* from a Dataset."""
    return dataset["input_ids"]  # type: ignore[index]


def _token_length_stats(samples: Sequence[Sequence[int]]):
    lengths = [len(s) for s in samples]
    return {
        "count": len(lengths),
        "mean": mean(lengths) if lengths else 0,
        "median": median(lengths) if lengths else 0,
        "std": stdev(lengths) if len(lengths) > 1 else 0,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }


def _unk_statistics(samples: Iterable[Sequence[int]], unk_id: int):
    total = 0
    unk = 0
    for s in samples:
        total += len(s)
        unk += sum(1 for i in s if i == unk_id)
    pct = (unk / total) * 100 if total else 0
    return {"unk_tokens": unk, "total_tokens": total, "pct_unk": pct}


def _vocab_coverage(samples: Iterable[Sequence[int]], model_vocab_size: int):
    vocab_ids: set[int] = set()
    for s in samples:
        vocab_ids.update(s)
    return {
        "unique_tokens": len(vocab_ids),
        "model_vocab_size": model_vocab_size,
        "coverage_pct": (len(vocab_ids) / model_vocab_size) * 100,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataQualityValidator:
    """Compute data-quality metrics for tokenised datasets."""

    def __init__(self, dataset: Dataset | DatasetDict, tokenizer: HFTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.report: Dict[str, Any] = {}

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        LOGGER.info("Starting data quality validation")
        if isinstance(self.dataset, DatasetDict):
            for split, ds in self.dataset.items():
                LOGGER.info("Analysing split: %s (%d rows)", split, ds.num_rows)
                self.report[split] = self._analyse_split(ds)
        else:
            self.report["dataset"] = self._analyse_split(self.dataset)
        return self.report

    def save(self, path: str | Path):
        path = Path(path)
        path.write_text(json.dumps(self.report, indent=2, ensure_ascii=False))
        LOGGER.info("Saved data-quality report → %s", path)

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    def _analyse_split(self, ds: Dataset):
        samples = _flatten_input_ids(ds)
        stats = {
            **_token_length_stats(samples),
            **_unk_statistics(samples, self.tokenizer.tokenizer.unk_token_id),
            **_vocab_coverage(samples, self.tokenizer.tokenizer.vocab_size),
            "duplicates": self._duplicate_count(samples),
        }
        return stats

    @staticmethod
    def _duplicate_count(samples: Sequence[Sequence[int]]):
        seen = Counter(tuple(s) for s in samples)
        dup_total = sum(c for c in seen.values() if c > 1)
        dup_unique = sum(1 for c in seen.values() if c > 1)
        return {"duplicate_rows": dup_total, "duplicate_variants": dup_unique}


# ---------------------------------------------------------------------------
# Optional CLI entry-point (python -m src.data.quality <dataset_dir> <outfile.json>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Data quality validation")
    parser.add_argument("dataset", help="Path to HF dataset (load_from_disk)")
    parser.add_argument("outfile", help="Output JSON report path")
    args = parser.parse_args()

    ds = DatasetDict.load_from_disk(args.dataset)
    tok = HFTokenizer()
    validator = DataQualityValidator(ds, tok)
    validator.run()
    validator.save(args.outfile)

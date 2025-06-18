"""Data loader module for `weolbu_posts.jsonl`.

This module provides helpers to stream newline-delimited JSON (JSONL) records
and extract the `body` and `parsed_content` fields which will later be cleaned
and tokenised for model training.

Example
-------
>>> from pathlib import Path
>>> from src.data import WeolbuPostsLoader
>>> loader = WeolbuPostsLoader(Path("dataset/weolbu_posts.jsonl"))
>>> next(loader.extract_body_and_parsed())
{'body': '...', 'parsed_content': '...'}

Notes
-----
* Uses only Python std-lib for streaming; `to_dataframe()` lazily imports
  *pandas* for convenience.
* Lines that fail JSON parsing are skipped with a warning.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional

LOGGER = logging.getLogger(__name__)


class WeolbuPostsLoader:
    """Iterate over JSONL records and yield selected fields.

    Parameters
    ----------
    jsonl_path : str | Path
        Path to the JSONL file.
    encoding : str, default "utf-8"
        File encoding.
    """

    def __init__(self, jsonl_path: str | Path, *, encoding: str = "utf-8") -> None:
        self.jsonl_path = Path(jsonl_path)
        self.encoding = encoding
        if not self.jsonl_path.is_file():
            raise FileNotFoundError(f"File not found: {self.jsonl_path}")
        LOGGER.debug("Initialised WeolbuPostsLoader for %s", self.jsonl_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _iter_lines(self) -> Generator[str, None, None]:
        """Yield non-empty, stripped lines from file."""
        with self.jsonl_path.open("r", encoding=self.encoding) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    yield stripped

    def _iter_json(self) -> Generator[Dict, None, None]:
        """Yield decoded JSON objects, skipping malformed lines."""
        for raw in self._iter_lines():
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed JSON line: %s", exc)
                continue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def iter_records(self) -> Generator[Dict, None, None]:
        """Return an iterator over every JSON object."""
        return self._iter_json()

    def extract_body_and_parsed(self) -> Generator[Dict[str, str], None, None]:
        """Yield dicts with only `body` and `parsed_content`."""
        for obj in self.iter_records():
            yield {
                "body": obj.get("body", ""),
                "parsed_content": obj.get("parsed_content", ""),
            }

    # ------------------------------------------------------------------
    # Convenience utils
    # ------------------------------------------------------------------
    def to_dataframe(self, limit: Optional[int] = None):  # noqa: D401
        """Return a pandas DataFrame of the extracted fields."""
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pandas is required for `to_dataframe()`. Install via pip."
            ) from exc

        rows: List[Dict[str, str]] = []
        for idx, rec in enumerate(self.extract_body_and_parsed()):
            rows.append(rec)
            if limit is not None and idx + 1 >= limit:
                break
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __iter__(self):
        return self.extract_body_and_parsed()

    def __len__(self):
        return sum(1 for _ in self._iter_lines())


__all__ = ["WeolbuPostsLoader"]

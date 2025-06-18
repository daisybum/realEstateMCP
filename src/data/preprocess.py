"""High-level preprocessing pipeline for body / parsed_content text.

This module orchestrates the cleaning logic in :pymod:`src.data.cleaning` and
adds optional HTML parsing via *BeautifulSoup* as well as duplicate detection
using SHA-256 hashes.  It is intended to satisfy **Task 1.2** in the Taskmaster
road-map.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from .cleaning import TextCleaner, clean_text

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover – graceful degradation
    BeautifulSoup = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _hash_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _strip_html_bs4(html_text: str) -> str:
    """Return visible text content extracted with *BeautifulSoup* (if available)."""
    if not html_text:
        return ""
    if BeautifulSoup is None:
        # Fallback – rely on regex-based stripping already inside clean_text
        return html_text
    soup = BeautifulSoup(html_text, "html.parser")
    # Remove script/style tags completely
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ")


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    """Configurable preprocessing pipeline.

    Parameters
    ----------
    remove_html : bool, default True
        Use BeautifulSoup to remove HTML elements before `clean_text`.
    deduplicate : bool, default True
        Skip records whose *clean* text is identical to one already seen.
    
    extra_clean_rules : Sequence[callable] | None
        Additional callables applied *after* built-in cleaning.
    """

    remove_html: bool = True
    deduplicate: bool = True
    extra_clean_rules: Optional[Sequence[callable]] = None
    _cleaner: TextCleaner = field(init=False, repr=False)

    def __post_init__(self):
        rules = list(self.extra_clean_rules or [])
        self._cleaner = TextCleaner(rules)
        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """Clean **one** text string according to configuration."""
        if self.remove_html:
            text = _strip_html_bs4(text)
        cleaned = self._cleaner(text)
        return cleaned

    def process_record(self, record: Dict[str, str]) -> Optional[Dict[str, Dict[str, str]]]:
        """Process a single record and return mapping with original / processed texts.

        The returned dict has shape::

            {
                "body": {"original": <str>, "processed": <str>},
                "parsed_content": {"original": <str>, "processed": <str>}
            }

        If ``deduplicate`` is enabled and the processed *body* text is a duplicate
        of a previously seen record, ``None`` is returned.
        """
        body_orig = record.get("body", "")
        parsed_orig = record.get("parsed_content", "")

        body_proc = self.preprocess(body_orig)
        parsed_proc = self.preprocess(parsed_orig)

        # Use body_proc as canonical text for duplication detection
        if self.deduplicate:
            h = _hash_sha256(body_proc)
            if h in self._seen_hashes:
                return None  # duplicate
            self._seen_hashes.add(h)

        return {
            "body": {"original": body_orig, "processed": body_proc},
            "parsed_content": {"original": parsed_orig, "processed": parsed_proc},
        }

    def process_records(
        self, records: Iterable[Dict[str, str]]
    ) -> Generator[Dict[str, Dict[str, str]], None, None]:
        """Yield processed records, applying deduplication if enabled."""
        for rec in records:
            out = self.process_record(rec)
            if out is not None:
                yield out


__all__ = ["Preprocessor"]

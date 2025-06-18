"""Text cleaning and normalisation utilities.

This module provides simple functions/classes for cleaning raw text scraped
from HTML pages before tokenisation. The focus is on Korean + mixed-language
content but the rules are generic.

The cleaning pipeline currently includes:
1. HTML tag stripping
2. HTML entity un-escaping
3. Unicode normalisation (NFKC)
4. Whitespace & control-character normalisation

Extend this file with additional rules (e.g., stop-word removal, emoji
handling) as you refine the dataset.
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return *clean* version of ``text`` suitable for tokenisation."""
    if not text:
        return ""

    # 1. Remove HTML tags quickly via regex (sufficient for simple tags).
    text = _HTML_TAG_RE.sub(" ", text)

    # 2. Unescape HTML entities (&nbsp;, &amp; etc.).
    text = html.unescape(text)

    # 3. Unicode normalisation to NFKC (standardises compatibility chars).
    text = unicodedata.normalize("NFKC", text)

    # 4. Collapse runs of whitespace/control chars to a single space & trim.
    text = _WS_RE.sub(" ", text).strip()

    return text


class TextCleaner:
    """Composable cleaner implementing the above rules with optional extras."""

    def __init__(self, extra_rules: Iterable[callable] | None = None):
        self.extra_rules: List[callable] = list(extra_rules) if extra_rules else []

    def __call__(self, text: str) -> str:
        cleaned = clean_text(text)
        for rule in self.extra_rules:
            try:
                cleaned = rule(cleaned)
            except Exception as exc:  # pragma: no cover – dev helper
                LOGGER.warning("Cleaning rule %s failed: %s", rule, exc)
        return cleaned

    def map(self, texts: Iterable[str]) -> List[str]:
        """Clean an *iterable* of ``texts`` and return a list."""
        return [self(t) for t in texts]


__all__ = ["clean_text", "TextCleaner"]

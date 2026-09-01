"""MCQ outcome extractor: ordered regex patterns, last match wins.

Ported from forking-fast answers.py (parse_mmlu_answer).
"""
from __future__ import annotations

import re

CATEGORIES = ["A", "B", "C", "D", "Other"]

_PATTERNS = [
    r"answer is:?\s*\(?([A-D])\)?",
    r"the correct answer is:?\s*\(?([A-D])\)?",
    r"correct answer is:?\s*\(?([A-D])\)?",
    r"answer:?\s*\(?([A-D])\)?",
    r"option\s*\(?([A-D])\)?",
    r"\*\*\(?([A-D])\)?[\).]",
    r"^\s*\(?([A-D])\)?\s*$",
    r"\(([A-D])\)\s*$",
    r"\b([A-D])\b\s*$",
]
_RES = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _PATTERNS]

#: emitted text ends right before this stop marker; the extractor should see the tail
_STOP_MARKER = "The answer is ("


class MCQExtractor:
    categories = CATEGORIES

    def extract(self, text: str) -> str:
        best = None
        for rx in _RES:
            matches = rx.findall(text)
            if matches:
                best = matches[-1].upper()
                break
        if best in ("A", "B", "C", "D"):
            return best
        # truncated-by-stop case: look for a final bare option letter
        tail = re.findall(r"\(?([A-D])\)?\s*(?:[.,]?)\s*$", text.strip())
        if tail and tail[-1].upper() in ("A", "B", "C", "D"):
            return tail[-1].upper()
        return "Other"

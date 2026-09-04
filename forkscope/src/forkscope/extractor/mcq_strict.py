"""Strict MCQ outcome extraction, and why the loose one is not usable here.

The original MCQExtractor scans the whole continuation with an ordered pattern
list and takes the last match of the first pattern that hits. On a Qwen3
thinking trace that reads the reasoning as if it were the answer. A recorded
college_physics_4 continuation ending

    "...the answer would be option B) The Pauli exclusion principle. Wait, but
     why not option A) Heisenberg...? Option C is the Bohr model, which is"

is labelled C, because `option\\s*([A-D])` matched last on an option the model
was in the middle of rejecting. In that case 71.6% of continuations hit the
1500-token cap and 66% of those parsed as Other, so the measured outcome
distribution was mostly reporting where the cap fell.

Two honest outcome definitions, both computed from the same text:

    terminal    only the region after </think> counts. A continuation that
                never closed its trace has no answer yet: "Unfinished". This is
                the primary definition; it makes truncation visible as its own
                outcome instead of laundering it into Other or into a letter.
    commitment  the last explicit "answer is (X)" anywhere in the text, closed
                trace or not: the model's latest stated answer. Kept as a
                robustness check, because on virology_5 every capped
                continuation carries one and dropping them would discard 43.6%
                of the sample.

Neither reads bare option mentions out of the reasoning.
"""
from __future__ import annotations

import re

CATEGORIES = ["A", "B", "C", "D", "Other", "Unfinished"]

_THINK_END = "</think>"
#: an explicit commitment, the only pattern trusted inside a reasoning trace
_COMMIT = re.compile(r"answer\s+is:?\s*\**\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE)
#: patterns allowed only inside a closed answer region, which is short and terse
_FINAL = [
    _COMMIT,
    re.compile(r"^\s*\**\s*\(?([A-D])\)?\s*\**\s*[.)]?\s*$", re.MULTILINE),
    re.compile(r"\(([A-D])\)\s*\.?\s*$"),
    re.compile(r"\b([A-D])\b\s*\.?\s*$"),
]


class StrictMCQExtractor:
    """mode: "terminal" (default) or "commitment". See the module docstring."""

    categories = CATEGORIES

    def __init__(self, mode: str = "terminal"):
        if mode not in ("terminal", "commitment"):
            raise ValueError(f"mode must be terminal or commitment, got {mode!r}")
        self.mode = mode

    def extract(self, text: str) -> str:
        closed = _THINK_END in text
        if self.mode == "commitment":
            m = _COMMIT.findall(text)
            if m:
                return m[-1].upper()
            return "Other" if closed else "Unfinished"

        if not closed:
            return "Unfinished"
        tail = text.split(_THINK_END)[-1]
        for rx in _FINAL:
            m = rx.findall(tail)
            if m:
                return m[-1].upper()
        return "Other"


def label_both(text: str) -> tuple[str, str]:
    """(terminal, commitment) for one continuation."""
    return (StrictMCQExtractor("terminal").extract(text),
            StrictMCQExtractor("commitment").extract(text))

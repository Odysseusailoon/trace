"""Outcome extraction protocol."""
from __future__ import annotations

from typing import Protocol


class Extractor(Protocol):
    categories: list[str]

    def extract(self, text: str) -> str:
        ...

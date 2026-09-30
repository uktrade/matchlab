"""Deduplication methodologies."""

from matchlab.models.dedupers.base import PreparedDeduper
from matchlab.models.dedupers.naive import NaiveDeduper

__all__ = (
    "PreparedDeduper",
    "NaiveDeduper",
)

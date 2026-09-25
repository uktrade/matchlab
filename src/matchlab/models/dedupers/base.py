"""Base class for deduplication methodologies."""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Deduper(BaseModel, ABC):
    """A methodology that finds candidate duplicate pairs within one record step.

    A `Model` step calls `prepare()` with the complete input before `dedupe()`.
    The action scores pairs involving supplied rows against that prepared input.
    Collection supplies the full input as the affected rows. `dedupe()` returns
    `left_id`, `right_id`, and `score`. `normalise_model_scores` casts the result
    to `SCHEMA_MODEL_EDGES`.

    Every field is a setting unless marked `matchlab.resources.FromResources`. A
    fingerprint ignores a resource, so a marked field must not change what this scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump it to retire every artifact the previous body wrote.
    # Left unset, the step re-runs on every collect, and so does everything below it.
    # See `matchlab.core.versioning`.
    version: ClassVar[int | None] = None

    id: Literal["id"] = Field(
        default="id", description="The unique ID field in the data to dedupe"
    )

    @abstractmethod
    def prepare(self, data: pl.DataFrame) -> None:
        """Prepare the complete input for later affected-input calls."""
        ...

    @abstractmethod
    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Score pairs involving `data` against the prepared input."""
        ...

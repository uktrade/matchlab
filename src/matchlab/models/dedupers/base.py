"""Base class for deduplication methodologies."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Deduper(BaseModel, ABC):
    """A methodology that finds candidate duplicate pairs within one record step.

    A `Model` step retains the `PreparedDeduper` returned by `prepare()`.
    Preparation does not change this specification.
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

    def prepare(self, data: pl.DataFrame) -> "PreparedDeduper":
        """Return runtime preparation from the full baseline without changing self.

        Override this to return derived state; the default needs no setup.
        """
        return PreparedDeduper(methodology=self)

    @abstractmethod
    def dedupe(
        self,
        prepared_state: Any,  # noqa: ANN401 - opaque state owned by the author
        data: pl.DataFrame,
        *,
        baseline: pl.DataFrame,
    ) -> pl.DataFrame:
        """Score pairs involving `data` against the prepared input."""
        ...


class PreparedDeduper(BaseModel):
    """In-memory preparation for one Deduper, independent of its specification.

    State can contain arbitrary backend objects and need not be serialisable.
    Baseline data is supplied explicitly to each action by the caller.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    methodology: Deduper
    state: Any = None

    def dedupe(self, data: pl.DataFrame, *, baseline: pl.DataFrame) -> pl.DataFrame:
        """Score supplied rows using this preparation and the explicit baseline."""
        return self.methodology.dedupe(self.state, data, baseline=baseline)

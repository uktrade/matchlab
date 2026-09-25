"""Base class for linkers."""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Linker(BaseModel, ABC):
    """A methodology that finds candidate matches between two record steps.

    A `Model` step calls `prepare()` with both complete inputs before `link()`.
    Supplied rows are additions to the prepared inputs, not replacements.
    `link()` scores pairs involving at least one supplied row. When both sides are
    supplied, it also scores new left rows against new right rows. It does not
    return baseline-only pairs or change the prepared inputs. Collection supplies
    both complete inputs. `link()` returns `left_id`, `right_id`, and `score`.
    `normalise_model_scores` casts the result to `SCHEMA_MODEL_EDGES`.

    Every field is a setting unless marked `matchlab.resources.FromResources`. A
    fingerprint ignores a resource, so a marked field must not change what this scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump it to retire every artifact the previous body wrote.
    # Left unset, the step re-runs on every collect, and so does everything below it.
    # See `matchlab.core.versioning`.
    version: ClassVar[int | None] = None

    left_id: Literal["id"] = Field(
        default="id", description="The unique ID field in the left data"
    )
    right_id: Literal["id"] = Field(
        default="id", description="The unique ID field in the right data"
    )

    @abstractmethod
    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Prepare complete left and right baseline inputs."""
        ...

    @abstractmethod
    def link(
        self, left: pl.DataFrame | None = None, right: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """Score pairs involving supplied rows against prepared and supplied rows.

        At least one side must be supplied. Supplying both full baselines produces
        the complete collection result.
        """
        ...

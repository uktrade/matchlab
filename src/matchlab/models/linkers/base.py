"""Base class for linkers."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Linker(BaseModel, ABC):
    """A methodology that finds candidate matches between two record steps.

    A `Model` step calls `prepare()` with both complete inputs, retaining the
    returned `PreparedLinker`. Preparation does not change this specification.
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

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> "PreparedLinker":
        """Return independent runtime preparation from complete baseline inputs.

        Override this to return derived state; the default needs no setup.
        """
        return PreparedLinker(methodology=self)

    @abstractmethod
    def link(
        self,
        prepared_state: Any,  # noqa: ANN401 - opaque state owned by the author
        left: pl.DataFrame | None = None,
        right: pl.DataFrame | None = None,
        *,
        baseline_left: pl.DataFrame,
        baseline_right: pl.DataFrame,
    ) -> pl.DataFrame:
        """Score pairs involving supplied rows against prepared and supplied rows.

        At least one side must be supplied. Supplying both full baselines produces
        the complete collection result.
        """
        ...


class PreparedLinker(BaseModel):
    """In-memory preparation for one Linker, independent of its specification.

    State can contain arbitrary backend objects and need not be serialisable.
    Baseline data is supplied explicitly to each action by the caller.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    methodology: Linker
    state: Any = None

    def link(
        self,
        left: pl.DataFrame | None = None,
        right: pl.DataFrame | None = None,
        *,
        baseline_left: pl.DataFrame,
        baseline_right: pl.DataFrame,
    ) -> pl.DataFrame:
        """Score supplied rows using this preparation and explicit baseline inputs."""
        return self.methodology.link(
            self.state,
            left,
            right,
            baseline_left=baseline_left,
            baseline_right=baseline_right,
        )

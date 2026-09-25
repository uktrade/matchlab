"""Explode changes granularity. Cross-join each `id`'s columns to every combination."""

from functools import reduce
from operator import mul
from typing import ClassVar

import polars as pl
from pydantic import Field, field_validator

from matchlab.transformers.base import Transformer


class Explode(Transformer):
    """Give every combination of each `id`'s non-null column values its own row.

    `Group` collapses each `id` to one row. `Explode` goes the other way, giving the
    true cross product a `list` aggregate can't express. Every non-null value in one
    column pairs with every non-null value in every other column, for the same `id`.
    Values are deduplicated per column first, so a value repeated across records
    counts once. A column that is all null for one `id` still keeps that row, with
    `null` there, instead of dropping it.

    An `id` with several populated columns can generate many rows fast, so
    `max_combinations` caps it and raises, naming the `id`, rather than silently
    building an enormous table.
    """

    version: ClassVar[int] = 1

    max_combinations: int = Field(
        default=100,
        description="Reject a group whose cross product would exceed this many rows.",
    )

    @field_validator("max_combinations")
    @classmethod
    def _positive(cls, max_combinations: int) -> int:
        """Zero or fewer combinations is a way to reject every group, not a cap."""
        if max_combinations <= 0:
            raise ValueError("max_combinations must be positive.")
        return max_combinations

    def prepare(self, data: pl.DataFrame) -> None:
        """This transform needs no baseline state."""

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        """Cross-join each `id`'s columns to every combination of their values."""
        columns = [column for column in data.columns if column != "id"]
        if not columns:
            return data.select("id").unique(maintain_order=True)

        grouped = data.group_by("id", maintain_order=True).agg(
            [pl.col(column).drop_nulls().unique() for column in columns]
        )

        # Length 1 stands in for an empty list too, since exploding one keeps the row.
        sizes = [pl.col(column).list.len().clip(lower_bound=1) for column in columns]
        counts = grouped.select("id", reduce(mul, sizes).alias("combinations"))
        overflow = counts.filter(pl.col("combinations") > self.max_combinations)
        if overflow.height:
            row = overflow.row(0, named=True)
            raise ValueError(
                f"id {row['id']} would generate {row['combinations']:,} "
                f"combinations, which exceeds the maximum of "
                f"{self.max_combinations}. This would cause a Cartesian explosion."
            )

        # One column at a time compounds into the true cross product. Exploding
        # several columns in one call zips them in parallel instead.
        exploded = grouped
        for column in columns:
            exploded = exploded.explode(column, empty_as_null=True)
        return exploded

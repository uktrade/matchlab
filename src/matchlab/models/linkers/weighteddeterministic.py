"""A linking methodology that applies different weights to field comparisons."""

from typing import ClassVar

import duckdb
import polars as pl
from pydantic import BaseModel, Field, field_validator

from matchlab.core.sql import SQLCondition
from matchlab.models import comparison
from matchlab.models.linkers.base import Linker
from matchlab.models.linkers.deterministic import _combine_input


class WeightedComparison(BaseModel):
    """A comparison condition, and the weight it contributes to a pair's score."""

    comparison: SQLCondition = Field(
        description="""
            A valid ON clause comparing fields between the left and the right data.

            Qualify every column with `l` or `r`. For example:

            "l.company_name = r.company_name"
        """
    )
    weight: float = Field(
        description="""
            A weight to give this comparison. Use 1 for all comparisons to give
            uniform weight to each.
        """
    )

    @field_validator("comparison")
    @classmethod
    def validate_comparison(cls, v: SQLCondition) -> SQLCondition:
        """Validate the comparison string."""
        comp_val = comparison(v, dialect="duckdb")
        return comp_val


class WeightedDeterministicLinker(Linker):
    """Scores a pair by the weighted share of comparisons it matches on."""

    version: ClassVar[int] = 2

    weighted_comparisons: list[WeightedComparison] = Field(
        description="A list of tuples in the form of a comparison, and a weight."
    )
    threshold: float = Field(
        description="""
            The score above which matches will be kept. 
            
            Inclusive, so a value of 1 will keep only exact matches across all 
            comparisons.
        """,
        ge=0,
        le=1,
    )

    def link(
        self,
        prepared_state: object,
        left: pl.DataFrame | None = None,
        right: pl.DataFrame | None = None,
        *,
        baseline_left: pl.DataFrame,
        baseline_right: pl.DataFrame,
    ) -> pl.DataFrame:
        """Score supplied additions against the baseline and each other.

        Keep only pairs scoring at or above `threshold`.
        """
        if left is None and right is None:
            raise ValueError("Provide at least one affected side to link()")

        # Used below but ruff can't detect
        left_df = _combine_input(baseline_left, left)  # noqa: F841
        right_df = _combine_input(  # noqa: F841
            baseline_right, right
        )

        match_subquery = []
        weights = []

        for weighted_comparison in self.weighted_comparisons:
            match_subquery.append(
                f"""
                    select distinct on (raw.left_id, raw.right_id)
                        raw.left_id,
                        raw.right_id,
                        1.0 * {weighted_comparison.weight} as score
                    from (
                        select
                            l.{self.left_id} as left_id,
                            r.{self.right_id} as right_id,
                        from
                            left_df l
                        inner join right_df r on
                            ({weighted_comparison.comparison})
                            and (l._matchlab_affected or r._matchlab_affected)
                    ) raw
                """
            )
            weights.append(weighted_comparison.weight)

        match_subquery = " union all ".join(match_subquery)
        total_weight = sum(weights)

        sql = f"""
            select
                matches.left_id,
                matches.right_id,
                sum(matches.score) / {total_weight} as score
            from
                ({match_subquery}) matches
            group by
                matches.left_id,
                matches.right_id
            having
                sum(matches.score) / 
                    {total_weight} >= {self.threshold};
        """

        return (
            duckdb.sql(sql)
            .pl()
            .with_columns(
                [
                    pl.col("left_id").cast(baseline_left[self.left_id].dtype),
                    pl.col("right_id").cast(baseline_right[self.right_id].dtype),
                    pl.col("score").cast(pl.Float32),
                ]
            )
        )

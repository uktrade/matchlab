"""A deduplication methodology based on a deterministic set of conditions."""

from typing import ClassVar

import duckdb
import polars as pl
from pydantic import Field, PrivateAttr

from matchlab.models.dedupers.base import Deduper


class NaiveDeduper(Deduper):
    """Groups records that match exactly on every field in `unique_fields`."""

    version: ClassVar[int] = 2

    unique_fields: list[str] = Field(
        description="A list of fields that will form a unique, deduplicated record"
    )

    _baseline: pl.DataFrame | None = PrivateAttr(default=None)

    def prepare(self, data: pl.DataFrame) -> None:
        """Keep the full input for comparisons with affected records."""
        self._baseline = data.clone()

    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Score pairs involving affected records, with each match scoring 1.0."""
        if self._baseline is None:
            raise RuntimeError("Call prepare() before dedupe()")

        id_dtype = self._baseline[self.id].dtype
        baseline = self._baseline  # noqa: F841
        affected = data.clone()  # noqa: F841

        join_clause = []
        for field in self.unique_fields:
            join_clause.append(f"l.{field} = r.{field}")
        join_clause_compiled = " and ".join(join_clause)

        # The first join arm includes baseline matches. The second includes pairs
        # within the affected input. Never join the baseline to itself.
        sql = f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1.0 as score
            from (
                select
                    l.{self.id} as left_id,
                    r.{self.id} as right_id
                from
                    affected l
                inner join baseline r on {join_clause_compiled}
                union all
                select
                    l.{self.id} as left_id,
                    r.{self.id} as right_id
                from
                    affected l
                inner join affected r on {join_clause_compiled}
            ) raw
                where raw.left_id != raw.right_id;
        """

        return (
            duckdb.sql(sql)
            .pl()
            .with_columns(
                [
                    pl.col("left_id").cast(id_dtype),
                    pl.col("right_id").cast(id_dtype),
                    pl.col("score").cast(pl.Float32),
                ]
            )
        )

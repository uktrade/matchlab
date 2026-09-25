"""Group changes granularity. Collapse each `id` to one row."""

from typing import ClassVar

import polars as pl
from pydantic import Field, field_validator
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchlab.core.sql import SQLExpression
from matchlab.transformers.base import Transformer, reject_id_output, run_sql


class Group(Transformer):
    """Collapse each `id` to a single row, combining columns with aggregate SQL.

    Reading several records per entity (through a resolver) gives a model more evidence,
    but a comparison needs one row. `Group` says how each column combines. `id` is the
    grouping key, and every named expression is a DuckDB aggregate, for example
    `any_value(crn_company)` where records agree or `list(distinct crn_town)` where they
    don't. Only `id` and the named aggregates survive, since grouping has to decide how
    every column collapses.
    """

    version: ClassVar[int] = 1

    aggregates: dict[str, SQLExpression] = Field(
        description="Output column name to a DuckDB aggregate expression."
    )

    @field_validator("aggregates")
    @classmethod
    def _non_empty(
        cls, aggregates: dict[str, SQLExpression]
    ) -> dict[str, SQLExpression]:
        """Grouping collapses rows, so each column must say how it combines."""
        if not aggregates:
            raise ValueError(
                "Group needs aggregate expressions: grouping collapses several records "
                "into one row, so each column has to say how it combines. Pass "
                "aggregates such as {'name': 'any_value(crn_company)'}."
            )
        reject_id_output(aggregates)
        return aggregates

    def prepare(self, data: pl.DataFrame) -> None:
        """This transform needs no baseline state."""

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        """Collapse each `id` to one row using the aggregate expressions.

        A non-aggregate expression raises DuckDB's own error, naming the column.
        """
        identifier = expressions.alias_(expressions.column("id"), "id")
        projection: list[expressions.Expression] = [identifier]
        for alias, sql in self.aggregates.items():
            projection.append(
                expressions.alias_(parse_one(sql, dialect="duckdb"), alias)
            )

        query = (
            sqlglot_select(*projection, dialect="duckdb").from_("data").group_by("id")
        )
        return run_sql(query.sql(dialect="duckdb"), data)

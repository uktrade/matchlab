"""Clean derives columns. Add or replace named columns, keep the rest."""

from typing import ClassVar

import polars as pl
from pydantic import Field, field_validator
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchlab.core.sql import SQLExpression
from matchlab.transformers.base import Transformer, reject_id_output, run_sql


class Clean(Transformer):
    """Derive columns with DuckDB SQL, without dropping unrelated fields.

    Each entry maps an output column name to a SQL expression over the record step's
    columns
    (the source-qualified names, `crn_company`). An alias that names an existing column
    replaces it. A new alias is added. Every other column passes through untouched.
    Dropping is `Select`'s job, not `Clean`'s.

    `id` is not writable. It is the grouping a model matches on, so an alias naming
    it is refused rather than allowed to displace it.
    """

    version: ClassVar[int] = 1

    cleaning: dict[str, SQLExpression] = Field(
        description="Output column name to a DuckDB SQL expression over the step."
    )

    @field_validator("cleaning")
    @classmethod
    def _non_empty(cls, cleaning: dict[str, SQLExpression]) -> dict[str, SQLExpression]:
        """A clean that derives nothing is meaningless. 'No cleaning' is no `Clean`."""
        if not cleaning:
            raise ValueError("Clean needs at least one expression to derive.")
        reject_id_output(cleaning)
        return cleaning

    def apply(
        self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
    ) -> pl.DataFrame:
        """Add or replace the named columns, keeping the rest.

        Invalid SQL raises at build time, naming the expression.
        """
        collisions = [alias for alias in self.cleaning if alias in data.columns]
        star = expressions.Star(except_=[expressions.column(a) for a in collisions])
        projection: list[expressions.Expression] = [star]
        for alias, sql in self.cleaning.items():
            projection.append(
                expressions.alias_(parse_one(sql, dialect="duckdb"), alias)
            )

        query = sqlglot_select(*projection, dialect="duckdb").from_("data")
        return run_sql(query.sql(dialect="duckdb"), data)

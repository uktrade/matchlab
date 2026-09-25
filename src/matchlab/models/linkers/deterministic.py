"""A linking methodology based on a deterministic set of conditions."""

import json
from typing import ClassVar

import duckdb
import polars as pl
from pydantic import Field, PrivateAttr, field_validator

from matchlab.core.logging import logger
from matchlab.core.sql import SQLCondition, SQLQuery
from matchlab.models import comparison
from matchlab.models.linkers.base import Linker


def _combine_input(
    baseline: pl.DataFrame, affected: pl.DataFrame | None
) -> pl.DataFrame:
    """Keep distinct baseline rows alongside supplied rows, marking the latter."""
    if affected is None:
        return baseline.with_columns(pl.lit(False).alias("_matchlab_affected"))
    remaining = baseline.join(
        affected.select(baseline.columns).unique(),
        on=baseline.columns,
        how="anti",
        nulls_equal=True,
    )
    return pl.concat(
        [
            remaining.with_columns(pl.lit(False).alias("_matchlab_affected")),
            affected.with_columns(pl.lit(True).alias("_matchlab_affected")),
        ]
    )


class DeterministicLinker(Linker):
    """A deterministic linker that links based on a set of boolean conditions.

    Uses DuckDB as the SQL backend, enabling rich SQL operations while maintaining
    a Polars DataFrame interface. Supports both parallel matching (single round)
    and sequential matching (multiple rounds where matched records are removed
    after each round).
    """

    version: ClassVar[int] = 2

    comparisons: list[SQLCondition] | list[list[SQLCondition]] = Field(
        description="""
            Match conditions, in DuckDB SQL. Qualify every column with `l` or `r`.

            A flat list applies every condition in parallel, unioned with OR logic:

                [
                    "l.company_number = r.company_number",
                    "l.name = r.name",
                ]

            A nested list runs sequential rounds instead. Conditions within a round
            use OR logic. After a round, matched records leave the pool before the
            next round runs:

                [
                    [
                        "l.company_number = r.company_number",
                        "l.name = r.name",
                    ],
                    [
                        "l.name_normalised = r.name_normalised",
                        "l.website = r.website",
                    ],
                ]

            Supports any DuckDB SQL expression, not just equality.
        """,
    )

    _baseline_left: pl.DataFrame | None = PrivateAttr(default=None)
    _baseline_right: pl.DataFrame | None = PrivateAttr(default=None)

    @field_validator("comparisons", mode="before")
    @classmethod
    def validate_comparison(
        cls, value: SQLCondition | list[SQLCondition] | list[list[SQLCondition]]
    ) -> list[list[SQLCondition]]:
        """Normalise to a list of rounds, and validate each comparison string."""
        if isinstance(value, str):
            return [[comparison(value, dialect="duckdb")]]
        if not value:
            raise ValueError("comparisons cannot be empty")
        if all(isinstance(v, str) for v in value):
            return [[comparison(v, dialect="duckdb") for v in value]]
        if all(isinstance(v, list) for v in value):
            for round_idx, round_comparisons in enumerate(value):
                if not round_comparisons:
                    raise ValueError(f"Round {round_idx} cannot be empty")
                if not all(isinstance(c, str) for c in round_comparisons):
                    raise ValueError(f"Round {round_idx} must contain only strings")
            return [[comparison(c, dialect="duckdb") for c in r] for r in value]
        raise ValueError(
            "comparisons must be a string, list of strings, or list of lists"
        )

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Keep both inputs for later affected-side calls."""
        self._baseline_left = left.clone()
        self._baseline_right = right.clone()

    def link(
        self, left: pl.DataFrame | None = None, right: pl.DataFrame | None = None
    ) -> pl.DataFrame:
        """Link supplied additions against the prepared baseline and each other.

        Sequential rounds share one call, so earlier matches leave the pool
        before later rounds run.
        """
        if self._baseline_left is None or self._baseline_right is None:
            raise RuntimeError("Call prepare() before link()")
        if left is None and right is None:
            raise ValueError("Provide at least one affected side to link()")

        con: duckdb.DuckDBPyConnection = duckdb.connect(":memory:")
        try:
            all_matches: list[pl.DataFrame] = []
            remaining_left = _combine_input(self._baseline_left, left)
            remaining_right = _combine_input(self._baseline_right, right)

            for round_num, round_comparisons in enumerate(self.comparisons, start=1):
                if remaining_left.is_empty() or remaining_right.is_empty():
                    logger.info(f"Round {round_num}: Skipping - no records remaining")
                    break

                logger.info(
                    f"Round {round_num}: {len(remaining_left):,} left × "
                    f"{len(remaining_right):,} right"
                )

                matches = self._link_round(
                    con, remaining_left, remaining_right, round_comparisons, round_num
                )

                logger.info(f"Round {round_num}: Found {len(matches):,} matches")

                if not matches.is_empty():
                    all_matches.append(matches)
                    matched_left = matches.select("left_id").unique()
                    matched_right = matches.select("right_id").unique()
                    remaining_left = remaining_left.join(
                        matched_left,
                        left_on=self.left_id,
                        right_on="left_id",
                        how="anti",
                    )
                    remaining_right = remaining_right.join(
                        matched_right,
                        left_on=self.right_id,
                        right_on="right_id",
                        how="anti",
                    )

            return self._finalise_results(all_matches)
        finally:
            con.close()

    def _link_round(
        self,
        con: duckdb.DuckDBPyConnection,
        left: pl.DataFrame,
        right: pl.DataFrame,
        comparisons: list[SQLCondition],
        round_num: int,
    ) -> pl.DataFrame:
        """Apply all comparisons in a round using OR logic via DuckDB."""
        con.register("left_df", left)
        con.register("right_df", right)

        subqueries: list[SQLQuery] = []
        for condition in comparisons:
            subquery: SQLQuery = f"""
                SELECT
                    l.{self.left_id} AS left_id,
                    r.{self.right_id} AS right_id,
                    1.0 AS score
                FROM left_df l
                INNER JOIN right_df r
                    ON ({condition})
                    AND (l._matchlab_affected OR r._matchlab_affected)
            """
            subqueries.append(subquery)

        query: SQLQuery = f"""
            SELECT DISTINCT *
            FROM ({" UNION ALL ".join(subqueries)})
        """

        max_est: int = self._get_max_cardinality(con, query)
        logger.info(f"Round {round_num}: Estimated max cardinality: {max_est:,}")

        return con.execute(query).pl()

    def _get_max_cardinality(
        self, con: duckdb.DuckDBPyConnection, query: SQLQuery
    ) -> int:
        """Get max cardinality estimate from DuckDB plan, or -1 if unavailable."""
        explain = con.execute(
            f"PRAGMA explain_output = 'all'; EXPLAIN (FORMAT json) {query}"
        ).fetchall()
        plans = {k: json.loads(v) for k, v in dict(explain).items()}

        estimates: list[dict] = []
        for plan_name, plan_type in [
            ("physical_plan", "physical"),
            ("logical_opt", "optimised"),
        ]:
            if plan := plans.get(plan_name):
                estimates.extend(self._traverse_plan(plan[0], plan_type))

        # Debug: log full tree breakdown
        logger.debug("Plan breakdown:")
        for plan_type in ["physical", "optimised"]:
            plan_ests = [e for e in estimates if e["plan_type"] == plan_type]
            if plan_ests:
                logger.debug(f"  {plan_type.capitalize()} plan:")
                for est in plan_ests:
                    indent = "    " + "  " * est["depth"]
                    logger.debug(f"{indent}{est['node_name']}: {est['cardinality']:,}")

        # Return max for info logging
        return max(
            (e["cardinality"] for e in estimates if e["cardinality"] > 0),
            default=-1,
        )

    def _traverse_plan(self, node: dict, plan_type: str, depth: int = 0) -> list[dict]:
        """Recursively collect cardinality estimates with metadata from plan tree."""
        estimates: list[dict] = []
        cardinality = node.get("extra_info", {}).get("Estimated Cardinality")
        if cardinality is not None:
            estimates.append(
                {
                    "cardinality": int(cardinality),
                    "node_name": node.get("name", "UNKNOWN"),
                    "depth": depth,
                    "plan_type": plan_type,
                }
            )
        for child in node.get("children", []):
            estimates.extend(self._traverse_plan(child, plan_type, depth + 1))
        return estimates

    def _finalise_results(self, all_matches: list[pl.DataFrame]) -> pl.DataFrame:
        """Combine matches from all rounds and ensure correct schema."""
        if all_matches:
            return pl.concat(all_matches).with_columns(pl.col("score").cast(pl.Float32))
        return pl.DataFrame(
            schema={
                "left_id": self._baseline_left[self.left_id].dtype,
                "right_id": self._baseline_right[self.right_id].dtype,
                "score": pl.Float32,
            }
        )

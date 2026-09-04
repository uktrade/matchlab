"""Connected-components resolver methodology."""

from collections.abc import Mapping
from typing import Annotated, ClassVar

import polars as pl
from fastdsu import connected_components
from pydantic import Field

from matchlab.resolvers.base import (
    SCHEMA_CLUSTERS,
    ResolverMethod,
    ResolverType,
)


class Components(ResolverMethod):
    """Resolver methodology that computes connected components.

    A threshold defaults to 0.0 if not set.
    """

    version: ClassVar[int] = 1

    resolver_type: ClassVar[ResolverType] = ResolverType.COMPONENTS
    thresholds: dict[int, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict,
        description=(
            "Minimum score for an edge to count, per input, keyed by the input's "
            "position. Write these as `{model: 0.9}`. `Resolver` takes the model "
            "object and works out the position."
        ),
    )

    def compute_clusters(  # noqa: D102
        self,
        model_edges: Mapping[int, pl.DataFrame],
    ) -> pl.DataFrame:
        filtered = [
            edges_item.filter(pl.col("score") >= self.thresholds.get(position, 0.0))
            for position, edges_item in model_edges.items()
            if edges_item.height > 0
        ]

        if not filtered:
            return pl.from_arrow(SCHEMA_CLUSTERS.empty_table())

        edges = pl.concat(filtered).select(
            pl.col("left_id").alias("src"),
            pl.col("right_id").alias("dst"),
        )

        return (
            pl.from_arrow(
                connected_components(
                    edges["src"].to_arrow(),
                    edges["dst"].to_arrow(),
                )
            )
            .rename({"label": "parent_id", "key": "child_id"})
            .cast(pl.Schema(SCHEMA_CLUSTERS))
        )

"""Tests for the Components resolver methodology."""

import polars as pl

from matchlab.resolvers import Components
from matchlab.resolvers.base import SCHEMA_CLUSTERS


def test_components_uses_thresholds() -> None:
    """`Components.compute_clusters` honours a per-model threshold."""
    method = Components(thresholds={0: 0.6})
    model_edges = {
        0: pl.DataFrame(
            {
                "left_id": [1, 2],
                "right_id": [2, 3],
                "score": [0.8, 0.4],
            },
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "score": pl.Float32,
            },
        )
    }

    clusters = method.prepare(model_edges).compute_clusters(
        model_edges=model_edges, baseline_model_edges=model_edges
    )

    grouped_clusters = {
        frozenset(group["child_id"].to_list())
        for group in clusters.partition_by("parent_id")
    }
    assert grouped_clusters == {frozenset({1, 2})}


def test_components_no_edges() -> None:
    """`Components.compute_clusters` works with no data."""
    clusters = (
        Components()
        .prepare({})
        .compute_clusters(model_edges={}, baseline_model_edges={})
    )
    assert clusters.height == 0
    assert clusters.schema == pl.Schema(SCHEMA_CLUSTERS)


def test_components_merges_models() -> None:
    """`Components.compute_clusters` merges edges from multiple models."""
    method = Components(
        thresholds={0: 0.0}  # the second input's threshold is implicit
    )
    model_edges = {
        0: pl.DataFrame(
            {"left_id": [1], "right_id": [2], "score": [0.9]},
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "score": pl.Float32,
            },
        ),
        1: pl.DataFrame(
            {"left_id": [3], "right_id": [4], "score": [0.8]},
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "score": pl.Float32,
            },
        ),
    }

    clusters = method.prepare(model_edges).compute_clusters(
        model_edges=model_edges, baseline_model_edges=model_edges
    )

    grouped_clusters = {
        frozenset(group["child_id"].to_list())
        for group in clusters.partition_by("parent_id")
    }
    assert grouped_clusters == {frozenset({1, 2}), frozenset({3, 4})}


def test_components_threshold_no_edges() -> None:
    """A methodology takes positions on trust. `Resolver` is what checks them.

    `Components.compute_clusters` doesn't validate that a threshold names a real
    input. That check happens when the resolver is constructed, which still has the
    model object in hand. See `test_threshold_must_name_input`.
    """
    method = Components(thresholds={7: 0.5})

    clusters = method.prepare(
        {
            0: pl.DataFrame(
                {"left_id": [1], "right_id": [2], "score": [0.9]},
                schema={
                    "left_id": pl.UInt64,
                    "right_id": pl.UInt64,
                    "score": pl.Float32,
                },
            )
        }
    ).compute_clusters(
        model_edges={
            0: pl.DataFrame(
                {"left_id": [1], "right_id": [2], "score": [0.9]},
                schema={
                    "left_id": pl.UInt64,
                    "right_id": pl.UInt64,
                    "score": pl.Float32,
                },
            )
        },
        baseline_model_edges={
            0: pl.DataFrame(
                {"left_id": [1], "right_id": [2], "score": [0.9]},
                schema={
                    "left_id": pl.UInt64,
                    "right_id": pl.UInt64,
                    "score": pl.Float32,
                },
            )
        },
    )

    assert {
        frozenset(group["child_id"].to_list())
        for group in clusters.partition_by("parent_id")
    } == {frozenset({1, 2})}

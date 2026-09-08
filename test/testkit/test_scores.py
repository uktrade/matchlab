"""Tests for generating entity scores and recovering clusters from them."""

import polars as pl
import pytest
from fastdsu import connected_components

from matchlab.core.schemas import SCHEMA_MODEL_EDGES
from matchlab.testkit.entities import Cluster, EntityReference, TrueEntity
from matchlab.testkit.models import generate_entity_scores
from test.testkit.test_entities import (
    make_cluster_entity,
    make_source_entity,
)


@pytest.mark.parametrize(
    ("left_entities", "right_entities", "true_entities", "score_range", "expected"),
    [
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["a2"]),
                ]
            ),
            None,  # Deduplication case
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.8, 1.0),
            {"edge_count": 1, "score_range": (0.8, 1.0)},
            id="basic_dedupe",
        ),
        pytest.param(
            frozenset([make_cluster_entity(1, "left", ["a1"])]),
            frozenset([make_cluster_entity(2, "right", ["b1"])]),
            frozenset(
                [
                    make_source_entity("left", ["a1"], "a"),
                    make_source_entity("right", ["b1"], "b"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 0, "score_range": (0.8, 1.0)},
            id="basic_link_no_match",
        ),
        pytest.param(
            frozenset([make_cluster_entity(1, "test", ["a1"])]),
            frozenset([make_cluster_entity(2, "test", ["a2"])]),
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.8, 1.0),
            {"edge_count": 1, "score_range": (0.8, 1.0)},
            id="successful_link",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1", "a2"]),
                    make_cluster_entity(2, "test", ["a2", "a3"]),
                    make_cluster_entity(3, "test", ["a3", "a4"]),
                ]
            ),
            None,
            frozenset(
                [make_source_entity("test", ["a1", "a2", "a3", "a4"], "entity_a")]
            ),
            (0.8, 1.0),
            {"edge_count": 3, "score_range": (0.8, 1.0)},
            id="overlapping_dedupe",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["b1"]),
                ]
            ),
            frozenset(
                [
                    make_cluster_entity(3, "test", ["a2"]),
                    make_cluster_entity(4, "test", ["b2"]),
                ]
            ),
            frozenset(
                [
                    make_source_entity("test", ["a1", "a2"], "a"),
                    make_source_entity("test", ["b1", "b2"], "b"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 2, "score_range": (0.8, 1.0)},
            id="multi_component_link",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["a2"]),
                    make_cluster_entity(3, "test", ["x1"]),  # No source for this
                    make_cluster_entity(4, "test", ["y1"]),  # No source for this
                ]
            ),
            None,
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.8, 1.0),
            {"edge_count": 1, "score_range": (0.8, 1.0)},
            id="partial_source_coverage",
        ),
        pytest.param(
            frozenset(),
            frozenset(),
            frozenset(),
            (0.8, 1.0),
            {"edge_count": 0, "score_range": (0.8, 1.0)},
            id="empty_sets",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["a2"]),
                ]
            ),
            None,
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.5, 0.7),
            {"edge_count": 1, "score_range": (0.5, 0.7)},
            id="different_score_range",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1", "a2"]),  # Merged
                    make_cluster_entity(2, "test", ["a3"]),  # Unmerged
                    make_cluster_entity(3, "test", ["b1"]),  # From different source
                ]
            ),
            None,
            frozenset(
                [
                    make_source_entity("test", ["a1", "a2", "a3"], "a"),
                    make_source_entity("test", ["b1"], "b"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 1, "score_range": (0.8, 1.0)},
            id="mixed_merged_unmerged",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "source1", ["1"]),
                    make_cluster_entity(2, "source2", ["A"]),
                    make_cluster_entity(3, "source3", ["X"]),
                ]
            ),
            None,
            frozenset(
                [
                    make_source_entity("source1", ["1"], "entity"),
                    make_source_entity("source2", ["A"], "entity"),
                    make_source_entity("source3", ["X"], "entity"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 0, "score_range": (0.8, 1.0)},
            id="multi_source_entity",
        ),
    ],
)
def test_generate_entity_scores_scenarios(
    left_entities: frozenset[Cluster],
    right_entities: frozenset[Cluster] | None,
    true_entities: frozenset[TrueEntity],
    score_range: tuple[float, float],
    expected: dict,
) -> None:
    """generate_entity_scores emits one edge per pair the true entities connect."""
    # Run the function
    result = generate_entity_scores(
        left_entities, right_entities, true_entities, score_range
    )

    assert result.schema == pl.Schema(SCHEMA_MODEL_EDGES)

    edges = list(
        zip(
            result["left_id"].to_list(),
            result["right_id"].to_list(),
            strict=True,
        )
    )

    assert len(edges) == expected["edge_count"]

    if edges:
        score_values = result["score"].to_numpy()
        score_min, score_max = expected["score_range"]
        assert all(score_min <= p <= score_max for p in score_values)


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal", "case"),
    [
        pytest.param(42, 42, True, "dedupe", id="same_seeds_dedupe"),
        pytest.param(1, 2, False, "dedupe", id="different_seeds_dedupe"),
        pytest.param(42, 42, True, "link", id="same_seeds_link"),
        pytest.param(1, 2, False, "link", id="different_seeds_link"),
    ],
)
def test_seed_determinism(
    seed1: int,
    seed2: int,
    should_be_equal: bool,
    case: str,
) -> None:
    """The same seed reproduces identical scores. A different seed changes them."""
    # Create test entities
    source = make_source_entity("test", ["a1", "a2", "a3"], "entity_a")
    entities = frozenset(
        [
            make_cluster_entity(1, "test", ["a1"]),
            make_cluster_entity(2, "test", ["a2"]),
            make_cluster_entity(3, "test", ["a3"]),
        ]
    )

    if case == "dedupe":
        right_entities = None
    else:
        # A linking case needs a second, disjoint set of entities to link against
        right_entities = frozenset(
            [
                make_cluster_entity(4, "test", ["a1"]),
                make_cluster_entity(5, "test", ["a2"]),
                make_cluster_entity(6, "test", ["a3"]),
            ]
        )

    result1 = generate_entity_scores(
        left_entities=entities,
        right_entities=right_entities,
        true_entities=frozenset([source]),
        score_range=(0.8, 1.0),
        seed=seed1,
    )

    result2 = generate_entity_scores(
        left_entities=entities,
        right_entities=right_entities,
        true_entities=frozenset([source]),
        score_range=(0.8, 1.0),
        seed=seed2,
    )

    assert result1.shape[0] > 0
    assert result2.shape[0] > 0

    if should_be_equal:
        assert result1.equals(result2)
    else:
        assert not result1.equals(result2)


def test_disjoint_set_recovery() -> None:
    """A DisjointSet built from generated scores recovers the planted entities."""
    source1 = make_source_entity("source1", ["1", "2", "3"], "entity1")
    source2 = make_source_entity("source1", ["4", "5", "6"], "entity2")

    clusters = frozenset(
        [
            make_cluster_entity(1, "source1", ["1"]),
            make_cluster_entity(2, "source1", ["2"]),
            make_cluster_entity(3, "source1", ["3"]),
            make_cluster_entity(4, "source1", ["4"]),
            make_cluster_entity(5, "source1", ["5"]),
            make_cluster_entity(6, "source1", ["6"]),
        ]
    )

    table = generate_entity_scores(
        left_entities=clusters,
        right_entities=None,
        true_entities=frozenset([source1, source2]),
        score_range=(0.9, 1.0),
    )

    edges = table.filter(pl.col("score") >= 0.9).select(["left_id", "right_id"])

    labels = pl.from_arrow(
        connected_components(edges["left_id"].to_arrow(), edges["right_id"].to_arrow())
    )

    # Splitting each entity into six one-record clusters should recover exactly two
    assert labels["label"].unique().len() == 2

    cluster_sizes = labels.group_by("label").agg(pl.col("key").count())["key"].to_list()
    assert cluster_sizes == [3, 3]


@pytest.mark.parametrize(
    "score_range",
    [
        pytest.param((-0.1, 0.5), id="negative_lower_bound"),  # Negative lower bound
        pytest.param((0.5, 1.1), id="upper_bound_too_high"),  # Upper bound > 1.0
        pytest.param((0.8, 0.7), id="decreasing_range"),  # Decreasing range
    ],
)
def test_invalid_score_ranges(score_range: tuple[float, float]) -> None:
    """An out-of-order or out-of-bounds score range raises ValueError."""
    source = make_source_entity("test", ["a1", "a2"], "entity")
    entities = frozenset(
        [
            make_cluster_entity(1, "test", ["a1"]),
            make_cluster_entity(2, "test", ["a2"]),
        ]
    )

    with pytest.raises(ValueError, match="Scores must be"):
        generate_entity_scores(
            left_entities=entities,
            right_entities=None,
            true_entities=frozenset([source]),
            score_range=score_range,
        )


def test_complex_entity_recovery() -> None:
    """An entity fragmented across three sources still recovers as one component."""
    source = TrueEntity(
        base_values={"name": "Complex Entity"},
        keys=EntityReference(
            {
                "source1": frozenset(["1", "2"]),
                "source2": frozenset(["A", "B"]),
                "source3": frozenset(["X"]),
            }
        ),
    )

    clusters = frozenset(
        [
            Cluster(keys=EntityReference({"source1": frozenset(["1"])})),
            Cluster(keys=EntityReference({"source1": frozenset(["2"])})),
            Cluster(keys=EntityReference({"source2": frozenset(["A"])})),
            Cluster(keys=EntityReference({"source2": frozenset(["B"])})),
            Cluster(keys=EntityReference({"source3": frozenset(["X"])})),
        ]
    )

    table = generate_entity_scores(
        left_entities=clusters,
        right_entities=None,
        true_entities=frozenset([source]),
        score_range=(0.9, 1.0),
    )

    # Every one of the 5 fragments pairs with every other: n*(n-1)/2 = 10 edges
    assert len(table) == 10

    edges = table.filter(pl.col("score") >= 0.9).select(["left_id", "right_id"])

    labels = pl.from_arrow(
        connected_components(edges["left_id"].to_arrow(), edges["right_id"].to_arrow())
    )

    assert labels["label"].unique().len() == 1

    cluster_sizes = labels.group_by("label").agg(pl.col("key").count())["key"].to_list()
    assert cluster_sizes[0] == 5

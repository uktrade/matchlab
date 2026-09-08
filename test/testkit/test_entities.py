"""Tests for entity references, clusters, and generated entities."""

import operator
from typing import Any

import polars as pl
import pytest
from faker import Faker

from matchlab.testkit._generate import generate_entities
from matchlab.testkit.compare import diff_entities, scores_to_clusters
from matchlab.testkit.entities import Cluster, EntityReference, TrueEntity
from matchlab.testkit.features import FeatureConfig


def make_cluster_entity(id: int, *args: Any) -> Cluster:
    """Build a Cluster from id, source, keys pairs, e.g. `1, "d1", ["1", "2"]`."""
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be pairs of source name and keys list")

    keys = {}
    for i in range(0, len(args), 2):
        source = args[i]
        keys_list = args[i + 1]
        if not isinstance(source, str):
            raise TypeError(f"source name must be a string, got {type(source)}")
        if not isinstance(keys_list, list):
            raise TypeError(f"keys must be a list, got {type(keys_list)}")
        keys[source] = frozenset(keys_list)

    return Cluster(id=id, keys=EntityReference(keys))


def make_source_entity(source: str, keys: list[str], base_val: str) -> TrueEntity:
    """Helper to create a TrueEntity."""
    entity = TrueEntity(base_values={"name": base_val})
    entity.add_source_reference(source, keys)
    return entity


@pytest.mark.parametrize(
    ("name", "keys"),
    (
        ("source1", frozenset({"1", "2", "3"})),
        ("source2", frozenset({"A", "B"})),
    ),
)
def test_entity_reference_creation(name: str, keys: frozenset[str]) -> None:
    """An EntityReference stores each source's keys and reports unknown sources."""
    ref = EntityReference({name: keys})
    assert ref[name] == keys
    assert name in ref
    with pytest.raises(KeyError):
        ref["nonexistent"]


def test_entity_reference_addition() -> None:
    """Adding two EntityReferences unions the keys each holds for a shared source."""
    ref1 = EntityReference({"source1": frozenset({"1", "2"})})
    ref2 = EntityReference(
        {"source1": frozenset({"2", "3"}), "source2": frozenset({"A"})}
    )
    combined = ref1 + ref2
    assert combined["source1"] == frozenset({"1", "2", "3"})
    assert combined["source2"] == frozenset({"A"})


def test_entity_reference_subset() -> None:
    """`<=` holds when every source's keys in one reference are covered by the other."""
    subset = EntityReference({"source1": frozenset({"1", "2"})})
    superset = EntityReference(
        {"source1": frozenset({"1", "2", "3"}), "source2": frozenset({"A"})}
    )

    assert subset <= superset
    assert not superset <= subset


def test_cluster_entity_creation() -> None:
    """A Cluster keeps the keys it was given and gets an integer id."""
    ref = EntityReference({"source1": frozenset({"1", "2"})})
    entity = Cluster(keys=ref)

    assert entity.keys == ref
    assert isinstance(entity.id, int)


def test_cluster_entity_addition() -> None:
    """Adding two Clusters unions their keys for a shared source."""
    entity1 = Cluster(keys=EntityReference({"source1": frozenset({"1"})}))
    entity2 = Cluster(keys=EntityReference({"source1": frozenset({"2"})}))

    combined = entity1 + entity2
    assert combined.keys["source1"] == frozenset({"1", "2"})


def test_source_entity_creation() -> None:
    """A TrueEntity keeps its base values, its keys, and gets an integer id."""
    base_values = {"name": "John", "age": 30}
    ref = EntityReference({"source1": frozenset({"1", "2"})})

    entity = TrueEntity(base_values=base_values, keys=ref)

    assert entity.base_values == base_values
    assert entity.keys == ref
    assert isinstance(entity.id, int)


def test_entity_reference_rejects_foreign_operands() -> None:
    """Union and subset against a non-reference are type errors, not coercions."""
    ref = EntityReference({"s": frozenset({"1"})})
    with pytest.raises(TypeError):
        ref + "not a reference"
    with pytest.raises(TypeError):
        operator.le(ref, "not a reference")


def test_entity_id_converts_to_int() -> None:
    """An entity stands in for its integer id. `int(entity)` is that id."""
    assert int(make_cluster_entity(42, "s", ["1"])) == 42


def test_entity_sorts_by_id() -> None:
    """Entities order by id, so a list of them sorts predictably."""
    entities = [make_cluster_entity(i, "s", [str(i)]) for i in (3, 1, 2)]
    assert [int(e) for e in sorted(entities)] == [1, 2, 3]


def test_entity_compares_by_id() -> None:
    """The ordering operators compare an entity's id, to an entity or a bare int."""
    two = make_cluster_entity(2, "s", ["1"])
    five = make_cluster_entity(5, "s", ["2"])

    # against another entity
    assert two < five
    assert five > two
    assert two <= five
    assert five >= two

    # against a bare int
    assert two < 5
    assert five > 2
    assert two <= 2
    assert five >= 5


def test_entity_comparison_rejects_foreign_operand() -> None:
    """Ordering against an incomparable type is a type error, not a silent answer."""
    entity = make_cluster_entity(1, "s", ["1"])
    for op in (operator.lt, operator.gt, operator.le, operator.ge):
        with pytest.raises(TypeError):
            op(entity, object())


def test_cluster_entity_add_identity() -> None:
    """Adding `None` returns the cluster unchanged, so `sum()` over clusters works."""
    cluster = make_cluster_entity(1, "s", ["1"])
    assert (cluster + None) is cluster

    total = sum(
        [make_cluster_entity(1, "s", ["1"]), make_cluster_entity(2, "s", ["2"])]
    )
    assert total.keys["s"] == frozenset({"1", "2"})


def test_cluster_entity_rejects_foreign_operand() -> None:
    """The cluster operators refuse a non-cluster rather than coercing it.

    Equality is the exception. `cluster == other` is defined for any object and simply
    reports inequality, since a set of clusters must be able to hold non-cluster keys.
    """
    cluster = make_cluster_entity(1, "s", ["1"])
    with pytest.raises(TypeError):
        cluster + "x"
    with pytest.raises(TypeError):
        cluster - "x"
    with pytest.raises(TypeError):
        5 + cluster  # __radd__ only absorbs the 0 that starts sum()
    assert (cluster == "x") is False


def test_cluster_entity_reverse_diff() -> None:
    """`__rsub__` mirrors `__sub__`, so `a - b` and `b.__rsub__(a)` agree."""
    a = make_cluster_entity(1, "s", ["1", "2"])
    b = make_cluster_entity(2, "s", ["1"])

    assert a - b == {"s": frozenset({"2"})}
    assert b.__rsub__(a) == a - b


def test_source_entity_equals_its_int_id() -> None:
    """A true entity equals its own integer id, for lookups keyed by id."""
    entity = make_source_entity("s", ["1"], "alice")
    assert entity == entity.id
    assert (entity == "not an int or entity") is False


@pytest.mark.parametrize(
    ("features", "n"),
    (
        ((FeatureConfig(name="name", base_generator="name"),), 1),
        (
            (
                FeatureConfig(name="name", base_generator="name"),
                FeatureConfig(name="email", base_generator="email"),
            ),
            5,
        ),
    ),
)
def test_generate_entities(features: tuple[FeatureConfig, ...], n: int) -> None:
    """generate_entities returns n entities, each holding a value for every feature."""
    faker = Faker(seed=42)
    entities = generate_entities(faker, features, n)

    assert len(entities) == n
    for entity in entities:
        # Check all features are present
        assert all(f.name in entity.base_values for f in features)
        # Check all values are strings (given our test features)
        assert all(isinstance(v, str) for v in entity.base_values.values())


@pytest.mark.parametrize(
    (
        "scores",
        "left_clusters",
        "right_clusters",
        "threshold",
        "expected_count",
    ),
    [
        pytest.param(
            pl.DataFrame(
                {
                    "left_id": [1, 2],
                    "right_id": [2, 3],
                    "score": [0.9, 0.85],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
                make_cluster_entity(3, "test", ["a3"]),
            ),
            None,
            0.8,
            1,  # One merged entity containing all three records
            id="basic_dedupe_chain",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "left_id": [1],
                    "right_id": [4],
                    "score": [0.95],
                }
            ),
            (make_cluster_entity(1, "left", ["a1"]),),
            (make_cluster_entity(4, "right", ["b1"]),),
            0.9,
            1,  # One merged entity from the link
            id="basic_link_match",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "left_id": [1, 2],
                    "right_id": [2, 3],
                    "score": [0.75, 0.7],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
                make_cluster_entity(3, "test", ["a3"]),
            ),
            None,
            0.8,
            3,  # No merging due to threshold
            id="threshold_prevents_merge",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "left_id": [],
                    "right_id": [],
                    "score": [],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
            ),
            None,
            0.8,
            2,  # No merging with empty scores
            id="empty_scores",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "left_id": [1],
                    "right_id": [1],
                    "score": [0.9],
                }
            ),
            (make_cluster_entity(1, "left", ["a1"]),),
            (make_cluster_entity(1, "right", ["b1"]),),
            0.8,
            1,  # One merged entity, even though left and right happen to share a raw id
            id="colliding_ids_across_sides",
        ),
    ],
)
def test_scores_to_clusters(
    scores: pl.DataFrame,
    left_clusters: tuple[Cluster, ...],
    right_clusters: tuple[Cluster, ...] | None,
    threshold: float,
    expected_count: int,
) -> None:
    """scores_to_clusters merges only pairs scoring at or above the threshold."""
    result = scores_to_clusters(
        scores=scores,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        threshold=threshold,
    )

    assert len(result) == expected_count

    all_inputs = set(left_clusters)
    if right_clusters:
        all_inputs.update(right_clusters)

    for input_entity in all_inputs:
        assert any(input_entity in output_entity for output_entity in result)


def assert_deep_approx_equal(
    got: float | dict | list, want: float | dict | list
) -> None:
    """Compare nested structures, treating floats as equal within a small tolerance."""
    # Handle float comparison
    if isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-2)
        return

    # Handle dictionary comparison
    if isinstance(want, dict):
        assert isinstance(got, dict)
        assert set(want.keys()) <= set(got.keys())
        for k, v in want.items():
            assert_deep_approx_equal(got[k], v)
        return

    # Handle list comparison
    if isinstance(want, list):
        assert isinstance(got, list)
        assert len(got) == len(want)

        # Sort dict lists by whichever id field they share, so order doesn't matter.
        if want and all(isinstance(x, dict) for x in want + got):
            for id_key in ["entity_id", "expected_entity_id", "actual_entity_id"]:
                if all(id_key in x for x in want + got):
                    got = sorted(got, key=lambda x: x[id_key])
                    want = sorted(want, key=lambda x: x[id_key])
                    break

        for w, g in zip(want, got, strict=True):
            assert_deep_approx_equal(g, w)
        return

    # Direct comparison for all other types
    assert got == want


@pytest.mark.parametrize(
    ("expected", "actual", "want_identical", "want_result"),
    [
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [make_cluster_entity(2, "d1", ["1", "2"])],
            True,
            {},
            id="identical_sets",
        ),
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3", "4"]),
            ],
            [make_cluster_entity(3, "d1", ["2", "3"])],
            False,
            {
                "perfect": 0,
                "subset": 0,
                "superset": 0,
                "wrong": 1,
                "invalid": 0,
            },
            id="completely_different_sets",
        ),
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2", "3"])],
            [make_cluster_entity(2, "d1", ["1", "2"])],
            False,
            {
                "perfect": 0,
                "subset": 1,
                "superset": 0,
                "wrong": 0,
                "invalid": 0,
            },
            id="subset_match",
        ),
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3"]),
            ],
            [make_cluster_entity(3, "d1", ["1", "2", "3"])],
            False,
            {
                "perfect": 0,
                "subset": 0,
                "superset": 1,
                "wrong": 0,
                "invalid": 0,
            },
            id="superset_match",
        ),
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [make_cluster_entity(2, "d1", ["1", "2", "3", "4"])],
            False,
            {
                "perfect": 0,
                "subset": 0,
                "superset": 0,
                "wrong": 0,
                "invalid": 1,
            },
            id="invalid_entity",
        ),
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3", "4"]),
            ],
            [
                make_cluster_entity(3, "d1", ["1", "2"]),  # perfect match
                make_cluster_entity(4, "d1", ["3"]),  # subset
                make_cluster_entity(4, "d1", ["1", "2", "3"]),  # superset
                make_cluster_entity(5, "d1", ["2", "3"]),  # wrong
                make_cluster_entity(6, "d1", ["7", "8", "9"]),  # invalid
            ],
            False,
            {
                "perfect": 1,
                "subset": 1,
                "superset": 1,
                "wrong": 1,
                "invalid": 1,
            },
            id="mixed_scenario",
        ),
    ],
)
def test_diff_entities(
    expected: list[Cluster],
    actual: list[Cluster],
    want_identical: bool,
    want_result: dict[str, Any],
) -> None:
    """diff_entities classifies each cluster as perfect, subset, superset, or wrong."""
    got_identical, got_result = diff_entities(expected, actual)

    assert got_identical == want_identical
    assert dict(got_result) == want_result


def test_source_to_results_conversion() -> None:
    """A TrueEntity projects onto a chosen subset of its sources as a Cluster."""
    # Create a true entity present in multiple sources
    source = TrueEntity(
        base_values={"name": "Test"},
        keys=EntityReference(
            {"source1": frozenset({"1", "2"}), "source2": frozenset({"A", "B"})}
        ),
    )

    # Project different subsets onto clusters
    results1 = source.cluster("source1")
    results2 = source.cluster("source1", "source2")
    results3 = source.cluster("source2")

    # Test different comparison scenarios
    identical, report = diff_entities([results1], [results1])
    assert identical
    assert report == {}

    # Compare partial overlap
    identical, report = diff_entities([results1], [results2])
    assert not identical
    assert "source2" in str(results2 - results1)

    # Compare disjoint sets
    identical, report = diff_entities([results1], [results3])
    assert not identical
    assert results1.similarity_ratio(results3) == 0.0

    # Test missing source returns None
    assert source.cluster("nonexistent") is None


@pytest.mark.parametrize(
    ("base_generator", "expected_type"),
    [
        pytest.param("name", pl.String, id="text_generator"),
        pytest.param("random_int", pl.Int64, id="integer_generator"),
        pytest.param("date_this_decade", pl.Date, id="date_generator"),
    ],
)
def test_feature_config_datatype_inference(
    base_generator: str, expected_type: pl.DataType
) -> None:
    """A FeatureConfig's datatype follows from its base_generator."""
    feature_config = FeatureConfig(name=base_generator, base_generator=base_generator)
    assert feature_config.datatype == expected_type

"""Deterministic and weighted-deterministic linkers, checked against the testkit oracle.

Each linker is a `pytest.param` in `LINKERS`, so covering a new one is a new row, not
a new test body. Splink also appears here because `configure_splink_linker` pins its
match probabilities to certainty, making it behave deterministically for this suite.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

import polars as pl
import pytest
from splink import SettingsCreator
from splink import blocking_rule_library as brl
from splink import comparison_library as cl
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.comparison_creator import ComparisonCreator

from matchlab.models import Model
from matchlab.models.linkers.base import Linker
from matchlab.models.linkers.deterministic import (
    DeterministicLinker,
)
from matchlab.models.linkers.splinklinker import SplinkLinker
from matchlab.models.linkers.weighteddeterministic import (
    WeightedDeterministicLinker,
)
from matchlab.sources import Source
from matchlab.testkit.features import FeatureConfig, SourceParameters
from matchlab.testkit.linked import linked_sources_factory
from matchlab.testkit.sources import GeneratedSource, source_factory

LinkerConfigurator = Callable[[GeneratedSource, GeneratedSource], dict[str, Any]]

# Methodology configuration adapters


def configure_deterministic_linker(
    left_testkit: GeneratedSource, right_testkit: GeneratedSource
) -> dict[str, Any]:
    """Build validated `DeterministicLinker` comparing every shared field."""
    left_fields = {
        name for name in left_testkit.field_names if name not in ("key", "id")
    }
    right_fields = {
        name for name in right_testkit.field_names if name not in ("key", "id")
    }
    shared_fields: list[str] = sorted(left_fields & right_fields)

    if not shared_fields:
        raise ValueError("Must have at least one shared field")

    comparisons: list[str] = []
    for field in shared_fields:
        comparisons.append(f"l.{field} = r.{field}")

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "comparisons": comparisons,
    }

    DeterministicLinker.model_validate(settings_dict)

    return settings_dict


def configure_deterministic_linker_sequential(
    left_testkit: GeneratedSource, right_testkit: GeneratedSource
) -> dict[str, Any]:
    """Like `configure_deterministic_linker`, but each round holds one comparison.

    `DeterministicLinker` runs each round in sequence rather than all at once, so this
    checks that behaviour too, not just the single-round case.
    """
    left_fields = {
        name for name in left_testkit.field_names if name not in ("key", "id")
    }
    right_fields = {
        name for name in right_testkit.field_names if name not in ("key", "id")
    }
    shared_fields: list[str] = sorted(left_fields & right_fields)

    if not shared_fields:
        raise ValueError("Must have at least one shared field")

    comparisons: list[list[str]] = []
    for field in shared_fields:
        comparisons.append([f"l.{field} = r.{field}"])

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "comparisons": comparisons,
    }

    # More than one round, or this configurator isn't testing sequential rounds at all.
    assert len(comparisons) > 1

    DeterministicLinker.model_validate(settings_dict)

    return settings_dict


def configure_weighted_deterministic_linker(
    left_testkit: GeneratedSource, right_testkit: GeneratedSource
) -> dict[str, Any]:
    """Build validated `WeightedDeterministicLinker` with equal weight per field."""
    left_fields = {
        name for name in left_testkit.field_names if name not in ("key", "id")
    }
    right_fields = {
        name for name in right_testkit.field_names if name not in ("key", "id")
    }
    shared_fields: list[str] = sorted(left_fields & right_fields)

    if not shared_fields:
        raise ValueError("Must have at least one shared field")

    weighted_comparisons = []
    for field in shared_fields:
        weighted_comparisons.append(
            {"comparison": f"l.{field} = r.{field}", "weight": 1}
        )

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "weighted_comparisons": weighted_comparisons,
        "threshold": 1,  # Require all comparisons to match
    }

    WeightedDeterministicLinker.model_validate(settings_dict)

    return settings_dict


def configure_splink_linker(
    left_testkit: GeneratedSource, right_testkit: GeneratedSource
) -> dict[str, Any]:
    """Build validated `SplinkLinker`, pinned to certainty for deterministic runs."""
    left_fields = {
        name for name in left_testkit.field_names if name not in ("key", "id")
    }
    right_fields = {
        name for name in right_testkit.field_names if name not in ("key", "id")
    }
    shared_fields: list[str] = sorted(left_fields & right_fields)

    if not shared_fields:
        raise ValueError("Must have at least one shared field")

    deterministic_matching_rules: list[str] = []
    blocking_rules_to_generate_predictions: list[BlockingRuleCreator] = []
    comparisons: list[ComparisonCreator] = []

    for field in shared_fields:
        deterministic_matching_rules.append(f"l.{field} = r.{field}")
        blocking_rules_to_generate_predictions.append(brl.block_on(field))
        comparisons.append(cl.ExactMatch(field).configure(m_probabilities=[1, 0]))

    linker_training_functions = [
        {
            "function": "estimate_probability_two_random_records_match",
            "arguments": {
                "deterministic_matching_rules": deterministic_matching_rules,
                "recall": 1,
            },
        },
        {
            "function": "estimate_u_using_random_sampling",
            "arguments": {"max_pairs": 1e4},
        },
    ]

    # m is pinned to 1 rather than estimated. Estimating it needs expectation
    # maximisation, which many of these tests can't run, since several have only one
    # shared field. A pinned m is fine here, since these tests only check raw
    # match/no-match behaviour, not a calibrated probability.
    linker_settings = SettingsCreator(
        link_type="link_only",
        retain_matching_columns=False,
        retain_intermediate_calculation_columns=False,
        blocking_rules_to_generate_predictions=blocking_rules_to_generate_predictions,
        comparisons=comparisons,
    )

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "linker_training_functions": linker_training_functions,
        "linker_settings": linker_settings,
        "threshold": None,
    }

    SplinkLinker.model_validate(settings_dict)

    return settings_dict


LINKERS = [
    pytest.param(
        DeterministicLinker, configure_deterministic_linker, id="Deterministic"
    ),
    pytest.param(
        DeterministicLinker,
        configure_deterministic_linker_sequential,
        id="DeterministicSequential",
    ),
    pytest.param(
        WeightedDeterministicLinker,
        configure_weighted_deterministic_linker,
        id="WeightedDeterministic",
    ),
    pytest.param(SplinkLinker, configure_splink_linker, id="Splink"),
    # Add more linker classes and configuration functions here
]

# Test cases


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
@patch.object(Source, "_read_cache")
def test_link_exact_match(
    mock_query_run: Mock, Linker: Linker, configure_linker: LinkerConfigurator
) -> None:
    """A linker recovers every match when both sources hold identical field values."""
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
        ),
    )

    configs = (
        SourceParameters(
            name="source_left",
            features=features,
            n_true_entities=10,
        ),
        SourceParameters(
            name="source_right",
            features=features,
            n_true_entities=10,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)
    linked.write_to_location()
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    # The side_effect order matches the call order, so left is queried before right.
    mock_query_run.side_effect = [
        pl.from_arrow(left_source.data),
        pl.from_arrow(right_source.data),
    ]

    assert left_source.data.select(["company", "email"]).equals(
        right_source.data.select(["company", "email"])
    )

    linker = Model(
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left=left_source.source,
        right=right_source.source,
    )
    results = linker.collect().edges()

    identical, report = linked.diff_model_edges(
        results, left=left_source, right=right_source
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
@patch.object(Source, "_read_cache")
def test_link_exact_with_duplicates(
    mock_query_run: Mock, Linker: Linker, configure_linker: LinkerConfigurator
) -> None:
    """A linker still recovers every match when both sources repeat each entity."""
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
        ),
    )

    configs = (
        SourceParameters(
            name="source_left",
            features=features,
            n_true_entities=10,
            repetition=1,  # Each entity appears twice
        ),
        SourceParameters(
            name="source_right",
            features=features,
            n_true_entities=10,
            repetition=3,  # Each entity appears four times
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)
    linked.write_to_location()
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    mock_query_run.side_effect = [
        pl.from_arrow(left_source.data),
        pl.from_arrow(right_source.data),
    ]

    linker = Model(
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left=left_source.source,
        right=right_source.source,
    )
    results = linker.collect().edges()

    identical, report = linked.diff_model_edges(
        results, left=left_source, right=right_source
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
@patch.object(Source, "_read_cache")
def test_link_partial_overlap(
    mock_query_run: Mock, Linker: Linker, configure_linker: LinkerConfigurator
) -> None:
    """A linker matches only the entities the right source actually has.

    The left source has every entity, but the right source has only half. A linker
    must not invent matches for the left source's extra entities.
    """
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="registration_id",
            base_generator="numerify",
            parameters=(("text", "######"),),
        ),
    )

    configs = (
        SourceParameters(
            name="source_left",
            features=features,
            n_true_entities=10,
        ),
        SourceParameters(
            name="source_right",
            features=features,
            n_true_entities=5,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)
    linked.write_to_location()
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    mock_query_run.side_effect = [
        pl.from_arrow(left_source.data),
        pl.from_arrow(right_source.data),
    ]

    linker = Model(
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left=left_source.source,
        right=right_source.source,
    )
    results = linker.collect().edges()

    identical, report = linked.diff_model_edges(
        results, left=left_source, right=right_source
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
@patch.object(Source, "_read_cache")
def test_link_no_matches(
    mock_query_run: Mock, Linker: Linker, configure_linker: LinkerConfigurator
) -> None:
    """A linker produces no matches when the two sources share no entities at all."""
    features = (
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="identifier", base_generator="uuid4"),
    )

    configs = (
        SourceParameters(
            name="source_left",
            features=features,
            n_true_entities=10,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=314)
    linked.write_to_location()
    left_source = linked.sources["source_left"]
    right_source = source_factory(
        name="source_right", features=features, n_true_entities=10, seed=159
    )
    right_source.write_to_location()

    mock_query_run.side_effect = [
        pl.from_arrow(left_source.data),
        pl.from_arrow(right_source.data),
    ]

    for column in ("company", "identifier"):
        l_col = set(left_source.data[column].to_pylist())
        r_col = set(right_source.data[column].to_pylist())
        assert l_col.isdisjoint(r_col)

    linker = Model(
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left=left_source.source,
        right=right_source.source,
    )
    results = linker.collect().edges()

    identical, report = linked.diff_model_edges(
        results, left=left_source, right=right_source
    )

    assert not identical
    assert len(results) == 0


@pytest.mark.parametrize(
    ("linker_class", "settings"),
    [
        pytest.param(
            DeterministicLinker,
            {"comparisons": [["l.key = r.key"], ["l.fallback = r.fallback"]]},
            id="sequential",
        ),
        pytest.param(
            WeightedDeterministicLinker,
            {
                "weighted_comparisons": [
                    {"comparison": "l.key = r.key", "weight": 1},
                ],
                "threshold": 1,
            },
            id="weighted",
        ),
    ],
)
@pytest.mark.parametrize(
    ("affected_side", "expected"),
    [
        pytest.param("left", {(3, 10)}, id="left"),
        pytest.param("right", {(1, 30)}, id="right"),
        pytest.param("both", {(3, 10), (1, 30), (3, 30)}, id="both"),
    ],
)
def test_affected_pairs(
    linker_class: type[Linker],
    settings: dict[str, Any],
    affected_side: str,
    expected: set[tuple[int, int]],
) -> None:
    """Preview includes every affected category, but no baseline-only pairs."""
    linker = linker_class.model_validate(settings)
    baseline_left = pl.DataFrame(
        {"id": [1, 2], "key": ["a", "b"], "fallback": ["a", "b"]}
    )
    baseline_right = pl.DataFrame(
        {"id": [10, 20], "key": ["a", "b"], "fallback": ["a", "b"]}
    )
    affected_left = pl.DataFrame({"id": [3], "key": ["a"], "fallback": ["a"]})
    affected_right = pl.DataFrame({"id": [30], "key": ["a"], "fallback": ["a"]})
    linker.prepare(baseline_left, baseline_right)

    kwargs = {}
    if affected_side in ("left", "both"):
        kwargs["left"] = affected_left
    if affected_side in ("right", "both"):
        kwargs["right"] = affected_right
    results = linker.link(**kwargs)

    pairs = set(results.select("left_id", "right_id").iter_rows())
    assert pairs == expected
    assert (2, 20) not in pairs
    assert results.schema == {
        "left_id": pl.Int64,
        "right_id": pl.Int64,
        "score": pl.Float32,
    }
    assert set(results["score"].to_list()) == {1.0}
    assert len(results) == len(expected)


@pytest.mark.parametrize(
    ("linker_class", "settings"),
    [
        pytest.param(
            DeterministicLinker,
            {"comparisons": ["l.key = r.key"]},
            id="deterministic",
        ),
        pytest.param(
            WeightedDeterministicLinker,
            {
                "weighted_comparisons": [{"comparison": "l.key = r.key", "weight": 1}],
                "threshold": 1,
            },
            id="weighted",
        ),
    ],
)
def test_affected_rejects_missing_inputs(
    linker_class: type[Linker], settings: dict[str, Any]
) -> None:
    """Linking requires preparation and at least one affected side."""
    linker = linker_class.model_validate(settings)
    data = pl.DataFrame({"id": [1], "key": ["a"]})

    with pytest.raises(RuntimeError, match="prepare"):
        linker.link(left=data)

    linker.prepare(data, data)
    with pytest.raises(ValueError, match="affected side"):
        linker.link()


@pytest.mark.parametrize(
    ("linker_class", "settings"),
    [
        pytest.param(
            DeterministicLinker,
            {"comparisons": ["l.key = r.key"]},
            id="deterministic",
        ),
        pytest.param(
            WeightedDeterministicLinker,
            {
                "weighted_comparisons": [{"comparison": "l.key = r.key", "weight": 1}],
                "threshold": 1,
            },
            id="weighted",
        ),
    ],
)
def test_affected_prepared_baseline(
    linker_class: type[Linker], settings: dict[str, Any]
) -> None:
    """Affected calls do not change later full-baseline collection results."""
    linker = linker_class.model_validate(settings)
    baseline_left = pl.DataFrame(
        {"id": [1, 1, 2], "key": ["a", "c", "b"], "note": [None, None, None]}
    )
    baseline_right = pl.DataFrame(
        {"id": [10, 20, 30], "key": ["a", "b", "c"], "note": [None, None, None]}
    )
    linker.prepare(baseline_left, baseline_right)

    linker.link(
        left=pl.DataFrame({"id": [3], "key": ["a"], "note": [None]}),
        right=pl.DataFrame({"id": [40], "key": ["a"], "note": [None]}),
    )
    results = linker.link(left=baseline_left, right=baseline_right)

    assert set(results.iter_rows()) == {
        (1, 10, 1.0),
        (1, 30, 1.0),
        (2, 20, 1.0),
    }
    assert len(results) == 3


@pytest.mark.parametrize(
    ("linker_class", "settings"),
    [
        pytest.param(
            DeterministicLinker,
            {"comparisons": ["l.key = r.key"]},
            id="deterministic",
        ),
        pytest.param(
            WeightedDeterministicLinker,
            {
                "weighted_comparisons": [{"comparison": "l.key = r.key", "weight": 1}],
                "threshold": 1,
            },
            id="weighted",
        ),
    ],
)
def test_affected_same_id_rows(
    linker_class: type[Linker], settings: dict[str, Any]
) -> None:
    """Adding one variant of an ID keeps its other baseline variants available."""
    linker = linker_class.model_validate(settings)
    linker.prepare(
        pl.DataFrame({"id": [1, 2], "key": ["old", "baseline"]}),
        pl.DataFrame({"id": [10, 20, 30], "key": ["old", "new", "baseline"]}),
    )

    results = linker.link(
        left=pl.DataFrame({"id": [1], "key": ["new"]}),
        right=pl.DataFrame({"id": [40], "key": ["old"]}),
    )

    assert set(results.iter_rows()) == {(1, 20, 1.0), (1, 40, 1.0)}
    assert len(results) == 2


def test_affected_sequential_rounds() -> None:
    """An early match removes its IDs before the next comparison runs."""
    linker = DeterministicLinker(
        comparisons=[["l.key = r.key"], ["l.fallback = r.fallback"]]
    )
    baseline_left = pl.DataFrame(
        {"id": [1, 2], "key": ["a", "b"], "fallback": ["z", "z"]}
    )
    baseline_right = pl.DataFrame(
        {"id": [10, 20], "key": ["a", "other"], "fallback": ["z", "z"]}
    )
    linker.prepare(baseline_left, baseline_right)

    results = linker.link(left=baseline_left)

    assert set(results.select("left_id", "right_id").iter_rows()) == {
        (1, 10),
        (2, 20),
    }


def test_affected_both_sides_share_sequential_rounds() -> None:
    """First-round matches remove IDs before cross-side fallback matching."""
    linker = DeterministicLinker(
        comparisons=[["l.key = r.key"], ["l.fallback = r.fallback"]]
    )
    linker.prepare(
        pl.DataFrame({"id": [1], "key": ["x"], "fallback": ["z"]}),
        pl.DataFrame({"id": [10], "key": ["a"], "fallback": ["x"]}),
    )

    results = linker.link(
        left=pl.DataFrame({"id": [3], "key": ["a"], "fallback": ["z"]}),
        right=pl.DataFrame({"id": [30], "key": ["b"], "fallback": ["z"]}),
    )

    assert set(results.select("left_id", "right_id").iter_rows()) == {
        (3, 10),
        (1, 30),
    }


def test_affected_weighted_score_is_not_counted_twice() -> None:
    """Every pair's conditions contribute once, including two-sided additions."""
    linker = WeightedDeterministicLinker(
        weighted_comparisons=[
            {"comparison": "l.key = r.key", "weight": 2},
            {"comparison": "l.fallback = r.fallback", "weight": 1},
        ],
        threshold=1,
    )
    linker.prepare(
        pl.DataFrame({"id": [1, 2], "key": ["a", "b"], "fallback": ["x", "y"]}),
        pl.DataFrame({"id": [10, 20], "key": ["a", "b"], "fallback": ["x", "y"]}),
    )

    results = linker.link(
        left=pl.DataFrame({"id": [3], "key": ["a"], "fallback": ["x"]}),
        right=pl.DataFrame({"id": [30], "key": ["a"], "fallback": ["x"]}),
    )

    assert set(results.iter_rows()) == {
        (3, 10, 1.0),
        (1, 30, 1.0),
        (3, 30, 1.0),
    }


def test_affected_empty() -> None:
    """No candidate pair leaves the prepared side's IDs in the output schema."""
    linker = DeterministicLinker(comparisons=["l.key = r.key"])
    left = pl.DataFrame({"id": [1], "key": ["a"]})
    right = pl.DataFrame({"id": [10], "key": ["a"]})
    linker.prepare(left, right)

    results = linker.link(left=left.clear())

    assert results.is_empty()
    assert results.schema == {
        "left_id": pl.Int64,
        "right_id": pl.Int64,
        "score": pl.Float32,
    }

"""Deterministic dedupers, checked against the testkit oracle.

Each deduper is a `pytest.param` in `DEDUPERS`, so covering a new one is a new row, not
a new test body. The oracle plants the answer. `diff_model_edges` scores the edges the
methodology emits against it.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

import polars as pl
import pytest

from matchlab.models import Model
from matchlab.models.dedupers.base import Deduper
from matchlab.models.dedupers.naive import NaiveDeduper
from matchlab.sources import Source
from matchlab.testkit.features import FeatureConfig, SourceParameters
from matchlab.testkit.linked import linked_sources_factory
from matchlab.testkit.sources import GeneratedSource

DeduperConfigurator = Callable[[GeneratedSource], dict[str, Any]]


def configure_naive_deduper(testkit: GeneratedSource) -> dict[str, Any]:
    """Build validated `NaiveDeduper` settings over every field but `key` and `id`."""
    fields = [name for name in testkit.field_names if name not in ("key", "id")]
    settings_dict = {"id": "id", "unique_fields": fields}
    NaiveDeduper.model_validate(settings_dict)
    return settings_dict


DEDUPERS = [
    pytest.param(NaiveDeduper, configure_naive_deduper, id="Naive"),
    # Add more deduper classes and their configurators here.
]


@pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
@pytest.mark.parametrize(
    "repetition",
    [
        pytest.param(0, id="no-duplicates"),
        pytest.param(2, id="exact-duplicates"),
    ],
)
@patch.object(Source, "_read_cache")
def test_recovers_planted(
    mock_read_cache: Mock,
    repetition: int,
    Deduper: Deduper,
    configure_deduper: DeduperConfigurator,
) -> None:
    """A deterministic deduper recovers the planted entities, duplicates or not.

    `repetition=0` plants each entity once, so there is nothing to merge.
    `repetition=2` plants it three times over. The oracle knows the answer either way,
    so the same body covers both. The edges must diff clean against it.
    """
    features = (
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="email", base_generator="email"),
    )
    source_parameters = SourceParameters(
        name="source_exact",
        features=features,
        n_true_entities=10,
        repetition=repetition,
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    linked.write_to_location()
    source = linked.sources["source_exact"]

    # Mock the record step read so the deduper scores the generated rows directly.
    mock_read_cache.return_value = pl.from_arrow(source.data)

    deduper = Model(
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left=source.source,
    )
    results = deduper.collect().edges()

    identical, report = linked.diff_model_edges(results, left=source)
    assert identical, report


def test_affected_pairs() -> None:
    """Only pairs touching an affected record appear, once per unordered ID pair."""
    deduper = NaiveDeduper(unique_fields=["key"])
    baseline = pl.DataFrame({"id": [1, 2, 4, 5], "key": ["a", "a", "b", "b"]})
    affected = pl.DataFrame({"id": [3, 3, 6, 7, 8], "key": ["a", "a", "a", "c", "c"]})
    prepared = deduper.prepare(baseline)

    results = prepared.dedupe(affected, baseline=baseline)

    pairs = {
        frozenset((row["left_id"], row["right_id"])) for row in results.rows(named=True)
    }
    assert pairs == {
        frozenset(pair) for pair in [(1, 3), (1, 6), (2, 3), (2, 6), (3, 6), (7, 8)]
    }
    assert len(results) == len(pairs)
    assert results.schema == {
        "left_id": pl.Int64,
        "right_id": pl.Int64,
        "score": pl.Float32,
    }
    assert set(results["score"].to_list()) == {1.0}


def test_affected_empty() -> None:
    """An empty affected input emits no baseline-only pairs."""
    baseline = pl.DataFrame({"id": [1, 2], "key": ["a", "a"]})
    deduper = NaiveDeduper(unique_fields=["key"])
    prepared = deduper.prepare(baseline)

    results = prepared.dedupe(baseline.clear(), baseline=baseline)

    assert results.is_empty()
    assert results.schema == {
        "left_id": pl.Int64,
        "right_id": pl.Int64,
        "score": pl.Float32,
    }

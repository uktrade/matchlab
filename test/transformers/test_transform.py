"""The `Transform` plan node, covering desugaring, construction, and the registry.

These assert on plan *shape*, so they build over a real `Source` from the `source`
fixture but never collect. The identities hold before any warehouse read.
"""

from collections.abc import Callable
from typing import ClassVar
from unittest.mock import patch

import polars as pl
import pytest
from pydantic import ValidationError

from matchlab import Source
from matchlab.stores import DuckDBStore
from matchlab.transformers import (
    Clean,
    Group,
    Select,
    Transform,
    Transformer,
    add_transformer_class,
)


def test_select_desugars_to_transform(source: Callable[..., Source]) -> None:
    """`source.select(...)` is a transform carrying a `Select`."""
    crn = source("crn")
    step = crn.select("crn_company", "crn_town")

    assert isinstance(step, Transform)
    assert isinstance(step.transformer, Select)
    assert step.transformer.columns == ("crn_company", "crn_town")
    assert step.parents == (crn,)


@pytest.mark.parametrize(
    ("verb", "explicit"),
    [
        pytest.param(
            lambda crn: crn.clean({"name": "lower(crn_company)"}),
            lambda crn: crn.transform(Clean(cleaning={"name": "lower(crn_company)"})),
            id="clean",
        ),
        pytest.param(
            lambda crn: crn.select("crn_company"),
            lambda crn: crn.transform(Select("crn_company")),
            id="select",
        ),
    ],
)
def test_verb_equals_transform_of_transformer(
    source: Callable[..., Source],
    verb: Callable[[Source], Transform],
    explicit: Callable[[Source], Transform],
) -> None:
    """The convenience verb is exactly `transform()` of the matching transformer."""
    crn = source("crn")
    assert verb(crn).spec == explicit(crn).spec


def test_transform_builds_from_name_and_dict(source: Callable[..., Source]) -> None:
    """A name plus a settings dict rebuilds the same transformer as an instance.

    This is the path `document.load` takes when reconstructing a plan.
    """
    crn = source("crn")
    from_dict = Transform(crn, "Clean", {"cleaning": {"name": "lower(crn_company)"}})
    from_instance = Transform(crn, Clean(cleaning={"name": "lower(crn_company)"}))

    assert from_dict.transformer == from_instance.transformer
    assert from_dict.spec == from_instance.spec


class _Double(Transformer):
    """A custom transformer that doubles one integer column, for the registry test."""

    column: str

    def apply(
        self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
    ) -> pl.DataFrame:
        return data.with_columns((pl.col(self.column) * 2).alias(self.column))


def test_add_transformer_class_makes_it_nameable(
    source: Callable[..., Source],
) -> None:
    """A registered custom transformer can be named in a plan, as a built-in can."""
    add_transformer_class(_Double)
    crn = source("crn")

    step = Transform(crn, "_Double", {"column": "n"})

    assert isinstance(step.transformer, _Double)
    assert step.transformer.prepare(pl.DataFrame({"n": [1, 2]})).apply(
        pl.DataFrame({"n": [1, 2]}), baseline=pl.DataFrame({"n": [1, 2]})
    )["n"].to_list() == [2, 4]


def test_add_transformer_class_rejects_non_transformer() -> None:
    """The registry only accepts `Transformer` subclasses."""
    with pytest.raises(ValueError, match="not a subclass of Transformer"):
        add_transformer_class(str)


class _Counting(Transformer):
    """An identity transformer for runtime preparation lifecycle tests."""

    def apply(
        self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
    ) -> pl.DataFrame:
        return data


class _CachedCounting(_Counting):
    """The same transformer with a stable collection fingerprint."""

    version: ClassVar[int] = 1


def test_transformer_instance_isolated(source: Callable[..., Source]) -> None:
    """Each step starts without preparation even when given the same specification."""
    transformer = _Counting()
    prepared = transformer.prepare(pl.DataFrame())
    crn = source("crn")
    first = crn.transform(transformer)
    second = crn.transform(transformer)

    assert first.transformer is not transformer
    assert first.transformer is not second.transformer
    assert first._prepared is second._prepared is None
    first.collect()
    assert first._prepared is not None
    assert first._prepared is not prepared
    assert second._prepared is None


def test_prepare_cache_retry(source: Callable[..., Source]) -> None:
    """A failed lazy preparation can retry against the same collected baseline."""
    crn = source("crn")
    crn.transform(_CachedCounting()).collect()
    cached = crn.transform(_CachedCounting()).collect()
    prepared = cached.transformer.prepare(crn._read_cache(cached._require_store()))

    with patch.object(
        _CachedCounting,
        "prepare",
        side_effect=[ValueError("baseline unavailable"), prepared],
    ) as prepare:
        with pytest.raises(ValueError, match="baseline unavailable"):
            cached._ensure_prepared()
        assert cached._prepared is None
        assert cached._ensure_prepared() is prepared
        assert cached._ensure_prepared() is prepared
        assert prepare.call_count == 2


def test_prepare_refresh(source: Callable[..., Source]) -> None:
    """Refreshing an unversioned transform replaces its prepared object."""
    step = source("crn").transform(_Counting())
    step.collect()
    first = step._prepared
    step.collect()

    assert step._prepared is not None
    assert step._prepared is not first


def test_prepare_invalidated_on_different_cached_output(
    source: Callable[..., Source],
) -> None:
    """Adopting another cached fingerprint drops the previous runtime preparation."""
    crn = source("crn")
    first = crn.transform(_Counting()).collect()
    second = crn.transform(_Counting()).collect()
    assert first._fp != second._fp
    assert first._prepared is not None
    with patch.object(Transform, "_fingerprint", return_value=second._fp):
        first.collect()
    assert first._fp == second._fp
    assert first._prepared is None
    assert first._ensure_prepared() is not second._prepared


def test_prepare_uncollected(source: Callable[..., Source]) -> None:
    """Runtime preparation requires a collected baseline."""
    step = source("crn").transform(_Counting())
    with pytest.raises(RuntimeError, match="Call collect"):
        step._ensure_prepared()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            {"up": "upper(crn_company)", "tag": "up || '!'"},
            {"tag": "up || '!'", "up": "upper(crn_company)"},
            id="cleaning",
        ),
        pytest.param(
            {"n": "count(*)", "double": "n * 2"},
            {"double": "n * 2", "n": "count(*)"},
            id="aggregates",
        ),
    ],
)
def test_spec_key_reordered(
    source: Callable[..., Source],
    first: dict[str, str],
    second: dict[str, str],
) -> None:
    """Entry order is part of a `Clean` or a `Group`, so it belongs in the spec key.

    Both project their entries in the order they were written, and DuckDB resolves an
    alias only against entries defined before it. These two orders are therefore
    different transforms, and the second cannot run at all. Hashing the settings with
    sorted keys gave them one fingerprint, so the second read back the first's table.
    """
    crn = source("crn")
    verb = crn.clean if "up" in first else crn.group

    assert verb(first)._spec_key() != verb(second)._spec_key()


# -- `id` is not writable -------------------------------------------------------------
#
# `id` is the grouping a model matches on, derived by matchlab from record content.
# A transformer that writes one changes which records count as the same, and the
# fingerprint covers the expression rather than what the expression displaced, so
# nothing downstream could notice.


@pytest.mark.parametrize(
    ("transformer", "settings"),
    [
        pytest.param(Clean, {"cleaning": {"id": "crn_company"}}, id="clean"),
        pytest.param(
            Group, {"aggregates": {"id": "any_value(crn_company)"}}, id="group"
        ),
    ],
)
def test_id_output_refused(transformer: type, settings: dict) -> None:
    """Refused when the transformer is built, before a plan can be collected."""
    with pytest.raises(ValidationError, match="`id` is not a column you can write"):
        transformer(**settings)


def test_custom_transformer_id_replaced(
    source: Callable[..., Source], store: DuckDBStore
) -> None:
    """The backstop for a transformer matchlab did not write.

    A custom one bypasses the checks above, so `Transform._execute` verifies `id`
    survived rather than storing a record step whose grouping is somebody's data.
    """

    class ClobbersId(Transformer):
        def apply(
            self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
        ) -> pl.DataFrame:
            return data.with_columns(pl.col("crn_company").alias("id"))

    step = source("crn").transform(ClobbersId())
    with pytest.raises(ValueError, match="ClobbersId replaced the `id` column"):
        step.collect(store)


def test_custom_transformer_id_drop(
    source: Callable[..., Source], store: DuckDBStore
) -> None:
    """Losing `id` altogether is the same failure, one step earlier."""

    class DropsId(Transformer):
        def apply(
            self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
        ) -> pl.DataFrame:
            return data.drop("id")

    step = source("crn").transform(DropsId())
    with pytest.raises(ValueError, match="DropsId dropped the `id` column"):
        step.collect(store)

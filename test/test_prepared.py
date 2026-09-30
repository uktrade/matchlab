"""Preparation belongs to independent runtime objects, never methodology specs."""

from collections.abc import Callable
from typing import ClassVar
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError

import matchlab as mb
from matchlab.models.models import _MODEL_CLASSES
from matchlab.stores import DuckDBStore

Prepared = (
    mb.PreparedTransformer
    | mb.PreparedDeduper
    | mb.PreparedLinker
    | mb.PreparedResolverMethod
)
Methodology = mb.Transformer | mb.Deduper | mb.Linker | mb.ResolverMethod
PreparedStep = mb.Transform | mb.Model | mb.Resolver


class Centred(mb.Transformer):
    """Centre supplied values using a mean learned from the full baseline."""

    version: ClassVar[int] = 1
    column: str

    def prepare(self, data: pl.DataFrame) -> mb.PreparedTransformer:
        """Return the mean without retaining data or changing this specification."""
        return mb.PreparedTransformer(methodology=self, state=data[self.column].mean())

    def apply(
        self, prepared_state: float, data: pl.DataFrame, *, baseline: pl.DataFrame
    ) -> pl.DataFrame:
        """Apply the baseline's mean to supplied values."""
        return data.with_columns(
            (pl.col(self.column) - prepared_state).alias(self.column)
        )


mb.add_transformer_class(Centred)


def test_preparations_are_independent() -> None:
    """Preparing another baseline cannot change the first object's learned mean."""
    methodology = Centred(column="value")
    settings = methodology.model_dump_json()
    baseline_a = pl.DataFrame({"id": [1, 2], "value": [2.0, 4.0]})
    baseline_b = pl.DataFrame({"id": [3, 4], "value": [10.0, 20.0]})
    before_a, before_b = baseline_a.clone(), baseline_b.clone()
    supplied = pl.DataFrame({"id": [5], "value": [8.0]})
    first = methodology.prepare(baseline_a)
    expected = first.apply(supplied, baseline=baseline_a)
    second = methodology.prepare(baseline_b)

    for prepared, baseline, value in [
        (first, baseline_a, 5.0),
        (second, baseline_b, -7.0),
        (first, baseline_a, 5.0),
    ]:
        assert prepared.apply(supplied, baseline=baseline)["value"].to_list() == [value]
    assert_frame_equal(first.apply(supplied, baseline=baseline_a), expected)
    assert_frame_equal(baseline_a, before_a)
    assert_frame_equal(baseline_b, before_b)
    assert methodology.model_dump_json() == settings
    assert not methodology.__pydantic_private__
    assert first.methodology is second.methodology is methodology
    assert first is not second


@pytest.mark.parametrize(
    ("methodology", "prepared_type", "verb", "kwargs"),
    [
        pytest.param(
            mb.Select("value"),
            mb.PreparedTransformer,
            "apply",
            {"data": pl.DataFrame(), "baseline": pl.DataFrame()},
            id="transformer",
        ),
        pytest.param(
            mb.NaiveDeduper(unique_fields=["value"]),
            mb.PreparedDeduper,
            "dedupe",
            {"data": pl.DataFrame(), "baseline": pl.DataFrame()},
            id="deduper",
        ),
        pytest.param(
            mb.DeterministicLinker(comparisons=["l.value = r.value"]),
            mb.PreparedLinker,
            "link",
            {
                "left": pl.DataFrame(),
                "right": None,
                "baseline_left": pl.DataFrame(),
                "baseline_right": pl.DataFrame(),
            },
            id="linker",
        ),
        pytest.param(
            mb.Components(),
            mb.PreparedResolverMethod,
            "compute_clusters",
            {"model_edges": {}, "baseline_model_edges": {}},
            id="resolver",
        ),
    ],
)
def test_prepared_family_delegates(
    methodology: Methodology, prepared_type: type[Prepared], verb: str, kwargs: dict
) -> None:
    """Public prepared verbs pass opaque state and explicit inputs to their family."""
    state = object()
    prepared = prepared_type.model_validate(
        {"methodology": methodology, "state": state}
    )
    output = pl.DataFrame({"result": [1]})
    with patch.object(type(methodology), verb, return_value=output) as action:
        assert getattr(prepared, verb)(**kwargs) is output
    expected_args = [state]
    input_names = {
        "link": ["left", "right"],
        "dedupe": ["data"],
        "apply": ["data"],
        "compute_clusters": ["model_edges"],
    }[verb]
    for name in input_names:
        expected_args.append(kwargs[name])
    action.assert_called_once_with(
        *expected_args, **{k: v for k, v in kwargs.items() if k not in input_names}
    )
    assert prepared.methodology is methodology
    assert prepared.state is state
    with pytest.raises(ValidationError, match="frozen"):
        prepared.state = None  # ty: ignore[invalid-assignment]


def test_prepared_types_are_not_methodologies() -> None:
    """Runtime wrappers must never be selectable as methodology registry names."""
    assert (
        not {
            "PreparedDeduper",
            "PreparedLinker",
            "PreparedTransformer",
            "PreparedResolverMethod",
        }
        & _MODEL_CLASSES.keys()
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda src: src.select(src.f("company")), id="transform"),
        pytest.param(
            lambda src: src.dedupe(
                mb.NaiveDeduper, {"unique_fields": [src.f("company")]}
            ),
            id="dedupe",
        ),
        pytest.param(
            lambda src: src.link(
                src,
                mb.DeterministicLinker,
                {"comparisons": [f"l.{src.f('company')} = r.{src.f('company')}"]},
            ),
            id="link",
        ),
        pytest.param(
            lambda src: src.dedupe(
                mb.NaiveDeduper, {"unique_fields": [src.f("company")]}
            ).resolve(),
            id="resolver",
        ),
    ],
)
def test_step_preparation_is_lazy_and_ephemeral(
    source: Callable[..., mb.Source],
    store: DuckDBStore,
    build: Callable[[mb.Source], PreparedStep],
) -> None:
    """Cache reads neither prepare nor persist runtime state or change the plan."""
    src = source("crn")
    first = build(src)
    document = mb.dump(first)
    first.collect()
    prepared = first._prepared
    assert prepared is not None
    assert mb.dump(first) == document
    methodology = prepared.methodology
    settings = methodology.model_dump_json()
    with patch.object(
        type(methodology), "prepare", side_effect=AssertionError("prepared again")
    ):
        first.collect()
    assert first._prepared is prepared
    rebuilt = build(src)
    original_prepare = type(methodology).prepare
    with patch.object(
        type(methodology),
        "prepare",
        autospec=True,
        side_effect=original_prepare,
    ) as prepare:
        rebuilt.collect()
        prepare.assert_not_called()
        assert rebuilt._prepared is None
        fingerprint = rebuilt._fp
        prepare.side_effect = ValueError("baseline unavailable")
        with pytest.raises(ValueError, match="baseline unavailable"):
            rebuilt._ensure_prepared()
        assert rebuilt._prepared is None
        prepare.reset_mock()
        prepare.side_effect = original_prepare
        with (
            patch.object(store, "store_transform", side_effect=AssertionError("write")),
            patch.object(store, "store_model", side_effect=AssertionError("write")),
            patch.object(store, "store_resolver", side_effect=AssertionError("write")),
        ):
            runtime = rebuilt._ensure_prepared()
            assert rebuilt._ensure_prepared() is runtime
        assert prepare.call_count == 1
    assert runtime is not prepared
    assert rebuilt._fp == fingerprint == first._fp
    assert mb.dump(rebuilt) == document
    assert methodology.model_dump_json() == settings
    assert not methodology.__pydantic_private__


def test_preparation_does_not_enter_plan_document(
    source: Callable[..., mb.Source],
) -> None:
    """A custom methodology's learned state remains absent when a plan is rebuilt."""
    plan = source("crn").transform(Centred(column="value"))
    before = mb.dump(plan)
    key = plan._spec_key()
    prepared = plan.transformer.prepare(pl.DataFrame({"value": [2.0, 4.0]}))
    plan._prepared = prepared
    assert mb.dump(plan) == before
    assert plan._spec_key() == key
    rebuilt = mb.load(
        before,
        resources={
            resource.name: resource.value
            for resource in plan._input._resources().values()
            if resource.name is not None
        },
    )
    assert rebuilt._prepared is None

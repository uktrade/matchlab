"""End-to-end tests for the plan API, over a real SQLite warehouse.

Covers the whole plan surface: building a plan with no DAG, laziness, collect with
plan-fingerprint caching, record step storage, lineage navigation, GC, and the terminal
reads (`get_lookup`, `lookup_key`).

Scenario: a dedupe feeding a cross-source link.

    crn: a1=(acme,london) a2=(acme,leeds) a3=(beta,hull)
    dh:  b1=(acme,bristol) b2=(gamma,york)

a1/a2 differ on town, so they are distinct leaves and the deduper does real work.
The apex is a *link*, yet the dedupe grouping of {a1,a2} must survive through it, and
a3/b2, reachable but matched by nothing, must stay singletons (merge-forward).
"""

import json
from collections.abc import Callable, Iterator
from typing import ClassVar
from unittest.mock import patch

import polars as pl
import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, create_engine, text

import matchlab as mb
from matchlab import lineage
from matchlab.core.exceptions import ResourceError, StepNotFound
from matchlab.core.kinds import StepKind
from matchlab.specs import ModelType, SourceSpec

# `warehouse`, `store` and the `source` factory come from `test/conftest.py`. The
# module-level `_apex`/`_fan_out_plan` builders take that factory, so this file holds no
# source-construction of its own beyond the few tests that need a bespoke shape.


def _dedupe_crn(crn: mb.Source) -> mb.Resolver:
    return crn.dedupe(
        model_class=mb.NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()


@pytest.mark.parametrize(
    ("build",),
    [
        pytest.param(lambda crn: crn.select(crn.f("company")), id="transform"),
        pytest.param(
            lambda crn: crn.dedupe(
                mb.NaiveDeduper, {"unique_fields": [crn.f("company")]}
            ),
            id="model",
        ),
        pytest.param(_dedupe_crn, id="resolver"),
    ],
)
def test_prepare_cached(
    source: Callable[..., mb.Source],
    build: Callable[[mb.Source], mb.Step],
) -> None:
    """A cache hit delays preparation until the rebuilt step needs its baseline."""
    crn = source("crn")
    first = build(crn).collect()
    assert first._prepared

    rebuilt = build(crn).collect()
    assert rebuilt._fp == first._fp
    assert not rebuilt._prepared

    rebuilt._ensure_prepared()
    assert rebuilt._prepared
    rebuilt._ensure_prepared()
    assert rebuilt._prepared


def _apex(source: Callable[..., mb.Source]) -> tuple[mb.Resolver, mb.Source, mb.Source]:
    """Build the full dedupe → link plan without ever constructing a DAG."""
    crn = source("crn")
    dh = source("dh")
    r_crn = _dedupe_crn(crn)
    apex = r_crn.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    return apex, crn, dh


def _required_resources(root: mb.Step) -> set[str]:
    """What `load` would have to be given, read off the dumped JSON."""
    return mb.PlanDocument.model_validate_json(mb.dump(root)).required_resources()


def _ids_by_key(matches: pl.DataFrame, column: str) -> dict[str, int]:
    return {
        row[column]: row["root"]
        for row in matches.iter_rows(named=True)
        if row[column] is not None
    }


# -- selective identifier reads ------------------------------------------------------


def test_identifiers_source_uncollected(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """An identifier read needs collected source leaves, even with an empty filter."""
    with pytest.raises(RuntimeError, match="has not been collected"):
        source("crn").identifiers(store, ids=set())


def test_identifiers_source_filtered(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """A source restricts identifiers by leaf without changing its full reading."""
    crn = source("crn").collect()
    all_ids = crn.identifiers(store)
    selected = all_ids.filter(pl.col("key") == "a1")["id"].item()

    assert crn.identifiers(store, ids={selected}).equals(
        all_ids.filter(pl.col("id") == selected)
    )


def test_identifiers_transform_filtered(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """A transform keeps the source's identifier reading when it reshapes data."""
    crn = source("crn")
    selected = crn.select("crn_company").collect()
    source_ids = crn.identifiers(store)
    leaf = source_ids.filter(pl.col("key") == "a2")["id"].item()

    assert selected.identifiers(store, ids={leaf}).equals(
        source_ids.filter(pl.col("id") == leaf)
    )


def test_identifiers_resolver_filtered(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """One affected root includes its leaves from both source readings."""
    apex, _, _ = _apex(source)
    apex.collect()
    all_ids = apex.identifiers(store)
    root = all_ids.filter((pl.col("source") == "crn") & (pl.col("key") == "a1"))[
        "id"
    ].item()

    assert set(apex.identifiers(store, ids={root})["key"]) == {"a1", "a2", "b1"}


def test_identifiers_resolver_empty(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """An empty filter keeps the schema when several sources are concatenated."""
    apex, _, _ = _apex(source)
    apex.collect()
    all_ids = apex.identifiers(store)
    empty = apex.identifiers(store, ids=set())

    assert empty.schema == all_ids.schema
    assert empty.is_empty()


# -- building and collecting ----------------------------------------------------------


def test_plan_needs_no_dag(source: Callable[..., mb.Source]) -> None:
    """A plan is reachable purely through upstream references, with no DAG object."""
    crn = source("crn")
    assert crn.parents == ()
    assert not crn.is_collected

    deduped = _dedupe_crn(crn)
    assert isinstance(deduped, mb.Resolver)
    # The plan is reachable purely through upstream references.
    assert crn in deduped.lineage()


def test_collect_is_lazy(source: Callable[..., mb.Source]) -> None:
    """Nothing runs until collect. Building a plan touches no warehouse."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)

    assert all(not step.is_collected for step in deduped.lineage())
    deduped.collect()
    assert all(step.is_collected for step in deduped.lineage())


def test_collect_across_sources(source: Callable[..., mb.Source]) -> None:
    """A dedupe feeding a link resolves records across both sources."""
    apex, _crn, _dh = _apex(source)

    lookup = apex.collect().get_lookup()
    crn_ids = _ids_by_key(lookup, "crn_pk")
    dh_ids = _ids_by_key(lookup, "dh_pk")

    # Dedupe survived forward through the apex link.
    assert crn_ids["a1"] == crn_ids["a2"]
    # The link joined acme across sources.
    assert dh_ids["b1"] == crn_ids["a1"]
    # Fall-through: records no model matched stay in their own clusters.
    assert crn_ids["a3"] != crn_ids["a1"]
    assert dh_ids["b2"] != crn_ids["a1"]
    assert crn_ids["a3"] != dh_ids["b2"]


def test_lookup_key_crosses_sources(source: Callable[..., mb.Source]) -> None:
    """lookup_key follows one record's entity across into another source."""
    apex, _crn, _dh = _apex(source)
    apex.collect()

    result = apex.lookup_key(from_source="crn", to_sources=["dh"], key="a1")
    assert set(result["crn"]) == {"a1", "a2"}
    assert set(result["dh"]) == {"b1"}


def test_lookup_key_unknown_raises(source: Callable[..., mb.Source]) -> None:
    """Looking up a key no source holds raises."""
    apex, _crn, _dh = _apex(source)
    apex.collect()
    with pytest.raises(StepNotFound):
        apex.lookup_key(from_source="crn", to_sources=["dh"], key="nope")


# -- laziness and caching -------------------------------------------------------------


def _sabotage(step) -> None:  # noqa: ANN001 - any Step
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError(f"{step!r} re-ran instead of being read from cache")

    step._execute = boom


def _seal(source) -> None:  # noqa: ANN001 - any Source
    """Make any further warehouse read fail, leaving the memoised one usable."""

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError(f"{source!r} re-read the warehouse")

    source.fetch = boom


def test_collect_recollect_noop(source: Callable[..., mb.Source]) -> None:
    """Re-collecting an unchanged plan re-runs nothing."""
    apex, _crn, _dh = _apex(source)
    apex.collect()

    for step in apex.lineage():
        _sabotage(step)
    apex.collect()  # must not raise


def test_collect_only_new_steps(
    source: Callable[..., mb.Source],
) -> None:
    """A new downstream branch runs only its new steps, not cached inputs."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    for step in deduped.lineage():
        _sabotage(step)

    # A brand-new downstream branch over already-collected inputs.
    dh = source("dh")
    apex = deduped.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    apex.collect()  # only dh + the new read/model/resolver run

    assert apex.get_lookup().height > 0


def test_renaming_resource(warehouse: Engine, store: mb.DuckDBStore) -> None:
    """Renaming a resource keeps the fingerprint stable."""

    def apex(resource_name: str) -> mb.Resolver:
        crn = mb.read_database(
            "crn",
            sql="select pk, company from crn",
            client=mb.Resource(resource_name, warehouse),
            key_field="pk",
        )
        return crn.dedupe(
            model_class="NaiveDeduper",
            model_settings={"unique_fields": ["crn_company"]},
        ).resolve()

    original = apex("warehouse").collect(store, interactive=False)
    renamed = apex("something_else_entirely").collect(store, interactive=False)

    assert renamed.fingerprints() == original.fingerprints()


def test_source_column_change_invalidates(
    warehouse: Engine, source: Callable[..., mb.Source]
) -> None:
    """Identity is the whole extract, so any selected column moves the fingerprint.

    Without that rule, a column outside identity could change in the warehouse
    without moving the source's fingerprint. The source would cache-hit instead of
    re-storing, and downstream record steps would keep reading the stale value.
    """
    et = "select pk, company, town from crn"

    cleaned = source("crn", et).clean({"town": "crn_town"})
    cleaned.collect()
    assert sorted(cleaned.data()["town"].to_list()) == ["hull", "leeds", "london"]

    with warehouse.begin() as conn:
        conn.execute(text("UPDATE crn SET town = 'oxford' WHERE pk = 'a1'"))

    refreshed = source("crn", et).clean({"town": "crn_town"})
    refreshed.collect()
    assert sorted(refreshed.data()["town"].to_list()) == ["hull", "leeds", "oxford"]


def test_source_identity_is_every_column(source: Callable[..., mb.Source]) -> None:
    """Selecting a column makes it part of the record, so it splits leaves.

    Leave it out of the extract and the rows collapse to one leaf. The SELECT is the
    whole declaration of a source's identity.
    """
    # a1/a2 agree on company and differ on town.
    with_town = source("crn", "select pk, company, town from crn")
    with_town.collect()
    leaves = dict(with_town.leaves().iter_rows())
    assert leaves["a1"] != leaves["a2"]

    without_town = source("crn", "select pk, company from crn")
    without_town.collect()
    leaves = dict(without_town.leaves().iter_rows())
    assert leaves["a1"] == leaves["a2"]


@pytest.mark.parametrize("name", ["crn-x", "crn.x", "1crn", "crn x", ""])
def test_source_name_must_prefix(warehouse: Engine, name: str) -> None:
    """Caught at construction, not as a SQL error three steps later.

    A source's name qualifies every column it contributes, and those land in clean and
    select SQL. `crn-x_company` parses as subtraction, `crn.x_company` as table.column.
    """
    with pytest.raises(ValueError, match="can't prefix a column name"):
        mb.read_database(
            name,
            sql="select pk, company from crn",
            client=warehouse,
            key_field="pk",
        )


def test_source_name_reserved_word(warehouse: Engine) -> None:
    """The name is only ever a prefix, so `select_company` is unambiguous."""
    source = mb.read_database(
        "select",
        sql="select pk, company from crn",
        client=warehouse,
        key_field="pk",
    )
    cleaned = source.clean({"name": f"lower({source.f('company')})"})
    cleaned.collect()
    assert sorted(cleaned.data()["name"].to_list()) == ["acme", "acme", "beta"]


def test_repeated_source_name(source: Callable[..., mb.Source]) -> None:
    """Two sources in one plan cannot share a name, and the join is where that shows."""
    left = source("crn")
    right = source("crn", "select pk, company, town from dh")

    with pytest.raises(ValueError, match="covers two different steps"):
        left.link(
            right,
            model_class=mb.DeterministicLinker,
            model_settings={"comparisons": ["l.crn_company = r.crn_company"]},
        )


def test_resource_sharing(warehouse: Engine) -> None:
    """Two locations can share one warehouse via resources.

    Sharing a name means sharing the object, which is what makes
    `test_repeated_resource_name` a refusal rather than the normal case.
    """
    client = mb.Resource("wh", warehouse)
    left = mb.read_database("crn", sql="select pk, company from crn", client=client)
    right = mb.read_database("dh", sql="select pk, company from dh", client=client)
    plan = left.link(
        right,
        model_class="DeterministicLinker",
        model_settings={"comparisons": ["l.crn_company = r.dh_company"]},
    ).resolve()

    assert _required_resources(plan) == {"wh"}


def test_repeated_resource_name(
    source: Callable[..., mb.Source],
    warehouse: Engine,
    sqlite_in_memory_warehouse: Engine,
) -> None:
    """One resource name over two objects is refused wherever the branches meet."""
    left = source("crn", client=mb.Resource("wh", warehouse)).clean(
        {"company": "lower(crn_company)"}
    )
    right = source("dh", client=mb.Resource("wh", sqlite_in_memory_warehouse))

    with pytest.raises(ResourceError, match="covers two different objects"):
        left.link(
            right,
            model_class=mb.DeterministicLinker,
            model_settings={"comparisons": ["l.company = r.dh_company"]},
        )


class _TwoClientLocation(mb.Location):
    """A location holding two clients it never reads through.

    No shipped location takes two resources, and one that does is the only way a leaf
    can conflict with itself.
    """

    reader: mb.FromResources[Engine]
    writer: mb.FromResources[Engine]

    def read(self, *args: object, **kwargs: object) -> Iterator[pl.DataFrame]:
        """Never called: this source is built, not collected."""
        raise NotImplementedError


def test_leaf_conflicts_with_itself(
    warehouse: Engine, sqlite_in_memory_warehouse: Engine
) -> None:
    """A source has no inputs, but its own two fields can still collide."""
    with pytest.raises(ResourceError, match="covers two different objects"):
        mb.Source(
            location_class=_TwoClientLocation,
            name="crn",
            location_resources={
                "reader": mb.Resource("session", warehouse),
                "writer": mb.Resource("session", sqlite_in_memory_warehouse),
            },
            key_field="pk",
        )


def test_resource_name_reuse(
    source: Callable[..., mb.Source],
    warehouse: Engine,
    sqlite_in_memory_warehouse: Engine,
) -> None:
    """A name is a promise to the steps that can see it, and no further.

    A plan is what is reachable upstream, so two that never meet are two namespaces and
    both may name an engine 'wh'.
    """
    left = source("crn", client=mb.Resource("wh", warehouse))
    right = source("dh", client=mb.Resource("wh", sqlite_in_memory_warehouse))

    assert _required_resources(left) == {"wh"}
    assert _required_resources(right) == {"wh"}


def test_anonymous_resources_no_collision(
    source: Callable[..., mb.Source], warehouse: Engine
) -> None:
    """A bare object has no name, so two different ones in a plan are no conflict.

    `test_resources.py::test_bare_value` covers `dump` refusing them afterwards.
    """
    left = source("crn", client=warehouse)
    right = source("dh", client=create_engine("sqlite://"))

    assert left.link(
        right,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": ["l.crn_company = r.dh_company"]},
    ).parents == (left, right)


def test_source_key_only_rejected(source: Callable[..., mb.Source]) -> None:
    """A source selecting only its key, nothing to match on, is rejected."""
    source = source("crn", "select pk from crn")
    with pytest.raises(ValueError, match="only its key field"):
        source.collect()


def test_source_no_rows_raises(source: Callable[..., mb.Source]) -> None:
    """A query that returns nothing is a mistake, not an empty source."""
    empty = source("crn", "select pk, company from crn where pk = 'nobody'")
    with pytest.raises(ValueError, match="returned no rows"):
        empty.collect()


def test_key_field_vs_mb_id(store: mb.DuckDBStore) -> None:
    """`key_field` defaults to `id`, which is also what a record step calls its own.

    The two never meet: `build_record_step` prefixes every column an extract returns
    with the source name before anything else, so no origin column of any name can reach
    a record step as bare `id`. The key is then renamed to `key`,
    joined on, and dropped. A record step's `id` is always the derived leaf.
    """
    source = mb.read_dataframe(
        "crn",
        df=pl.DataFrame({"id": ["a1", "a2"], "company": ["acme", "beta"]}),
    )
    source.collect(store)

    records = source.data()
    assert set(records.columns) == {"crn_company", "id"}
    assert set(records["id"]).isdisjoint({"a1", "a2"}), "origin keys reached `id`"
    assert set(source.leaves()["key"]) == {"a1", "a2"}


def test_source_missing_key_field(warehouse: Engine) -> None:
    """A missing key column is explained, including where the name came from."""
    with pytest.raises(ValueError, match="has no column 'id'") as caught:
        mb.read_database(
            "crn", sql="select pk, company from crn", client=warehouse
        ).collect()
    assert "It returned: pk, company" in str(caught.value)
    assert "defaults to 'id'" in str(caught.value)

    # A frame reaches the same check by the same route.
    with pytest.raises(ValueError, match="has no column 'id'"):
        mb.read_dataframe(
            "dh", df=pl.DataFrame({"pk": ["b1"], "company": ["acme"]})
        ).collect()

    # And so does an explicit key that the location does not return.
    with pytest.raises(ValueError, match="has no column 'pak'"):
        mb.read_database(
            "crn",
            sql="select pk, company from crn",
            client=warehouse,
            key_field="pak",
        ).collect()


def test_source_null_keys_raises(source: Callable[..., mb.Source]) -> None:
    """A null key can't anchor a record, so it's rejected at read time."""
    null_keyed = source("crn", "select null as pk, company from crn")
    with pytest.raises(ValueError, match="has null keys"):
        null_keyed.collect()


def test_source_keys_are_strings(warehouse: Engine) -> None:
    """Whatever the warehouse types the key as, matchlab hands back a string."""
    with warehouse.begin() as conn:
        conn.execute(text("CREATE TABLE nums (id INTEGER, company TEXT)"))
        conn.execute(text("INSERT INTO nums VALUES (1,'acme'),(2,'beta')"))

    source = mb.read_database(
        "nums", sql="select id, company from nums", client=warehouse
    )
    source.collect()
    assert sorted(source.leaves()["key"].to_list()) == ["1", "2"]


def test_plan_over_dataframes(store: mb.DuckDBStore) -> None:
    """A whole plan can run with nothing but dataframes."""
    crn = mb.read_dataframe(
        "crn",
        df=pl.DataFrame(
            {
                "pk": ["a1", "a2", "a3"],
                "company": ["acme", "acme", "beta"],
                "town": ["london", "leeds", "hull"],
            }
        ),
        key_field="pk",
    )
    dh = mb.read_dataframe(
        "dh",
        df=pl.DataFrame(
            {"pk": ["b1", "b2"], "company": ["acme", "gamma"], "town": ["a", "b"]}
        ),
        key_field="pk",
    )

    resolved = crn.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": ["l.crn_company = r.dh_company"]},
    ).resolve()
    resolved.collect(store)

    lookup = resolved.get_lookup()
    # Every key on both sides is reachable, including the ones nothing matched.
    assert set(lookup["crn_pk"].drop_nulls()) == {"a1", "a2", "a3"}
    assert set(lookup["dh_pk"].drop_nulls()) == {"b1", "b2"}


def test_source_memoises_read(
    warehouse: Engine, source: Callable[..., mb.Source]
) -> None:
    """Re-collecting an existing Source must not re-read the warehouse.

    Asserted against `fetch`, the call that actually goes to the warehouse, rather than
    against `_read_origin`, which is the memo itself. `_ensure` recomputes every
    fingerprint on every collect, so a source's `_spec_key` does run again and does
    re-hash the rows. What it must not do is fetch them again.
    """
    crn = source("crn")
    crn.collect()

    _seal(crn)
    crn.collect()  # the memoised read carries the fingerprint


# -- immutability ---------------------------------------------------------------------


def test_spec_attributes_read_only(source: Callable[..., mb.Source]) -> None:
    """Everything a spec is built from refuses assignment, saying what to do instead."""
    crn = source("crn")
    model = crn.dedupe(
        model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["crn_company"]}
    )
    resolver = model.resolve()

    # Non-comprenehsinve list of read-only attributes
    for step, attribute, value in [
        (crn, "location", None),
        (model, "model_settings", None),
        (resolver, "resolver_settings", None),
    ]:
        with pytest.raises(AttributeError, match="read-only"):
            setattr(step, attribute, value)

    # The settings objects are frozen too, so there is no way round the guard.
    with pytest.raises(ValidationError):
        model.model_instance.unique_fields = ["crn_town"]
    with pytest.raises(ValidationError):
        resolver.resolver_instance.thresholds = {0: 0.9}


def test_parents(source: Callable[..., mb.Source]) -> None:
    """`parents` is derived for all step kinds."""
    crn, dh = source("crn"), source("dh")
    transform = crn.clean({"crn_company": "lower(crn_company)"})
    dedupe = transform.dedupe(mb.NaiveDeduper, {"unique_fields": ["crn_company"]})
    link = crn.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": "l.crn_company = r.dh_company"},
    )
    resolver = dedupe.resolve()

    assert crn.parents == ()  # a source is a leaf
    assert transform.parents == (crn,)
    assert dedupe.parents == (transform,)  # a deduper reads one
    assert link.parents == (crn, dh)  # a linker reads two, left then right
    assert resolver.parents == (dedupe,)  # a resolver reads models

    # Derived, so there is nothing to assign to.
    for step in (crn, transform, dedupe, resolver):
        with pytest.raises(AttributeError):
            step.parents = ()


def test_editable_step_attributes() -> None:
    """The read-only guard does not cover every attribute."""

    class Custom(mb.Step):
        kind = StepKind.SOURCE

        def __init__(self) -> None:
            super().__init__()
            self.label = "mine"

        @property
        def parents(self) -> tuple[mb.Step, ...]:
            return ()

        @property
        def spec(self) -> SourceSpec:  # pragma: no cover - never collected
            raise NotImplementedError

        def _execute(self, store: object, fp: bytes) -> None:  # pragma: no cover
            raise NotImplementedError

        def _methodology_class(self) -> type | None:  # pragma: no cover
            return None

    assert Custom().label == "mine"


def test_rebuilding_unchanged_cache(source: Callable[..., mb.Source]) -> None:
    """Rebuilding is how a plan changes, and it costs only the steps that moved.

    The source and the transform above the edit address the same artifacts as before,
    so they are cache hits. Only the model and the resolver run again. That is what
    makes rebuilding an acceptable price for immutability.
    """
    crn = source("crn")

    def build(unique_fields: list[str]) -> mb.Resolver:
        return (
            crn.clean({"crn_company": "lower(crn_company)"})
            .dedupe(
                model_class=mb.NaiveDeduper, model_settings={"unique_fields": unique}
            )
            .resolve()
        )

    unique = ["crn_company"]
    first = build(unique)
    first.collect()
    before = [step._fp for step in first.lineage()]

    unique = ["crn_town"]  # the edit
    second = build(unique)
    second.collect()
    after = [step._fp for step in second.lineage()]

    # source, transform unchanged; model, resolver moved.
    assert before[:2] == after[:2]
    assert before[2:] != after[2:]


def test_derived_class_attributes_drift(source: Callable[..., mb.Source]) -> None:
    """`transformer_class` and `model_type` are read off what they describe."""
    crn = source("crn")
    transform = crn.clean({"crn_company": "lower(crn_company)"})
    model = crn.dedupe(
        model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["crn_company"]}
    )

    assert transform.transformer_class is type(transform.transformer)
    assert transform.spec.transformer_class == "Clean"
    assert model.model_type is ModelType.DEDUPER

    # Neither is a stored attribute, so neither can be written out of agreement.
    with pytest.raises(AttributeError):
        transform.transformer_class = mb.Select
    with pytest.raises(AttributeError):
        model.model_type = ModelType.LINKER


def test_mistyped_settings(source: Callable[..., mb.Source]) -> None:
    """A setting no methodology declares is an error, at every kind of step."""
    crn = source("crn")

    with pytest.raises(ValidationError, match="typo"):
        mb.Source(
            location_class=crn.location_class,
            name="crn",
            location_settings={"sql": "select pk from crn", "typo": 1},
            location_resources=crn.location_resources,
        )

    with pytest.raises(ValidationError, match="max_combinatons"):
        crn.transform(mb.Explode, {"max_combinatons": 5})

    with pytest.raises(ValidationError, match="unique_feilds"):
        crn.dedupe(model_class=mb.NaiveDeduper, model_settings={"unique_feilds": ["x"]})

    deduped = crn.dedupe(
        model_class=mb.NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    )
    with pytest.raises(ValidationError, match="threshold"):
        deduped.resolve(resolver_settings={"threshold": 0.5})


def test_settings_vs_methodology_attr(
    source: Callable[..., mb.Source],
) -> None:
    """`spec` describes the object `_execute` runs, so the two cannot disagree.

    A step used to hold settings passed separately from the instance built out of them,
    which is two things that start equal and can stop being equal. A step now keeps one
    instance and reads its settings back off it.

    That also normalises defaults: a field the call site omitted is still part of the
    configuration, so it belongs in the key. Otherwise the same methodology, written two
    ways, would fingerprint differently and re-run for nothing.
    """
    crn = source("crn")
    model = crn.dedupe(
        model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["crn_company"]}
    )
    assert model.model_settings == model.model_instance.model_dump(mode="json")
    assert model.model_settings["id"] == "id"  # defaulted, never passed

    resolver = model.resolve()
    assert resolver.resolver_settings == resolver.resolver_instance.model_dump(
        mode="json"
    )


# -- Record step storage ---------------------------------------------------------


@pytest.fixture
def computes(monkeypatch: pytest.MonkeyPatch) -> dict[int, int]:
    """Count `Transform._execute` calls per transform, keyed by `id`."""
    counts: dict[int, int] = {}
    original = mb.Transform._execute

    def counting(self: mb.Transform, store: mb.DuckDBStore, fp: bytes) -> None:
        counts[id(self)] = counts.get(id(self), 0) + 1
        return original(self, store, fp)

    monkeypatch.setattr(mb.Transform, "_execute", counting)
    return counts


@pytest.fixture
def model_runs(monkeypatch: pytest.MonkeyPatch) -> list[mb.Model]:
    """Record which models actually executed, rather than hitting cache."""
    ran: list[mb.Model] = []
    original = mb.Model._execute

    def counting(self: mb.Model, store: mb.DuckDBStore, fp: bytes) -> None:
        ran.append(self)
        return original(self, store, fp)

    monkeypatch.setattr(mb.Model, "_execute", counting)
    return ran


def _shared_record_step_plan(
    source: Callable[..., mb.Source], comparison: str | None = None
) -> tuple[mb.Resolver, mb.Transform]:
    """One cleaned record step feeding both a dedupe and a link, joined by a resolver.

    `comparison` retunes the linker without touching the record step, so a second plan
    can invalidate the models while the transform's fingerprint still hits.
    """
    crn = source("crn")
    dh = source("dh")
    record_step = crn.clean({"name": "crn_company"})
    deduped = record_step.dedupe(
        model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["name"]}
    )
    linked = record_step.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": comparison or f"l.name = r.{dh.f('company')}"},
    )
    return deduped.resolve(linked), record_step


def test_record_step_stored(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """Collecting a transform's consumer stores the transform too."""
    crn = source("crn")
    record_step = crn.clean({"name": "crn_company"})
    deduped = record_step.dedupe(
        model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["name"]}
    ).resolve()
    deduped.collect()

    assert record_step.is_collected
    assert store.has(record_step._fp)


def test_record_step_one_session(
    source: Callable[..., mb.Source],
    store: mb.DuckDBStore,
    computes: dict[int, int],
) -> None:
    """The point of storing: fan-out costs one computation, not one per consumer."""
    apex, record_step = _shared_record_step_plan(source)

    apex.collect()

    assert computes[id(record_step)] == 1
    assert store.has(record_step._fp)


def test_record_step_across_sessions(
    source: Callable[..., mb.Source],
    store: mb.DuckDBStore,
    computes: dict[int, int],
    model_runs: list[mb.Model],
) -> None:
    """A stored record step is read back across sessions, not recomputed.

    The second plan is fresh objects, as a new process would build, with the linker
    retuned so the models genuinely re-run. Their inputs are unchanged, so the transform
    still fingerprints the same and hits cache, reading the record step off disk rather
    than recomputing it.
    """
    dh = source("dh")
    apex, _record_step = _shared_record_step_plan(source)
    apex.collect()
    computes.clear()
    model_runs.clear()

    retuned, record_step = _shared_record_step_plan(
        source, comparison=f"r.{dh.f('company')} = l.name"
    )
    retuned.collect()

    assert computes == {}, "a stored record step was recomputed"
    assert store.has(record_step._fp)
    # The linker really did re-run, so something genuinely asked for the record step.
    # Without this the assertion above would hold trivially.
    assert [model.model_class for model in model_runs] == [mb.DeterministicLinker]


def test_record_step_collection(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """A record step collected on its own materialises its table."""
    crn = source("crn")
    record_step = crn.clean({"name": "crn_company"})

    data = record_step.collect().data()
    assert store.has(record_step._fp)
    assert "name" in data.columns
    assert data.height == 3


# -- identifiers ----------------------------------------------------------------------


@pytest.fixture
def identifier_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bytes | None]]:
    """Record the `(source_name, resolver_fp)` of every `read_identifiers` call."""
    calls: list[tuple[str, bytes | None]] = []
    original = mb.DuckDBStore.read_identifiers

    def counting(
        self: mb.DuckDBStore,
        source_fp: bytes,
        source_name: str,
        resolver_fp: bytes | None = None,
    ) -> pl.DataFrame:
        calls.append((source_name, resolver_fp))
        return original(self, source_fp, source_name, resolver_fp)

    monkeypatch.setattr(mb.DuckDBStore, "read_identifiers", counting)
    return calls


def _fan_out_plan(
    source: Callable[..., mb.Source],
) -> tuple[mb.Resolver, list[mb.Model]]:
    """Three models over one shared record step, so the resolver sees repeated readings.

    `Resolver._execute` walks `(model, record_step)` pairs, four of them here. Only two
    distinct readings exist between them, which is what it must collapse to. Linking
    every pair of n sources makes that ratio quadratic.
    """
    crn = source("crn")
    dh = source("dh")
    record_step = crn.clean({"name": "crn_company"})
    models = [
        record_step.dedupe(
            model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["name"]}
        ),
        record_step.link(
            dh,
            model_class=mb.DeterministicLinker,
            model_settings={"comparisons": f"l.name = r.{dh.f('company')}"},
        ),
        record_step.dedupe(
            model_class=mb.NaiveDeduper,
            model_settings={"unique_fields": ["name", "id"]},
        ),
    ]
    return models[0].resolve(*models[1:]), models


def test_read_identifiers_per_source(
    source: Callable[..., mb.Source],
    store: mb.DuckDBStore,
    identifier_reads: list[tuple[str, bytes | None]],
) -> None:
    """Once per source, not once per model: the quadratic the resolver collapses."""
    apex, models = _fan_out_plan(source)
    assert sum(len(model.parents) for model in apex.parents) == 4  # the pairs it walks

    # Collect the models first so their own reads are done and cleared. What the apex
    # collect records is then the resolver's alone.
    for model in models:
        model.collect()
    identifier_reads.clear()
    apex.collect()

    assert sorted(identifier_reads) == [("crn", None), ("dh", None)]


def test_read_identifiers_dedup(
    source: Callable[..., mb.Source],
) -> None:
    """The merge-forward guarantee, which the dedup must not weaken.

    Every reachable leaf has to reach the resolver output, including records no model
    matched. A reading dropped here loses clusters silently rather than failing, so
    assert on the records rather than on the call count.
    """
    apex, _models = _fan_out_plan(source)

    resolver_output = apex.collect().entities()

    assert dict(resolver_output.group_by("source").len().iter_rows()) == {
        "crn": 3,
        "dh": 2,
    }
    assert set(resolver_output.filter(pl.col("source") == "crn")["key"]) == {
        "a1",
        "a2",
        "a3",
    }
    assert set(resolver_output.filter(pl.col("source") == "dh")["key"]) == {
        "b1",
        "b2",
    }


# -- lineage navigation ---------------------------------------------------------------


def test_lineage_upstream_only(source: Callable[..., mb.Source]) -> None:
    """A step knows its inputs and nothing else, so lineage never looks down."""
    apex, crn, _dh = _apex(source)

    assert crn in apex.lineage()
    assert apex not in crn.lineage()
    assert crn.lineage() == [crn]


def test_lineage_draw_sub_plan(source: Callable[..., mb.Source]) -> None:
    """draw() from a sub-plan shows only that sub-plan."""
    apex, crn, _dh = _apex(source)

    assert "crn" in apex.draw()
    assert "dh" in apex.draw()
    assert "resolver" not in crn.draw()  # a source's sub-plan is just itself


def test_resolver_exposes_sources(source: Callable[..., mb.Source]) -> None:
    """A resolver exposes the sources reachable beneath it."""
    apex, _crn, _dh = _apex(source)
    assert {source.name for source in apex.sources} == {"crn", "dh"}


# -- storage lifetime -----------------------------------------------------------------


def test_gc_keeps_artifacts(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """A store keeps what it was given. Dropping the Python objects reclaims nothing.

    Deliberate. An artifact's value has nothing to do with whether this process still
    holds the variable that produced it. The next process rebuilding the same plan
    wants a cache hit. Reclaiming is the owner's explicit act, not the library's.
    """
    import gc as pygc  # noqa: PLC0415 - the interpreter's, to force reachability

    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()
    fp = deduped._fp
    assert store.has(fp)

    del crn, deduped
    pygc.collect()

    assert store.has(fp)


def test_fingerprints_every_artifact(
    source: Callable[..., mb.Source],
) -> None:
    """Every step in a collected plan carries a fingerprint."""
    crn = source("crn")
    plan = _dedupe_crn(crn)
    plan.collect()

    assert plan.fingerprints() == {step._fp for step in plan.lineage()}


def test_fingerprints_collapse_shared(
    source: Callable[..., mb.Source],
) -> None:
    """Two distinct steps can be the same artifact: same spec, same inputs.

    A set is what makes that safe further down. A store told to keep one of them and
    delete the other would delete the bytes both of them are.
    """
    crn = source("crn")
    cleaning = {"name": "crn_company"}
    twins = [crn.clean(cleaning), crn.clean(cleaning)]
    plan = (
        twins[0]
        .dedupe(model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["name"]})
        .resolve(
            twins[1].dedupe(
                model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["name"]}
            )
        )
    )
    plan.collect()

    assert twins[0] is not twins[1]
    assert twins[0]._fp == twins[1]._fp
    assert len(plan.fingerprints()) < len(plan.lineage())


def test_fingerprints_uncollected(source: Callable[..., mb.Source]) -> None:
    """An uncollected plan names no artifacts, so it must not answer with a smaller set.

    Silently returning what happens to be collected would tell a caller that less is
    worth keeping than they think. The caller here is about to delete the rest.
    """
    plan = _dedupe_crn(source("crn"))

    with pytest.raises(RuntimeError, match="has not been collected"):
        plan.fingerprints()


def test_prune_to_plan_cache(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """The property a prune has to have: it removes only what was superseded.

    Editing a cleaning expression strands the whole subtree below it. That is what
    fills a store. Pruning to the plan you kept should take exactly those, and leave a
    store the same plan still hits cache on. If it took one artifact too many the next
    collect quietly recomputes, and the prune has cost work rather than saved space.
    """

    def build(cleaning: str) -> mb.Resolver:
        return (
            source("crn")
            .clean({"name": cleaning})
            .dedupe(
                model_class=mb.NaiveDeduper, model_settings={"unique_fields": ["name"]}
            )
            .resolve()
        )

    first = build("crn_company")
    first.collect()
    superseded = {step._fp for step in first.lineage()}

    plan = build("upper(crn_company)")  # the edit
    plan.collect()
    live = {step._fp for step in plan.lineage()}

    stranded = superseded - live
    assert stranded, "the edit should have stranded something"

    result = store.prune(keep=plan.fingerprints())

    assert result.removed == len(stranded)
    assert all(store.has(fp) for fp in live)
    assert not any(store.has(fp) for fp in stranded)

    # The real assertion: rebuild the same plan over the pruned store and let it run.
    # Nothing may execute here. If the prune took one artifact too many, this raises.
    rebuilt = build("upper(crn_company)")
    for step in rebuilt.lineage():
        _sabotage(step)
    rebuilt.collect()


# -- reading through a resolver, and grouping -----------------------------------------


def test_resolver_record_step(source: Callable[..., mb.Source]) -> None:
    """A resolver is a record step, read at `id`=root and matchable on top of directly.

    Reading it returns its records regrouped by entity. Linking it directly to a
    source, with no transform in between, still carries the dedupe grouping forward.
    That's the merge-forward guarantee working through a resolver record step.
    """
    crn = source("crn")
    dh = source("dh")
    deduped = _dedupe_crn(crn)

    # Read as a record step. crn's records sit at `id`=root, so a1/a2 share one id.
    records = deduped.data()
    assert records.height == 3
    assert records["id"].n_unique() == 2

    # Matched on top of directly, with no `.read()` and no transform between.
    apex = deduped.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    lookup = apex.collect().get_lookup()
    crn_ids = _ids_by_key(lookup, "crn_pk")
    dh_ids = _ids_by_key(lookup, "dh_pk")

    assert crn_ids["a1"] == crn_ids["a2"] == dh_ids["b1"]  # dedupe survived the link
    assert crn_ids["a3"] != crn_ids["a1"]  # fall-through singleton


def test_resolver_read_per_record(
    source: Callable[..., mb.Source],
) -> None:
    """Without a group, a read is record-grained even when `id` is an entity."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    record_step = deduped.clean({"name": "crn_company"})
    record_step.collect()
    data = record_step.data()

    # a1/a2 deduped to one entity, but each contributes a row.
    assert data.height == 3
    assert data["id"].n_unique() == 2


def test_group_one_row_per_entity(source: Callable[..., mb.Source]) -> None:
    """`group` collapses each id, with the aggregate saying how per column."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    record_step = deduped.group(
        {
            "name": "any_value(crn_company)",
            "towns": "list(distinct crn_town)",
        }
    )
    record_step.collect()
    data = record_step.data().sort("name")

    assert data.height == 2
    assert data["name"].to_list() == ["acme", "beta"]
    # The deduped entity keeps both towns rather than silently losing one.
    assert sorted(data["towns"][0]) == ["leeds", "london"]


def test_group_merges_forward(source: Callable[..., mb.Source]) -> None:
    """Grouping changes the record step's grain, never the resolver output's.

    Leaves travel via `identifiers()`, read from the store, so collapsing rows in
    the record step cannot lose a record from the resolver output below it.
    """
    crn = source("crn")
    dh = source("dh")
    deduped = _dedupe_crn(crn)

    apex = (
        deduped.group({"name": "any_value(crn_company)"})
        .link(
            dh,
            model_class=mb.DeterministicLinker,
            model_settings={"comparisons": f"l.name = r.{dh.f('company')}"},
        )
        .resolve()
    )
    lookup = apex.collect().get_lookup()
    crn_ids = _ids_by_key(lookup, "crn_pk")
    dh_ids = _ids_by_key(lookup, "dh_pk")

    # Every source record still appears, with the dedupe carried forward.
    assert set(crn_ids) == {"a1", "a2", "a3"}
    assert set(dh_ids) == {"b1", "b2"}
    assert crn_ids["a1"] == crn_ids["a2"] == dh_ids["b1"]
    assert crn_ids["a3"] != crn_ids["a1"]


def test_group_multi_source(
    source: Callable[..., mb.Source],
) -> None:
    """Why `group` exists: several sources under one entity.

    Reading two sources through a resolver concatenates diagonally, so crn rows carry
    null dh columns and vice versa, and a comparison on `l.dh_company` is null on
    every crn row. Grouping puts the entity on a single populated row.
    """
    crn = source("crn")
    dh = source("dh")
    linked = crn.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    linked.collect()

    ungrouped = linked.clean({"c": "crn_company", "d": "dh_company"})
    ungrouped.collect()
    acme = ungrouped.data().filter(pl.col("c") == "acme")
    # Every crn row for the acme entity has a null dh column.
    assert acme["d"].null_count() == acme.height

    grouped = linked.group(
        {
            "company": "any_value(crn_company)",
            "towns": "list(distinct coalesce(crn_town, dh_town))",
        }
    )
    grouped.collect()
    result = grouped.data().filter(pl.col("company") == "acme")

    # One row, both sources' values present, because any_value skips the nulls.
    assert result.height == 1
    assert set(result["towns"][0]) == {"london", "leeds", "bristol"}


def test_group_without_aggregates(source: Callable[..., mb.Source]) -> None:
    """Grouping with no aggregates to say how columns combine is rejected."""
    crn = source("crn")
    with pytest.raises(ValueError, match="aggregate expressions"):
        crn.group({})


def test_explode(source: Callable[..., mb.Source]) -> None:
    """`Explode` gives the cross product across sources, the case `group` skips.

    `group` collapses this diagonally-concatenated record step to one populated row
    (see `test_group_multi_source`). `Explode` instead gives one row per combination of
    each source's populated values.
    """
    crn = source("crn")
    dh = source("dh")
    linked = crn.link(
        dh,
        model_class=mb.DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    linked.collect()

    exploded = linked.transform(mb.Explode())
    exploded.collect()
    acme = exploded.data().filter(pl.col("dh_company") == "acme")

    # crn's two towns cross dh's one, both sources' values present on every row.
    assert acme.height == 2
    assert set(acme["crn_town"].to_list()) == {"london", "leeds"}
    assert acme["dh_town"].unique().to_list() == ["bristol"]


# -- specs ----------------------------------------------------------------------------


def test_spec_serialisable(source: Callable[..., mb.Source]) -> None:
    """Each step kind reports its settings through a model, and it round-trips JSON."""
    apex, _crn, _dh = _apex(source)
    apex.collect()

    kinds = set()
    for step in apex.lineage():
        dumped = step.spec.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped
        # The spec is the fingerprint payload, so it must be stable. That holds
        # because every methodology here declares a version. An unversioned one
        # keys off a fresh nonce and is deliberately different every call.
        assert step._spec_key() == step._spec_key()
        kinds.add(step.kind)

    assert kinds == {"source", "model", "resolver"}


def test_spec_upstream_settings(source: Callable[..., mb.Source]) -> None:
    """Specs describe a step's own settings. Edges live on `upstream`."""
    apex, crn, _dh = _apex(source)

    resolver_spec = apex.spec.model_dump(mode="json")
    assert "sql" not in json.dumps(resolver_spec)
    # No upstream reference at all: inputs arrive as parent fingerprints, and a
    # setting that points at one uses its position.
    assert set(resolver_spec) == {"resolver_class", "resolver_settings"}


def test_read_direct_vs_resolver(
    source: Callable[..., mb.Source],
) -> None:
    """Reading a source directly and through a resolver are different record steps.

    Read directly, a source is a leaf record step (`id`=leaf). A resolver is itself a
    record step (`id`=root), so matching on it reads the same records regrouped by
    entity. Deduping again over that resolver output is a genuinely different operation
    from the first dedupe, so the two models get different fingerprints.
    """
    crn = source("crn")
    settings = {"unique_fields": [crn.f("company")]}

    first = crn.dedupe(model_class=mb.NaiveDeduper, model_settings=settings)
    deduped = first.resolve()  # a resolver, and a record step reading crn at id=root
    second = deduped.dedupe(model_class=mb.NaiveDeduper, model_settings=settings)

    assert crn.parents == ()  # a source read directly is a leaf
    assert second.left is deduped  # the model reads the resolver directly, no wrapper

    second.resolve().collect()
    assert first._fp != second._fp  # a second pass is not the first one


# -- identity: positions, and published names -----------------------------------------


def test_steps_use_positions(source: Callable[..., mb.Source]) -> None:
    """Nothing in a plan is named.

    Cleaning one source two ways, or comparing two methodologies over it, needs no
    names and cannot collide.
    """
    crn = source("crn")
    strict = crn.clean({"name": f"upper({crn.f('company')})"})
    loose = crn.clean({"name": f"lower({crn.f('company')})"})
    first = strict.dedupe(mb.NaiveDeduper, {"unique_fields": ["name"]})
    second = loose.dedupe(mb.NaiveDeduper, {"unique_fields": ["name", "id"]})

    apex = first.resolve(second)
    apex.collect()

    assert not hasattr(first, "name")
    positions = lineage.number(apex)
    assert positions[id(first)] != positions[id(second)]
    assert apex.entities().height > 0


def test_draw_keys_the_log(source: Callable[..., mb.Source]) -> None:
    """A log line saying `step 4` has to be findable in the plan's drawing."""
    apex, crn, _dh = _apex(source)
    apex.collect()

    drawing = apex.draw()
    for step, position in lineage.number(apex).items():
        del step
        assert f"[{position}] " in drawing
    # A source is drawn with its name, because a source has one.
    assert f"source '{crn.name}'" in drawing
    assert "model(DeterministicLinker)" in drawing
    assert "model(NaiveDeduper)" in drawing
    assert "resolver(Components)" in drawing


def test_publish_points_label(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """Publishing is an act on a result, not a property of the plan."""
    apex, _crn, _dh = _apex(source)
    apex.collect(store)

    assert store.labels() == []  # collecting publishes nothing
    apex.publish("entities")

    assert store.labels() == ["entities"]
    assert store.find("entities") == apex._fp


def test_publish_idempotent(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """Re-running an unchanged pipeline must not fail. Repointing must be deliberate."""
    apex, crn, _dh = _apex(source)
    apex.collect(store).publish("entities")
    apex.publish("entities")  # same resolver output, same name, so this is a no-op

    other = _dedupe_crn(crn).collect(store)
    with pytest.raises(
        ValueError, match="already points at a different resolver output"
    ):
        other.publish("entities")

    other.publish("entities", overwrite=True)
    assert store.find("entities") == other._fp


def test_publish_pins_a_refreshed_run(
    source: Callable[..., mb.Source], store: mb.DuckDBStore
) -> None:
    """A label keeps resolving to the output it was published against.

    An unversioned step re-keys on every collect, so a later collect writes a second
    artifact rather than replacing the first. That is what keeps a label meaningful,
    and what lets `publish` tell a moved result from an unchanged one: were the run
    stored back over its own fingerprint, the label would follow the new bytes with
    nothing raised and nothing said.
    """
    global _SUFFIX
    crn = source("crn")
    tagged = crn.transform(_Unversioned(column=crn.f("company")))
    resolver = tagged.dedupe(mb.NaiveDeduper, {"unique_fields": ["tag"]}).resolve()
    resolver.collect(store).publish("entities")
    published = store.find("entities")

    _SUFFIX = "-second"
    resolver.collect(store)
    _SUFFIX = "-first"

    assert store.find("entities") == published
    assert resolver._fp != published, "a refreshed run must not overwrite the old one"
    with pytest.raises(
        ValueError, match="already points at a different resolver output"
    ):
        resolver.publish("entities")


def test_publish_needs_collected(
    source: Callable[..., mb.Source],
) -> None:
    """There is nothing to point a name at until the resolver output exists."""
    apex, _crn, _dh = _apex(source)
    with pytest.raises(RuntimeError, match="has not been collected"):
        apex.publish("entities")


# -- resolver settings that point at inputs -------------------------------------------


def test_threshold_stored_as_position(
    source: Callable[..., mb.Source],
) -> None:
    """You hold the model. The plan works out where it sits.

    Positions rather than names are what let inputs be referred to without a naming
    scheme, and what keep the resolver's fingerprint out of it.
    """
    crn = source("crn")
    dh = source("dh")
    first = crn.dedupe(mb.NaiveDeduper, {"unique_fields": [crn.f("company")]})
    second = dh.dedupe(mb.NaiveDeduper, {"unique_fields": [dh.f("company")]})

    resolver = first.resolve(
        second, resolver_settings={"thresholds": {second: 0.8, first: 0.5}}
    )

    assert resolver.resolver_instance.thresholds == {0: 0.5, 1: 0.8}


def test_threshold_names_input(source: Callable[..., mb.Source]) -> None:
    """Unnamed threshold caught while the model object is still in hand."""
    crn = source("crn")
    settings = {"unique_fields": [crn.f("company")]}
    used = crn.dedupe(mb.NaiveDeduper, settings)
    unused = crn.dedupe(mb.NaiveDeduper, settings)

    with pytest.raises(ValueError, match="a model this resolver does not read"):
        used.resolve(resolver_settings={"thresholds": {unused: 0.8}})


def test_edges_keyed_by_position(
    source: Callable[..., mb.Source],
) -> None:
    """The translated thresholds are only useful if the edges arrive aligned to them."""
    crn = source("crn")
    dh = source("dh")
    first = crn.dedupe(mb.NaiveDeduper, {"unique_fields": [crn.f("company")]})
    second = dh.dedupe(mb.NaiveDeduper, {"unique_fields": [dh.f("company")]})

    resolver = first.resolve(second, resolver_settings={"thresholds": {second: 0.8}})

    seen: dict[int, int] = {}

    class Spy:
        """Wrap the methodology rather than patching it.

        `ResolverMethod` is a pydantic model and rejects undeclared attributes.
        """

        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def prepare(self, model_edges: dict[int, pl.DataFrame]) -> None:
            self.wrapped.prepare(model_edges=model_edges)

        def compute_clusters(
            self, model_edges: dict[int, pl.DataFrame]
        ) -> pl.DataFrame:
            seen.update({position: len(df) for position, df in model_edges.items()})
            return self.wrapped.compute_clusters(model_edges=model_edges)

    # Past the read-only guard: a test double, not a configuration change.
    methodology = resolver.resolver_instance
    object.__setattr__(resolver, "resolver_instance", Spy(methodology))
    resolver.collect()

    assert set(seen) == {0, 1}
    assert methodology.thresholds == {1: 0.8}
    assert seen[1] == len(second.edges())


# -- methodology versions -------------------------------------------------------------
#
# A fingerprint covers a step's settings, never the code those settings configure. A
# methodology that declares a `version` promises its output is a deterministic function
# of its settings, and is cached. One that declares none is re-keyed on every collect.
#
# These assert on what a caller reads back, rather than on how often `_execute` ran,
# so `_SUFFIX` stands in for the edit that starts the whole problem.

_SUFFIX = "-first"


class _Unversioned(mb.Transformer):
    """A transformer whose output depends on code it never declared a version for."""

    column: str

    def prepare(self, data: pl.DataFrame) -> None:
        """No baseline state is needed."""

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns((pl.col(self.column) + _SUFFIX).alias("tag"))


class _Versioned(_Unversioned):
    """The same transformer, promising its output follows from its settings."""

    version: ClassVar[int] = 1


def test_version_unset_refreshes(source: Callable[..., mb.Source]) -> None:
    """An unversioned methodology re-runs, so an edit to its code is picked up.

    The regression this pins: editing a custom methodology used to change nothing the
    cache could see, so `collect` handed back the artifact the previous body wrote.
    """
    global _SUFFIX
    crn = source("crn")
    step = crn.transform(_Unversioned(column=crn.f("company"))).collect()
    assert sorted(step.data()["tag"].to_list()) == [
        "acme-first",
        "acme-first",
        "beta-first",
    ]

    _SUFFIX = "-second"  # the edit
    step.collect()

    assert sorted(step.data()["tag"].to_list()) == [
        "acme-second",
        "acme-second",
        "beta-second",
    ]
    _SUFFIX = "-first"


def test_version_unset_refreshes_downstream(source: Callable[..., mb.Source]) -> None:
    """A versioned step below an unversioned one re-runs too.

    Its own fingerprint moves because it folds in its parent's. The step it holds is
    settled, though, so without `_cacheable` it would short-circuit on the fingerprint
    it already had and report a cache hit against a parent that just moved.
    """
    global _SUFFIX
    crn = source("crn")
    unversioned = crn.transform(_Unversioned(column=crn.f("company")))
    downstream = unversioned.clean({"shout": "upper(tag)"}).collect()
    assert sorted(downstream.data()["shout"].to_list()) == [
        "ACME-FIRST",
        "ACME-FIRST",
        "BETA-FIRST",
    ]

    _SUFFIX = "-second"
    downstream.collect()

    assert sorted(downstream.data()["shout"].to_list()) == [
        "ACME-SECOND",
        "ACME-SECOND",
        "BETA-SECOND",
    ]
    _SUFFIX = "-first"


def test_version_bump_invalidates(source: Callable[..., mb.Source]) -> None:
    """Bumping a version retires the artifacts the previous body wrote.

    A versioned methodology is cached, so this is the only way to tell matchlab that
    the code behind one changed.
    """
    global _SUFFIX
    crn = source("crn")
    first = crn.transform(_Versioned(column=crn.f("company"))).collect()
    assert sorted(first.data()["tag"].to_list()) == [
        "acme-first",
        "acme-first",
        "beta-first",
    ]

    _SUFFIX = "-second"
    unbumped = crn.transform(_Versioned(column=crn.f("company"))).collect()
    assert sorted(unbumped.data()["tag"].to_list()) == [
        "acme-first",
        "acme-first",
        "beta-first",
    ], "a versioned methodology is cached, which is what a version promises"

    with patch.object(_Versioned, "version", 2):
        bumped = crn.transform(_Versioned(column=crn.f("company"))).collect()
        assert sorted(bumped.data()["tag"].to_list()) == [
            "acme-second",
            "acme-second",
            "beta-second",
        ]

    _SUFFIX = "-first"

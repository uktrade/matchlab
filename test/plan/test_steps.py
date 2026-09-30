"""Tests for what every kind of step shares, whatever methodology it runs."""

import inspect

import polars as pl
import pytest
from pydantic import ConfigDict
from sqlalchemy import Engine

from matchlab import FromResources, Resource, Source, read_database
from matchlab.core.exceptions import ResourceError
from matchlab.core.kinds import StepKind
from matchlab.core.versioning import declared_version
from matchlab.models.dedupers.base import Deduper
from matchlab.models.linkers.base import Linker
from matchlab.resolvers.base import ResolverMethod
from matchlab.resolvers.components import Components
from matchlab.steps import Step
from matchlab.transformers import Select
from matchlab.transformers.base import Transformer


def library_steps() -> set[type[Step]]:
    """Every concrete step kind matchlab ships.

    Walked at call time, because `Step.__subclasses__()` also finds the test doubles in
    `test_progress.py` and `test_lineage.py`, and which of those exist depends on what
    has been imported. Those are filtered by module.
    """
    found: set[type[Step]] = set()
    stack: list[type[Step]] = [Step]
    while stack:
        for subclass in stack.pop().__subclasses__():
            stack.append(subclass)
            if not inspect.isabstract(subclass) and subclass.__module__.startswith(
                "matchlab."
            ):
                found.add(subclass)
    return found


def test_settings_arg_prefix() -> None:
    """A step's two methodology arguments have to share one prefix.

    `_build_methodology` quotes `{prefix}_settings` and `{prefix}_resources` when a
    field was passed in the wrong one, so an error must not send a caller to a
    parameter that does not exist.

    `__init_subclass__` derives the prefix from `*_resources`, which settles that half
    by construction. `*_settings` is still only a convention, and this is what holds
    the two names together.
    """
    assert library_steps(), "found no step kinds, so this asserted nothing"

    for step_class in library_steps():
        parameters = inspect.signature(step_class.__init__).parameters
        expected = f"{step_class._METHODOLOGY}_settings"

        assert expected in parameters, (
            f"{step_class.__name__} takes '{step_class._METHODOLOGY}_resources', so a "
            f"resource error tells callers to use '{expected}', which it does not "
            f"accept. Its arguments are: {', '.join(parameters)}."
        )


def test_two_resources_arguments() -> None:
    """A step runs one methodology, so one `*_resources` argument names it.

    Two would leave `__init_subclass__` no way to pick, and picking the first silently
    would misname the other in every error. It fails when the class is written instead.
    """
    with pytest.raises(TypeError, match="more than one resources argument"):

        class TwoMethodologies(Step):
            def __init__(
                self,
                left_resources: dict | None = None,
                right_resources: dict | None = None,
            ) -> None: ...

            @property
            def parents(self) -> tuple[Step, ...]:
                return ()


def test_name_checks() -> None:
    """A fifth kind of step must call `_check_names`, and this is what says so.

    The call is explicit in each `__init__` rather than installed by the base class, so
    this is the only thing holding it there for a kind nobody has written yet. Dropping
    the call from an existing kind is caught by the tests in `test_plan.py`, which
    cover what the check refuses; a kind that never had one is caught only here.
    """
    for step_class in library_steps():
        assert "self._check_names()" in inspect.getsource(step_class.__init__), (
            f"{step_class.__name__}.__init__ never calls `_check_names`."
        )


# -- settings vs resources -------------------------------------------------------------

# No built-in methodology takes a resource, so a test for one has to define them.


class NeedyDeduper(Deduper):
    """A deduper holding a connection it never reads through."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unique_fields: list[str]
    session: FromResources[Engine]

    def dedupe(
        self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
    ) -> pl.DataFrame:
        """Never called: this plan is inspected, not collected."""
        raise NotImplementedError


class NeedySelect(Select):
    """A transformer holding one."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    session: FromResources[Engine]


class NeedyComponents(Components):
    """A resolver methodology holding one."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: FromResources[Engine]


# Where each kind of spec keeps its serialised settings.
SETTINGS_FIELD = {
    StepKind.SOURCE: "location_settings",
    StepKind.TRANSFORM: "transformer_settings",
    StepKind.MODEL: "model_settings",
    StepKind.RESOLVER: "resolver_settings",
}


def test_resource_not_in_spec(warehouse: Engine) -> None:
    """A step keeps its resources out of the settings it hashes."""
    session = Resource("session", warehouse)
    plan = (
        read_database(
            "crn",
            sql="select pk, company from crn",
            client=Resource("wh", warehouse),
            key_field="pk",
        )
        .transform(NeedySelect, {"columns": ("crn_company",)}, {"session": session})
        .dedupe(NeedyDeduper, {"unique_fields": ["crn_company"]}, {"session": session})
        .resolve(
            resolver_class=NeedyComponents, resolver_resources={"session": session}
        )
    )

    for step in plan.lineage():
        settings = getattr(step.spec, SETTINGS_FIELD[step.kind])
        resources = step._resources()

        assert resources, f"{step.kind} bound no resource"
        assert not set(resources) & set(settings), (
            f"{step.kind} leaked a resource field into its settings"
        )


# -- which argument a field belongs in ------------------------------------------------


def test_resource_marked_field(warehouse: Engine) -> None:
    """A marked field must be a resource."""
    with pytest.raises(ResourceError, match="'client' on RelationalDB is marked"):
        Source(
            location_class="RelationalDB",
            name="crn",
            location_settings={"sql": "select pk from crn", "client": warehouse},
        )


def test_resource_unmarked_field(warehouse: Engine) -> None:
    """An unmarked field must be a setting."""
    source = read_database("crn", sql="select pk, company from crn", client=warehouse)

    with pytest.raises(ResourceError, match="'columns' on Select is a setting"):
        source.transform(Select, transformer_resources={"columns": ("crn_company",)})


def test_unknown_field(warehouse: Engine) -> None:
    """A field the methodology doesn't have says so."""
    source = read_database("crn", sql="select pk, company from crn", client=warehouse)

    with pytest.raises(ResourceError, match="Select has no field 'nonexistent'"):
        source.transform(Select, transformer_resources={"nonexistent": 1})


def library_methodologies() -> set[type]:
    """Every concrete methodology matchlab ships, whatever kind of step runs it.

    Walked the same way as `library_steps`, and filtered by module for the same
    reason: the test doubles in this suite subclass these bases too.
    """
    found: set[type] = set()
    stack: list[type] = [Transformer, Deduper, Linker, ResolverMethod]
    while stack:
        for subclass in stack.pop().__subclasses__():
            stack.append(subclass)
            if not inspect.isabstract(subclass) and subclass.__module__.startswith(
                "matchlab."
            ):
                found.add(subclass)
    return found


def test_methodology_versions_declared() -> None:
    """Every methodology matchlab ships declares a version.

    A methodology that declares none is re-keyed on every collect, taking everything
    below it with it. That is the right default for code matchlab knows nothing about,
    and the wrong one for code it ships, which would refresh a plan forever. Bump the
    version instead when a change here changes what the class computes.
    """
    assert library_methodologies(), "found no methodologies, so this asserted nothing"

    unversioned = sorted(
        methodology.__name__
        for methodology in library_methodologies()
        if declared_version(methodology) is None
    )
    assert not unversioned, (
        f"{', '.join(unversioned)} ship without a version, so any plan using one "
        "re-runs it and everything below it on every collect"
    )

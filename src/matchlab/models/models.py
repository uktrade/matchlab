"""Model — a deduper or linker producing scored candidate edges."""

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from matchlab.core.kinds import StepKind
from matchlab.core.logging import logger
from matchlab.core.schemas import SCHEMA_MODEL_EDGES
from matchlab.models import dedupers, linkers
from matchlab.models.dedupers.base import Deduper
from matchlab.models.linkers.base import Linker
from matchlab.recordstep import RecordStep
from matchlab.resources import Resource
from matchlab.specs import ModelSpec, ModelType
from matchlab.steps import Step
from matchlab.stores import Fingerprint, Store

if TYPE_CHECKING:
    # `matchlab.resolvers` imports this module
    from matchlab.record import Record
    from matchlab.resolvers import Resolver
    from matchlab.resolvers.base import ResolverMethod

_MODEL_CLASSES: dict[str, type[Linker] | type[Deduper]] = {
    **dict(inspect.getmembers(dedupers, inspect.isclass)),
    **dict(inspect.getmembers(linkers, inspect.isclass)),
}


def add_model_class(model_class: type[Linker] | type[Deduper]) -> None:
    """Register a custom deduper or linker so it can be named in a plan."""
    if not issubclass(model_class, Linker | Deduper):
        raise ValueError("The argument is not a subclass of Deduper or Linker.")
    _MODEL_CLASSES[model_class.__name__] = model_class


def _model_type_of(model_class: "type[Linker] | type[Deduper]") -> ModelType:
    """Whether a methodology class links or dedupes."""
    return ModelType.LINKER if issubclass(model_class, Linker) else ModelType.DEDUPER


def normalise_model_scores(scores: pl.DataFrame) -> pl.DataFrame:
    """Validate a methodology's raw output and cast it to `SCHEMA_MODEL_EDGES`.

    Raises `ValueError` if `scores` isn't a `DataFrame` with exactly `left_id`,
    `right_id`, and a numeric `score` in `[0.0, 1.0]`. An unordered pair can appear
    twice, once as `(a, b)` and once as `(b, a)`. When that happens, this keeps only
    the highest-scoring row and logs a warning, since a resolver expects one edge per
    pair.
    """
    if not isinstance(scores, pl.DataFrame):
        raise ValueError(f"Expected a polars DataFrame, got {type(scores)}.")

    expected_fields = set(SCHEMA_MODEL_EDGES.names)
    if set(scores.columns) != expected_fields:
        raise ValueError(f"Expected {expected_fields}.\nFound {set(scores.columns)}.")

    if scores.height == 0:
        scores = pl.DataFrame(schema=pl.Schema(SCHEMA_MODEL_EDGES))

    if not scores["score"].dtype.is_numeric():
        raise ValueError(
            "Score column must contain numeric values in the range [0.0, 1.0]."
        )

    normalised_scores = scores.with_columns(pl.col("score").cast(pl.Float32))
    invalid_scores = normalised_scores.filter(
        pl.col("score").is_null()
        | pl.col("score").is_nan()
        | (pl.col("score") < 0.0)
        | (pl.col("score") > 1.0)
    )
    if invalid_scores.height:
        min_score = normalised_scores["score"].min()
        max_score = normalised_scores["score"].max()
        raise ValueError(f"Score range misconfigured: [{min_score}, {max_score}]")

    unique_scores = (
        normalised_scores.with_columns(
            pl.concat_list(
                [pl.col("left_id").cast(pl.Utf8), pl.col("right_id").cast(pl.Utf8)]
            )
            .list.sort()
            .list.join("_")
            .alias("sorted_ids")
        )
        .sort("score", descending=True)  # sort so largest score comes first
        .unique(
            subset=["sorted_ids"], keep="first"
        )  # keep first occurrence after sorting
    ).drop("sorted_ids")
    if len(scores) != len(unique_scores):
        logger.warning("Duplicate pairs! Keeping only pairs with highest score.")

    return unique_scores.cast(pl.Schema(SCHEMA_MODEL_EDGES))


class Model(Step):
    """A step that runs one methodology (a Deduper or a Linker) over record steps."""

    kind: ClassVar[StepKind] = StepKind.MODEL

    _READ_ONLY: ClassVar[frozenset[str]] = Step._READ_ONLY | frozenset(
        {
            "left",
            "right",
            "model_class",
            "model_settings",
            "model_instance",
            "model_resources",
        }
    )

    # Settled at construction. See `Step.parents` for why these are declared.
    left: RecordStep
    right: RecordStep | None
    model_class: type[Deduper] | type[Linker]
    model_settings: dict[str, Any]
    model_resources: dict[str, Resource]
    model_instance: Deduper | Linker

    def __init__(
        self,
        left: RecordStep,
        model_class: type[Deduper] | type[Linker] | str,
        model_settings: dict[str, Any] | None = None,
        right: RecordStep | None = None,
        model_resources: dict[str, Any] | None = None,
    ) -> None:
        """Define a model.

        Args:
            left: The record step to deduplicate, or the left side of a link.
            model_class: A `Deduper`/`Linker` subclass, or its registered name.
            model_settings: That class's configuration, as a dict.
            right: The right side of a link. Omit for a deduper.
            model_resources: Resources the methodology needs that cannot be serialised,
                keyed by field name. See `matchlab.resources`.

        Raises:
            ValueError: If a linker was given no right input or a deduper was given
                one, or if the two inputs bring two different sources under one name.
            ResourceError: If a field was passed in the wrong one of `model_settings`
                and `model_resources`, or if the two inputs bring one resource name
                over two different objects.
        """
        resolved_class = (
            _MODEL_CLASSES[model_class] if isinstance(model_class, str) else model_class
        )

        if (_model_type_of(resolved_class) is ModelType.LINKER) != (right is not None):
            raise ValueError(
                "A linker requires a right input; a deduper must not have one."
            )

        instance, settings, resources = self._build_methodology(
            resolved_class, model_settings, model_resources
        )

        super().__init__()

        self._set(
            left=left,
            right=right,
            model_class=resolved_class,
            model_settings=settings,
            model_resources=resources,
            model_instance=instance,
        )
        self._prepared = False
        self._check_names()

    # -- inputs -----------------------------------------------------------------------

    @property
    def parents(self) -> tuple[RecordStep, ...]:
        """The record steps this model reads: one for a deduper, two for a linker."""
        return (self.left,) if self.right is None else (self.left, self.right)

    # -- Step contract ----------------------------------------------------------------

    @property
    def model_type(self) -> ModelType:
        """Whether this model dedupes or links.

        Derived from `model_class`, so it cannot drift from it.
        """
        return _model_type_of(self.model_class)

    def __str__(self) -> str:
        """A model is drawn with the class implementing it."""
        return f"{self.kind}({self.model_class.__name__})"

    @property
    def spec(self) -> ModelSpec:
        """The serialisable spec for this model."""
        return ModelSpec(
            model_type=self.model_type,
            model_class=self.model_class.__name__,
            model_settings=self.model_settings,
        )

    def _methodology_class(self) -> type:
        """The deduper or linker whose code this model runs."""
        return self.model_class

    def _execute(self, store: Store, fp: Fingerprint) -> None:
        left = self.left._read_cache(store)
        right = self.right._read_cache(store) if self.right else None
        self._prepared = False
        self._prepare_baseline(left, right)

        if self.model_type == ModelType.LINKER:
            scores = self.model_instance.link(left=left, right=right)
        else:
            scores = self.model_instance.dedupe(data=left)

        store.store_model(fp, normalise_model_scores(scores))

    def _prepare_baseline(self, left: pl.DataFrame, right: pl.DataFrame | None) -> None:
        """Prepare the methodology on the complete collected inputs."""
        if self.model_type == ModelType.LINKER:
            if right is None:
                raise RuntimeError("A linker requires a right input.")
            self.model_instance.prepare(left, right)
        else:
            self.model_instance.prepare(left)
        self._prepared = True

    def _ensure_prepared(self) -> None:
        """Prepare from collected inputs when collection used a cached artifact."""
        store = self._require_store()
        if self._prepared:
            return
        left = self.left._read_cache(store)
        right = self.right._read_cache(store) if self.right else None
        self._prepare_baseline(left, right)

    def _execute_record(
        self,
        store: Store,
        *,
        left: "Record | None" = None,
        right: "Record | None" = None,
    ) -> "Record":
        """Score affected sides against the prepared baseline without writing edges.

        A model is a data boundary. Its output has no current data. The edges and
        affected IDs carry the work to resolver execution.
        """
        from matchlab.record import Record  # noqa: PLC0415 - avoids a cycle

        if self._require_store() is not store:
            raise ValueError("Record execution must use this model's collected store.")
        if not self._prepared:
            raise RuntimeError(
                f"{self} is not prepared. Prepare from its collected inputs first."
            )
        if self.model_type == ModelType.DEDUPER:
            if left is None or right is not None:
                raise ValueError("A deduper requires only an affected left input.")
            if left.data is None:
                raise ValueError(
                    "A deduper requires input data before the model boundary."
                )
            scores = self.model_instance.dedupe(left.data)
        else:
            if left is None and right is None:
                raise ValueError("A linker requires at least one affected input.")
            if (
                left is not None
                and right is not None
                and (
                    left.source is not right.source
                    or not left.source_data.equals(right.source_data)
                )
            ):
                raise ValueError("Affected linker inputs must belong to one record.")
            scores = self.model_instance.link(
                left=self._record_data(left) if left is not None else None,
                right=self._record_data(right) if right is not None else None,
            )

        edges = normalise_model_scores(scores)
        inputs = [record for record in (left, right) if record is not None]
        carrier = inputs[0]
        ledger: dict[Model, pl.DataFrame] = {}
        for record in inputs:
            for model, prior in record.edges.items():
                if model in ledger and not ledger[model].equals(prior):
                    raise ValueError("Affected branches disagree on prior model edges.")
                ledger[model] = prior.clone()
        ledger[self] = edges
        ids = set().union(*(record.ids for record in inputs))
        ids.update(edges["left_id"].to_list())
        ids.update(edges["right_id"].to_list())
        return Record(
            data=None,
            ids=ids,
            source=carrier.source,
            source_data=carrier.source_data,
            edges=ledger,
        )

    @staticmethod
    def _record_data(record: "Record") -> pl.DataFrame:
        """Require current record data on a linker input."""
        if record.data is None:
            raise ValueError("A linker requires input data before the model boundary.")
        return record.data

    # -- data -------------------------------------------------------------------------

    def edges(self) -> pl.DataFrame:
        """Return this model's scored edges. Collects the plan first if needed."""
        if not self.is_collected:
            self.collect()
        store, fp = self._collected()
        return store.read_model(fp)

    # -- verbs ------------------------------------------------------------------------

    def resolve(
        self,
        *other_models: "Model",
        resolver_class: "type[ResolverMethod] | str" = "Components",
        resolver_settings: dict[str, Any] | None = None,
        resolver_resources: dict[str, Any] | None = None,
    ) -> "Resolver":
        """Resolve this model (and any others) into clusters."""
        from matchlab.resolvers import Resolver  # noqa: PLC0415 - avoids a cycle

        return Resolver(
            self,
            *other_models,
            resolver_class=resolver_class,
            resolver_settings=resolver_settings,
            resolver_resources=resolver_resources,
        )

"""Transform, a plan node that reshapes one record step with a `Transformer`.

`Transform` is to `Transformer` what `Model` is to a `Deduper`/`Linker`, the lazy plan
node that wraps a serialisable methodology, folds its configuration into a cache key,
and runs it on collect. Its single input is a `RecordStep`, so transforms chain, each
its own cached artifact.
"""

from typing import Any, ClassVar

import polars as pl

from matchlab.core.kinds import StepKind
from matchlab.recordstep import IdentifierRead, RecordStep
from matchlab.resources import Resource
from matchlab.specs import TransformSpec
from matchlab.stores import Fingerprint, Store
from matchlab.transformers.base import Transformer
from matchlab.transformers.clean import Clean
from matchlab.transformers.explode import Explode
from matchlab.transformers.group import Group
from matchlab.transformers.select import Select

_TRANSFORMER_CLASSES: dict[str, type[Transformer]] = {}


def add_transformer_class(transformer_class: type[Transformer]) -> None:
    """Register a custom transformer so it can be named in a plan (and a document)."""
    if not issubclass(transformer_class, Transformer):
        raise ValueError("The argument is not a subclass of Transformer.")
    _TRANSFORMER_CLASSES[transformer_class.__name__] = transformer_class


for _builtin in (Select, Clean, Group, Explode):
    add_transformer_class(_builtin)


class Transform(RecordStep):
    """A record step reshaped by one transformer."""

    kind: ClassVar[StepKind] = StepKind.TRANSFORM

    _READ_ONLY: ClassVar[frozenset[str]] = RecordStep._READ_ONLY | frozenset(
        {"transformer", "transformer_settings", "transformer_resources", "_input"}
    )

    # Settled at construction. See `Step.parents` for why this is declared.
    transformer: Transformer
    transformer_settings: dict[str, Any]
    transformer_resources: dict[str, Resource]

    def __init__(
        self,
        parent: RecordStep,
        transformer: Transformer | type[Transformer] | str,
        transformer_settings: dict | None = None,
        transformer_resources: dict[str, Any] | None = None,
    ) -> None:
        """Define a transform.

        Args:
            parent: The record step to reshape.
            transformer: A `Transformer` instance, or a subclass / its registered name
                to build from `transformer_settings`. The instance form is what
                `select`, `clean` and `group` use.
            transformer_settings: The configuration dict, when `transformer` is a class
                or a name. Ignored when it is already an instance.
            transformer_resources: Resources the transformer needs that cannot be
                serialised, keyed by field name. See `matchlab.resources`.

        Raises:
            ValueError: If resources are given alongside an already-built instance,
                which has nowhere left to put them.
            ResourceError: If a field was passed in the wrong one of
                `transformer_settings` and `transformer_resources`, or if this
                transformer's resources conflict with one already in the plan.
        """
        if isinstance(transformer, Transformer):
            # An instance is already built, so there is nothing left to build resources
            # into, and its settings are simply what it resolved them to.
            if transformer_resources:
                raise ValueError(
                    "Resources need a transformer class to be built into. Pass "
                    "`transformer=SomeTransformer` with `transformer_settings`, rather "
                    "than an already-built instance."
                )
            built = type(transformer).model_validate(transformer.model_dump())
            settings = built.model_dump(mode="json")
            resources: dict[str, Resource] = {}
        else:
            transformer_class = (
                _TRANSFORMER_CLASSES[transformer]
                if isinstance(transformer, str)
                else transformer
            )
            built, settings, resources = self._build_methodology(
                transformer_class, transformer_settings, transformer_resources
            )

        super().__init__()
        self._set(_input=parent)
        self._set(
            transformer=built,
            transformer_settings=settings,
            transformer_resources=resources,
        )
        self._prepared = False
        self._check_names()

    # Settled at construction. The record step this transform reshapes.
    _input: RecordStep

    @property
    def parents(self) -> tuple[RecordStep, ...]:
        """The record step this transform reshapes. See `Model.parents` on the type."""
        return (self._input,)

    # -- Step contract ----------------------------------------------------------------

    @property
    def transformer_class(self) -> type[Transformer]:
        """The class implementing this transform."""
        return type(self.transformer)

    def __str__(self) -> str:
        """A transform is drawn with the transformer implementing it."""
        return f"{self.kind}({self.transformer_class.__name__})"

    @property
    def spec(self) -> TransformSpec:
        """The serialisable spec for this transform."""
        return TransformSpec(
            transformer_class=self.transformer_class.__name__,
            transformer_settings=self.transformer_settings,
        )

    def _methodology_class(self) -> type:
        """The transformer whose code this transform runs."""
        return self.transformer_class

    def _execute(self, store: Store, fp: Fingerprint) -> None:
        records = self._input._read_cache(store)
        self._prepared = False
        self.transformer.prepare(records)
        self._prepared = True
        reshaped = self.transformer.apply(records)

        # The built-ins refuse an `id` output when they are built. This catches a custom
        # transformer that writes one anyway.
        if "id" not in reshaped.columns:
            raise ValueError(
                f"{self.transformer_class.__name__} dropped the `id` column. A "
                "transformer reshapes a record step's columns and must pass `id` "
                "through, since it is the grouping a model matches on."
            )
        if reshaped["id"].dtype != records["id"].dtype:
            raise ValueError(
                f"{self.transformer_class.__name__} replaced the `id` column, which "
                f"held {records['id'].dtype} and now holds {reshaped['id'].dtype}. "
                "`id` is derived from record content and must pass through untouched."
            )

        store.store_transform(fp, reshaped)

    def _ensure_prepared(self) -> None:
        """Prepare from the collected input when collection used a cached artifact."""
        store = self._require_store()
        if self._prepared:
            return
        self.transformer.prepare(self._input._read_cache(store))
        self._prepared = True

    # -- RecordStep contract ------------------------------------------------------

    def _read_cache(self, store: Store) -> pl.DataFrame:
        if self._fp is None:  # collect orders upstream first
            raise RuntimeError(
                "This transform has not been collected. Call collect() first."
            )
        return store.read_transform(self._fp)

    @property
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        """A transform reads the same records as its input, so the reads delegate up."""
        return self._input._identifier_reads

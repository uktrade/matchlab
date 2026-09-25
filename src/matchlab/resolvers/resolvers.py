"""Resolver collapses model edges into clusters and materialises the resolver output."""

from typing import Any, ClassVar, Self

import polars as pl

from matchlab.core.dataframes import qualify
from matchlab.core.exceptions import StepNotFound
from matchlab.core.kinds import StepKind
from matchlab.core.resolver_output import materialise_resolver_output
from matchlab.models import Model
from matchlab.recordstep import IdentifierRead, RecordStep, build_record_step
from matchlab.resolvers.base import ResolverMethod
from matchlab.resolvers.components import Components
from matchlab.resources import Resource
from matchlab.sources import Source
from matchlab.specs import ResolverSpec
from matchlab.stores import Fingerprint, Store

_RESOLVER_CLASSES: dict[str, type[ResolverMethod]] = {}


def add_resolver_class(resolver_class: type[ResolverMethod]) -> None:
    """Register a resolver methodology so it can be named in a plan."""
    if not issubclass(resolver_class, ResolverMethod):
        raise ValueError("The argument is not a subclass of ResolverMethod.")
    _RESOLVER_CLASSES[resolver_class.__name__] = resolver_class


add_resolver_class(Components)


class Resolver(RecordStep):
    """Clusters computed over one or more models' edges.

    A resolver is also a `RecordStep`. Read as records, its rows are the sources it
    covers, with `id` set to the entity root, so matching on top of a resolver is how a
    plan layers (`deduped.link(dh, …)`). The resolver output it stores is a separate
    artifact, and the record step itself is derived on read.
    """

    kind: ClassVar[StepKind] = StepKind.RESOLVER

    _READ_ONLY: ClassVar[frozenset[str]] = RecordStep._READ_ONLY | frozenset(
        {
            "_inputs",
            "resolver_class",
            "resolver_settings",
            "resolver_instance",
            "resolver_resources",
        }
    )

    # Settled at construction. See `Step.parents` for why these are declared.
    _inputs: tuple[Model, ...]
    resolver_class: type[ResolverMethod]
    resolver_settings: dict[str, Any]
    resolver_resources: dict[str, Resource]
    resolver_instance: ResolverMethod

    def __init__(
        self,
        *models: Model,
        resolver_class: type[ResolverMethod] | str = "Components",
        resolver_settings: dict[str, Any] | None = None,
        resolver_resources: dict[str, Any] | None = None,
    ) -> None:
        """Define a resolver.

        Args:
            *models: The models whose edges to resolve. At least one.
            resolver_class: A `ResolverMethod` subclass or its registered name.
                Defaults to connected components.
            resolver_settings: Settings for that methodology.
            resolver_resources: Resources the methodology needs that cannot be
                serialised, keyed by field name. See `matchlab.resources`.

        Raises:
            ValueError: If an input is not a model, if there are none, or if the models
                bring two different sources under one name.
            ResourceError: If a field was passed in the wrong one of
                `resolver_settings` and `resolver_resources`, or if the models bring
                one resource name over two different objects.
        """
        deduped: list[Model] = []
        for model in models:
            if not isinstance(model, Model):
                raise ValueError(f"Resolver inputs must be models, got {type(model)}")
            if not any(model is seen for seen in deduped):
                deduped.append(model)
        if not deduped:
            raise ValueError("A resolver needs at least one model")

        super().__init__()
        self._set(_inputs=tuple(deduped))

        resolved = (
            _RESOLVER_CLASSES[resolver_class]
            if isinstance(resolver_class, str)
            else resolver_class
        )
        # `_positions` runs first: a setting may name one of this resolver's inputs by
        # object, and only the built methodology's own position form belongs in a spec.
        instance, settings, resources = self._build_methodology(
            resolved,
            {
                field: self._positions(field, value)
                for field, value in (resolver_settings or {}).items()
            },
            resolver_resources,
        )

        self._set(
            resolver_class=resolved,
            resolver_settings=settings,
            resolver_resources=resources,
            resolver_instance=instance,
        )
        self._prepared = False
        self._check_names()

    @property
    def parents(self) -> tuple[Model, ...]:
        """The models this resolver reads."""
        return self._inputs

    def _positions(self, field: str, value: Any) -> Any:  # noqa: ANN401 - any setting
        """Replace `Model` keys in a setting with the position of that input.

        A setting that points at one of this resolver's inputs, such as a per-model
        threshold, is written as `{model: 0.9}`. That holds the object you already
        have, rather than retyping its name. Here is the only place that can be
        translated, because here is where both the models and the order they were
        given in are known.

        Positions rather than names, because a name is not identity. Renaming a model
        would move this resolver's fingerprint without changing a byte of its output.
        Reordering the inputs *does* reassign thresholds, but that is not inconsistent.
        Reordering already changes the fingerprint anyway, since parents fold in in
        order, making this a genuinely different resolver.

        Raises:
            ValueError: If a setting names a model that is not an input.
        """
        if not isinstance(value, dict):
            return value

        position = {id(model): index for index, model in enumerate(self.parents)}
        translated: dict[Any, Any] = {}
        for key, setting in value.items():
            if not isinstance(key, Model):
                translated[key] = setting
                continue
            if id(key) not in position:
                raise ValueError(
                    f"'{field}' has an entry for a model this resolver does not read. "
                    f"It resolves {len(self.parents)} model(s), and a setting may "
                    "only point at one of those."
                )
            translated[position[id(key)]] = setting
        return translated

    # -- Step contract ----------------------------------------------------------------

    def __str__(self) -> str:
        """A resolver is drawn with the methodology implementing it."""
        return f"{self.kind}({self.resolver_class.__name__})"

    @property
    def spec(self) -> ResolverSpec:
        """The serialisable spec for this resolver."""
        return ResolverSpec(
            resolver_class=self.resolver_class.__name__,
            resolver_settings=self.resolver_settings,
        )

    # -- publishing -------------------------------------------------------------------

    def publish(self, label: str, overwrite: bool = False) -> Self:
        """Point a label at this resolver's output, so it can be found without the plan.

        Publishing is an act, not a property of the plan. A label changes nothing
        about what gets computed, and there is nothing to point at until the resolver
        output exists. Publish after collection, for example
        `resolver.collect().publish("x")`. A plan that is never published is still
        perfectly runnable, just unlabelled.

        A *label* rather than a name, because a name is something else here. A
        source's name is part of its output, prefixing every column it contributes. A
        label belongs to the store, and points at whichever resolver output you last
        aimed it at.

        Re-publishing the same label for the same resolver output is a no-op, so
        re-running an unchanged pipeline is safe. Aiming an existing label at a
        *different* resolver output needs `overwrite=True`, because that is how you
        lose track of what a label used to mean.

        Args:
            label: The label to publish under.
            overwrite: Move the label if it already points somewhere else.

        Returns:
            This resolver, so it chains off `collect()`.

        Raises:
            RuntimeError: If this resolver has not been collected.
            ValueError: If `label` already points at a different resolver output and
                `overwrite` is not set.
        """
        store, fp = self._collected()
        existing = store.find(label)
        if existing is not None and existing != fp and not overwrite:
            raise ValueError(
                f"The label '{label}' already points at a different resolver output "
                f"({existing.hex()[:8]}). Pass overwrite=True to move it, or publish "
                "under another label."
            )
        store.publish(label, fp)
        return self

    # -- Step contract ----------------------------------------------------------------

    def _methodology_class(self) -> type:
        """The resolver methodology whose code this resolver runs."""
        return self.resolver_class

    def _execute(self, store: Store, fp: Fingerprint) -> None:
        edges = {
            position: store.read_model(model._fp)
            for position, model in enumerate(self.parents)
        }
        self._prepared = False
        self.resolver_instance.prepare(model_edges=edges)
        self._prepared = True
        clusters = self.resolver_instance.compute_clusters(model_edges=edges)

        # materialise_resolver_output carries forward every leaf reachable through
        # this resolver's inputs, including records no model formed an edge over
        # (the merge-forward / fall-through requirement).

        # Deduplicate what is read. Linking every pair of n sources gives n(n-1)
        # (model, record_step) pairs, but only a handful of distinct readings between
        # them, since they share an upstream resolver and cover the same sources.
        # Asking per pair would otherwise repeat the same query a quadratic number of
        # times. `dict.fromkeys` dedupes while keeping lineage order, so the record
        # step is built the same way every run.
        reads = dict.fromkeys(
            read
            for model in self.parents
            for record_step in model.parents
            for read in record_step._identifier_reads
        )
        upstream = pl.concat(
            [store.read_identifiers(*read) for read in reads],
            how="vertical",
        ).unique()

        # Record which source artifacts this resolver output covers. A resolver's
        # output names its sources, but a store can hold several generations of a
        # name. That is what lets it be read back without the plan that built it.
        # Those names are data, tagging which source each row came from. They are
        # nothing to do with publishing, which is `publish()` and happens after this.
        store.store_resolver(
            fp=fp,
            resolver_output=materialise_resolver_output(clusters, upstream),
            sources={source.name: source._fp for source in self.sources},
        )

    def _ensure_prepared(self) -> None:
        """Prepare from collected model edges after a cache hit."""
        store = self._require_store()
        if self._prepared:
            return
        edges = {
            position: store.read_model(model._fp)
            for position, model in enumerate(self.parents)
        }
        self.resolver_instance.prepare(model_edges=edges)
        self._prepared = True

    # -- data -------------------------------------------------------------------------

    @property
    def sources(self) -> tuple[Source, ...]:
        """The sources reachable through this resolver, in lineage order."""
        return tuple(step for step in self.lineage() if isinstance(step, Source))

    def entities(self, sources: list[str] | None = None) -> pl.DataFrame:
        """Return `(root, leaf, key, source)`. Collects the plan first if needed.

        One row per source record. `root` is the entity it resolved to, `leaf` its
        content-addressed record identity, and `key` its key in the original source.

        Args:
            sources: Restrict to these source names. Defaults to all of them.

        Raises:
            StepNotFound: If `sources` names something this resolver does not read.
        """
        if not self.is_collected:
            self.collect()
        store, fp = self._collected()
        resolver_output = store.read_resolver(fp)

        if sources is None:
            return resolver_output

        return resolver_output.filter(pl.col("source").is_in(self._named(sources)))

    def _named(self, sources: list[str]) -> list[str]:
        """Narrow this resolver's sources to those named, in lineage order.

        A name that isn't one of them is an error, not an empty result. Asking
        for a source this resolver never read is a mistake in the caller, and silently
        returning nothing is how it stays one.
        """
        available = [source.name for source in self.sources]
        unknown = [name for name in sources if name not in available]
        if unknown:
            raise StepNotFound(
                f"This resolver does not read {', '.join(repr(n) for n in unknown)}. "
                f"It covers: {', '.join(available)}."
            )
        if not sources:
            raise StepNotFound("No compatible source was found")
        return [name for name in available if name in sources]

    def get_lookup(self, sources: list[str] | None = None) -> pl.DataFrame:
        """Return `root` plus one qualified-key column per source.

        The wide form of `entities()` is a row per entity, joined across sources, with
        nulls where a source has no record in it. This is the table you hand to
        someone who just wants their identifiers lined up.

        Args:
            sources: Restrict to these source names. Defaults to all of them.
        """
        resolver_output = self.entities(sources)
        names = self._named(sources) if sources is not None else None

        # This is never empty. A resolver always has a source, and `_named` raises
        # rather than narrow to none, so there is always a table to start the join
        # from.
        columns = [
            resolver_output.filter(pl.col("source") == source.name).select(
                "root", pl.col("key").alias(source.qualified_key)
            )
            for source in self.sources
            if names is None or source.name in names
        ]

        lookup = columns[0]
        for keys in columns[1:]:
            lookup = lookup.join(keys, on="root", how="full", coalesce=True)
        return lookup

    def leaf_sets(self, sources: list[str] | None = None) -> list[list[int]]:
        """Return each entity as a sorted list of record identities.

        Cluster IDs are dropped, so two resolver outputs over the same records can be
        compared by structure alone. Leaves are deduplicated, so each one appears once
        per key it holds.

        Args:
            sources: Restrict to these source names. Defaults to all of them.
        """
        groups = self.entities(sources).group_by("root").agg("leaf")["leaf"].to_list()
        return [sorted(set(group)) for group in groups]

    def view_entity(self, root: int, merge_fields: bool = False) -> pl.DataFrame:
        """Return the stored rows for every record in one entity.

        Values come from the extract cached when each source was collected, not from a
        fresh warehouse read. That is the data the matching actually saw, the same
        rows the reviewer puts on screen, and it means looking at an entity needs no
        warehouse connection.

        Args:
            root: The entity to look at.
            merge_fields: Collapse the source qualifier on index fields, so a field two
                sources share lands in one column. Key fields stay qualified, since
                they are what tells you which source a row came from.

        Raises:
            KeyError: If no source has a record in that entity.
        """
        store, fp = self._collected()
        resolver_output = self.entities().filter(pl.col("root") == root)
        stored = store.resolver_output_sources(fp)

        rows: list[pl.DataFrame] = []
        key_columns: list[str] = []
        for source in self.sources:
            keys = resolver_output.filter(pl.col("source") == source.name)["key"]
            if keys.len() == 0:
                continue

            records, qualified_key = store.read_source_records(
                stored[source.name], source.name, keys
            )
            if merge_fields:
                prefix = qualify(source.name)
                records = records.rename(
                    {
                        column: column.removeprefix(prefix)
                        for column in records.columns
                        if column != qualified_key
                    }
                )

            key_columns.append(qualified_key)
            rows.append(records)

        if not rows:
            raise KeyError(f"Entity {root} not available")

        # Coerce fields to their common super-type, and lead with the keys.
        joined = pl.concat(rows, how="diagonal_relaxed")
        remaining = [column for column in joined.columns if column not in key_columns]
        return joined.select(*key_columns, *remaining)

    def lookup_key(
        self, from_source: str, to_sources: list[str], key: str
    ) -> dict[str, list[str]]:
        """Find the keys in `to_sources` that resolve to the same entity as `key`.

        Args:
            from_source: The source `key` belongs to.
            to_sources: The sources to find matching keys in.
            key: The key to look up.

        Returns:
            Source name → matching keys, including `from_source` itself.
        """
        resolver_output = self.entities()
        origin = resolver_output.filter(
            (pl.col("source") == from_source) & (pl.col("key") == key)
        )
        if origin.height == 0:
            raise StepNotFound(f"Key '{key}' not found in source '{from_source}'")

        cluster = resolver_output.filter(pl.col("root") == origin["root"][0])

        def keys_in(source_name: str) -> list[str]:
            return cluster.filter(pl.col("source") == source_name)["key"].to_list()

        return {from_source: keys_in(from_source)} | {
            target: keys_in(target) for target in to_sources
        }

    # -- RecordStep contract -------------------------------------------------------

    def _read_cache(self, store: Store) -> pl.DataFrame:
        """Derive the table, this resolver's sources with `id` set to the entity root.

        A re-derivable join over artifacts already stored (the source extracts and this
        resolver's output), so nothing is materialised for it.
        """
        if self._fp is None:  # collect orders upstream first
            raise RuntimeError(
                "This resolver has not been collected. Call collect() first."
            )
        return build_record_step(store, self.sources, self)

    @property
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        return tuple((source._fp, source.name, self._fp) for source in self.sources)

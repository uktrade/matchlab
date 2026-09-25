"""RecordStep, the records a model matches over, and the verbs that build a plan.

A `RecordStep` is a step whose artifact is a table of records carrying an `id`.
`Source`, `Resolver` and `Transform` are the three kinds. On a `Source`, `id` is the
leaf. On a `Resolver`, `id` is the entity root. Every `RecordStep` chains the same
verbs (`select`, `clean`, `group`, `transform`, `dedupe`, `link`). A `Model` reads a
`RecordStep` and yields edges, not records.

`RecordStep` is not user-facing. Users hold a `Source`, a `Resolver` or a `Transform`
and call verbs on it. Each concrete kind supplies only how it materialises
(`_read_cache`) and which source rows it stands for (`_identifier_reads`).

A record step does not store `identifiers()`, the `(id, source, key, leaf)` mapping a
downstream resolver needs. Reshaping cannot change it, so it is read back from the
source leaves and the upstream resolver output instead.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

import polars as pl

from matchlab.core.dataframes import (
    DataFrameClass,
    DataFrameType,
    qualify,
    to_dataframe,
)
from matchlab.core.sql import SQLExpression
from matchlab.steps import Step
from matchlab.stores import Fingerprint, Store

if TYPE_CHECKING:
    # Each of these modules imports this one. Only for annotations, so importing the
    # real classes (rather than typing them `Any`) costs nothing at runtime.
    from matchlab.models import Model
    from matchlab.models.dedupers.base import Deduper
    from matchlab.models.linkers.base import Linker
    from matchlab.resolvers import Resolver
    from matchlab.sources import Source
    from matchlab.transformers import Transform, Transformer

# What one source's identifiers are read with, `(source_fp, source_name, resolver_fp)`,
# the arguments of `Store.read_identifiers`. Deduplicating these is deduplicating the
# queries, which is why they travel as a value rather than as a call.
IdentifierRead = tuple[Fingerprint | None, str, Fingerprint | None]


def build_record_step(
    store: Store,
    sources: "tuple[Source, ...]",
    resolver: "Resolver | None",
) -> pl.DataFrame:
    """Assemble the `id` + qualified-columns table from stored extracts and identifiers.

    Every source's extract is prefixed with the source name (`company` → `crn_company`)
    and joined to its identifiers, so each row gains the `id` it belongs to. That is the
    record's leaf when read directly, or the entity root when read through `resolver`.
    Several sources are concatenated diagonally, each row carrying its own source's
    columns and nulls for the rest. This is what both `Source` and `Resolver` read.
    """
    resolver_fp = resolver._fp if resolver is not None else None
    identifiers = pl.concat(
        [
            store.read_identifiers(source._fp, source.name, resolver_fp)
            for source in sources
        ],
        how="vertical",
    )

    extracts = [
        store.read_source_extract(source._fp)
        .select(pl.all().name.prefix(qualify(source.name)))
        .with_columns(pl.lit(source.name).alias("source"))
        .rename({source.qualified_key: "key"})
        for source in sources
    ]

    return (
        pl.concat(extracts, how="diagonal")
        .join(identifiers.drop("leaf"), how="inner", on=("source", "key"))
        .drop("source", "key")
    )


class RecordStep(Step):
    """A step whose artifact is a table of records a model matches over."""

    # -- RecordStep contract -------------------------------------------------------

    @abstractmethod
    def _read_cache(self, store: Store) -> pl.DataFrame:
        """Return this record step's records from wherever they are materialised."""
        ...

    @property
    @abstractmethod
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        """The `Store.read_identifiers` arguments this record step's data comes from.

        One per source, naming what is read rather than reading it. Two record steps
        built from the same source and resolver read identical rows, even if they then
        reshape those rows differently. A downstream resolver wants the *set* of these
        reads across every record step feeding it, so naming them lets it ask once
        instead of once per record step.
        """
        ...

    def identifiers(self, store: Store, ids: set[int] | None = None) -> pl.DataFrame:
        """Return `(id, source, key, leaf)` for this record step's IDs.

        `id` is the resolver's entity root when reading through one, otherwise the
        source leaf. This is the upstream resolver output a downstream resolver needs to
        carry every reachable leaf forward, including records no model matched.
        With `ids=None`, the read includes every identifier. An empty set returns none.
        """
        readings = []
        for source_fp, source_name, resolver_fp in self._identifier_reads:
            if source_fp is None:
                raise RuntimeError("This record step has not been collected.")
            readings.append(
                store.read_identifiers(source_fp, source_name, resolver_fp, ids=ids)
            )
        return pl.concat(readings, how="vertical")

    def data(self, return_type: DataFrameType = DataFrameType.POLARS) -> DataFrameClass:
        """Return this record step's records, collecting the plan first if needed."""
        if not self.is_collected:
            self.collect()
        return to_dataframe(self._read_cache(self._require_store()), return_type)

    # -- verbs ------------------------------------------------------------------------

    def transform(
        self,
        transformer: "Transformer | type[Transformer] | str",
        transformer_settings: dict | None = None,
        transformer_resources: dict | None = None,
    ) -> "Transform":
        """Reshape this record step with a transformer."""
        from matchlab.transformers import Transform  # noqa: PLC0415 - avoids a cycle

        return Transform(self, transformer, transformer_settings, transformer_resources)

    def select(self, *columns: str) -> "Transform":
        """Keep only the named columns, plus `id`."""
        from matchlab.transformers import Select  # noqa: PLC0415 - avoids a cycle

        return self.transform(Select(*columns))

    def clean(self, cleaning: dict[str, SQLExpression]) -> "Transform":
        """Derive columns with DuckDB SQL, keeping the rest."""
        from matchlab.transformers import Clean  # noqa: PLC0415 - avoids a cycle

        return self.transform(Clean(cleaning=cleaning))

    def group(self, aggregates: dict[str, SQLExpression]) -> "Transform":
        """Collapse each `id` to one row using aggregate SQL."""
        from matchlab.transformers import Group  # noqa: PLC0415 - avoids a cycle

        return self.transform(Group(aggregates=aggregates))

    def dedupe(
        self,
        model_class: "type[Deduper] | str",
        model_settings: dict | None = None,
        model_resources: dict | None = None,
    ) -> "Model":
        """Deduplicate this record step."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(
            left=self,
            model_class=model_class,
            model_settings=model_settings,
            model_resources=model_resources,
        )

    def link(
        self,
        other: "RecordStep",
        model_class: "type[Linker] | str",
        model_settings: dict | None = None,
        model_resources: dict | None = None,
    ) -> "Model":
        """Link this record step to another. A `Source` is one, needing no wrapping."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(
            left=self,
            right=other,
            model_class=model_class,
            model_settings=model_settings,
            model_resources=model_resources,
        )

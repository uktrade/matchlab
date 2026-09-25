"""Transient record data for one online match.

A Record follows a collected source's branch without reading from that source.
Its source evidence stays intact while transforms reshape the current data.
"""

from collections.abc import Mapping
from types import MappingProxyType
from uuid import uuid4

import polars as pl
from pydantic import BaseModel, ConfigDict, PrivateAttr

from matchlab.core.dataframes import qualify
from matchlab.core.hash import HashMethod, hash_rows
from matchlab.core.resolver_output import leaf_id
from matchlab.models import Model
from matchlab.sources import Source
from matchlab.stores import Store


class Record(BaseModel):
    """Carry affected data and original source evidence through one plan branch.

    `ids` belong to the current step's ID space. They need not equal the IDs in
    `data`. A model result has no current record data. Public dataframe access returns
    a copy, and the ID set and edge ledger are immutable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    ids: frozenset[int]
    source: Source

    _data: pl.DataFrame | None = PrivateAttr(default=None)
    _source_data: pl.DataFrame = PrivateAttr()
    _edges: Mapping[Model, pl.DataFrame] = PrivateAttr()
    _sealed: bool = PrivateAttr(default=False)

    def __init__(
        self,
        data: pl.DataFrame | None,
        ids: set[int] | frozenset[int],
        source: Source,
        source_data: pl.DataFrame,
        edges: Mapping[Model, pl.DataFrame] | None = None,
    ) -> None:
        """Build an isolated record value from its branch-local state."""
        super().__init__(ids=frozenset(ids), source=source)
        self._data = data.clone() if data is not None else None
        self._source_data = source_data.clone()
        self._edges = MappingProxyType(
            {model: frame.clone() for model, frame in (edges or {}).items()}
        )
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        """Keep private dataframe state fixed after construction."""
        if name in {
            "_data",
            "_source_data",
            "_edges",
        } and self.__pydantic_private__.get("_sealed", False):
            raise AttributeError("Record is immutable.")
        super().__setattr__(name, value)

    @property
    def data(self) -> pl.DataFrame | None:
        """Return a copy of current data, or `None` after a model boundary."""
        return self._data.clone() if self._data is not None else None

    @property
    def source_data(self) -> pl.DataFrame:
        """Return a copy of the original source evidence."""
        return self._source_data.clone()

    @property
    def edges(self) -> Mapping[Model, pl.DataFrame]:
        """Return an immutable ledger with independent edge frames."""
        return MappingProxyType(
            {model: frame.clone() for model, frame in self._edges.items()}
        )

    def with_data(self, data: pl.DataFrame) -> "Record":
        """Return this record's branch with reshaped current data."""
        return Record(
            data=data,
            ids=self.ids,
            source=self.source,
            source_data=self._source_data,
            edges=self._edges,
        )

    @classmethod
    def from_input(
        cls, source: Source, values: Mapping[str, object], store: Store
    ) -> "Record":
        """Build a record from caller data using the collected extract's schema.

        The source names the branch and stored extract. Its location is not read.
        The generated key is needed for later source evidence, but it does not
        contribute to the content-addressed leaf.
        """
        _, fp = source._collected()
        extract = store.read_source_extract(fp)
        fields = set(extract.columns) - {source.key_field}
        supplied = set(values)
        missing = sorted(fields - supplied)
        extra = sorted(supplied - fields)
        if missing or extra:
            raise ValueError(
                f"Record for source '{source.name}' has missing fields {missing} "
                f"and unexpected fields {extra}. Supply exactly the non-key "
                "columns of its collected extract."
            )

        columns = {}
        for name in extract.columns:
            value = (
                f"__matchlab_online_{uuid4().hex}"
                if name == source.key_field
                else values[name]
            )
            try:
                columns[name] = pl.Series(name, [value]).cast(
                    extract.schema[name], strict=True
                )
            except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
                raise ValueError(
                    f"Record for source '{source.name}' has an invalid value "
                    f"for '{name}', expected {extract.schema[name]}."
                ) from exc

        source_data = pl.DataFrame(columns)
        row_hash = hash_rows(
            source_data,
            columns=sorted(fields),
            method=HashMethod.XXH3_128,
        )
        leaf = int(
            pl.DataFrame({"hash": row_hash})
            .select(leaf_id(pl.col("hash")).alias("leaf"))["leaf"]
            .item()
        )
        data = (
            source_data.select(pl.all().name.prefix(qualify(source.name)))
            .drop(source.qualified_key)
            .with_columns(pl.lit(leaf, dtype=pl.UInt64).alias("id"))
        )
        return cls(
            data=data,
            ids={leaf},
            source=source,
            source_data=source_data,
        )

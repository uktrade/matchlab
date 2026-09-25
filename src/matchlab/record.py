"""Transient record data for one online match.

A Record follows a collected source's branch without reading from that source.
Its source evidence stays intact while transforms reshape the current data.
"""

from collections.abc import Mapping
from uuid import uuid4

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from matchlab.core.dataframes import qualify
from matchlab.core.hash import HashMethod, hash_rows
from matchlab.core.resolver_output import leaf_id
from matchlab.models import Model
from matchlab.sources import Source
from matchlab.stores import Store


class Record(BaseModel):
    """Carry affected data and original source evidence through one plan branch.

    `ids` belong to the current step's ID space. They need not equal the IDs in
    `data`. Each branch owns its ID set and model-edge ledger.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data: pl.DataFrame
    ids: set[int]
    source: Source
    source_data: pl.DataFrame
    edges: dict[Model, pl.DataFrame] = Field(default_factory=dict)

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

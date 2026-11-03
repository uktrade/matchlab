"""Base class for transformer methodologies, and the DuckDB query runner some share.

A `Transformer` is a pure function of a record step, called by a `Transform` step's
`apply()` on every collect. Both input and output must carry `id`, the grouping every
downstream model and resolver reads.

Transformers are declarative and serialisable. Their fields are the configuration a
`Transform` folds into its spec and cache key, so keep them to plain data, with no
callables, exactly as a `Deduper`'s settings are.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict

from matchlab.core.sql import SQLQuery


class Transformer(BaseModel, ABC):
    """Base contract every transformer implements.

    Concrete transformers (`Select`, `Clean`, `Group`, `Explode`) carry their
    configuration as flat fields, so `MyTransformer(...)` reads naturally, and
    `model_dump(mode="json")` is the whole of its serialisation.

    Frozen, and `extra="forbid"` so a mistyped setting is refused rather than silently
    ignored — which would leave the transform running a default under a fingerprint that
    never mentioned the field.

    Every field is a setting unless marked `matchlab.resources.FromResources`. A
    fingerprint ignores a resource, so a marked field must not change the output.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump it to retire every artifact the previous body wrote.
    # Left unset, the step re-runs on every collect, and so does everything below it.
    # See `matchlab.core.versioning`.
    version: ClassVar[int | None] = None

    @abstractmethod
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return `data` reshaped. Both the input and the output carry `id`."""
        ...


def reject_id_output(names: Iterable[str]) -> None:
    """Refuse `id` as a column a transformer writes.

    `id` is the grouping every model matches on, derived by matchlab from record
    content. A transformer that assigns to it changes which records a model treats as
    the same, and nothing downstream can tell that happened: the fingerprint covers the
    expression, not what the expression displaced.

    Raises:
        ValueError: If `id` is among the output names.
    """
    if "id" in names:
        raise ValueError(
            "`id` is not a column you can write. It is the grouping every model "
            "matches on, derived from record content, so replacing it would silently "
            "change which records count as the same. Give the expression another name."
        )


def run_sql(query: SQLQuery, data: pl.DataFrame) -> pl.DataFrame:
    """Execute one DuckDB query against `data`, registered as `data`."""
    with duckdb.connect(":memory:") as connection:
        connection.register("data", data)
        return connection.sql(query).pl()

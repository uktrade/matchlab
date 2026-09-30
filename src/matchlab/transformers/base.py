"""Base class for transformer methodologies and their shared DuckDB query runner.

A `Transform` prepares its transformer on the full input before applying it. Later
calls may apply it to affected rows using that prepared state. Both input and output
must carry `id`, the grouping every downstream model and resolver reads.

Transformers are declarative and serialisable. Their fields are the configuration a
`Transform` folds into its spec and cache key, so keep them to plain data, with no
callables, exactly as a `Deduper`'s settings are.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

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

    def prepare(self, data: pl.DataFrame) -> "PreparedTransformer":
        """Return runtime preparation from the full baseline without changing self.

        Override this to return derived state; the default needs no setup.
        """
        return PreparedTransformer(methodology=self)

    @abstractmethod
    def apply(
        self,
        prepared_state: Any,  # noqa: ANN401 - opaque state owned by the author
        data: pl.DataFrame,
        *,
        baseline: pl.DataFrame,
    ) -> pl.DataFrame:
        """Reshape supplied data using prepared state, preserving `id`."""
        ...


class PreparedTransformer(BaseModel):
    """In-memory preparation for one Transformer, independent of its specification.

    State can contain arbitrary backend objects and need not be serialisable.
    Baseline data is supplied explicitly to each action by the caller.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    methodology: Transformer
    state: Any = None

    def apply(self, data: pl.DataFrame, *, baseline: pl.DataFrame) -> pl.DataFrame:
        """Reshape supplied rows using this preparation and the explicit baseline."""
        return self.methodology.apply(self.state, data, baseline=baseline)


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
        return connection.execute(query).pl()

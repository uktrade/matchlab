"""Base classes for resolver methodologies."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, Final

import polars as pl
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

SCHEMA_CLUSTERS: Final[pa.Schema] = pa.schema(
    [
        ("parent_id", pa.uint64()),
        ("child_id", pa.uint64()),
    ]
)
"""A resolver methodology's cluster assignments take this shape."""


class ResolverType(StrEnum):
    """Enumeration of supported resolver methodology types.

    This lives with `ResolverMethod`, not in `matchlab.specs`, because it is not part
    of any spec. A `ResolverSpec` identifies a methodology only by its registered
    `resolver_class`, so a `ResolverType` never reaches a fingerprint or a document.
    It is just a class attribute methodologies use to declare which family they
    belong to, and nothing more.
    """

    COMPONENTS = "components"


class ResolverMethod(BaseModel, ABC):
    """Base class for resolver methodologies.

    A field that points at one of the resolver's inputs refers to it by **position**.
    `Resolver` accepts the `Model` object at the API and translates it, so nothing here
    holds a plan node: a methodology stays a pure function of edges and numbers.

    Frozen and `extra="forbid"`, so a mistyped setting is refused rather than silently
    ignored, and a built methodology cannot drift from the fingerprint naming it.

    Every field is a setting unless marked `matchlab.resources.FromResources`. A
    fingerprint ignores a resource, so a marked field must not change the clustering.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump it to retire every artifact the previous body wrote.
    # Left unset, the step re-runs on every collect, and so does everything below it.
    # See `matchlab.core.versioning`.
    version: ClassVar[int | None] = None

    resolver_type: ClassVar[ResolverType]

    @abstractmethod
    def prepare(self, model_edges: Mapping[int, pl.DataFrame]) -> None:
        """Prepare from the complete edges, keyed by model input position."""
        ...

    @abstractmethod
    def compute_clusters(self, model_edges: Mapping[int, pl.DataFrame]) -> pl.DataFrame:
        """Compute clusters from supplied edges using prepared state.

        Args:
            model_edges: Input position to that model's edges, conforming to
                SCHEMA_MODEL_EDGES. Positions index the resolver's inputs, in the
                order they were given, and are what per-model settings key by.

        Returns:
            A Polars DataFrame which conforms to SCHEMA_CLUSTERS
        """
        ...

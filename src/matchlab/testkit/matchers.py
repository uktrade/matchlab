"""Matchers that already know the answer.

A perfect matcher makes no mistakes, so pointing one at generated data leaves the *plan*
as the only variable. That is what makes an end-to-end assertion mean something. Build
them through `LinkedSources.dedupe()` / `.link()`. They are public here only so the
registry can name them.

The subtlety is how a matcher identifies a record. Pre-generating edges between record
IDs does not work, because matchlab derives cluster IDs by content-hashing rows at
collect time. The IDs a model actually receives are unknowable when a fixture is built,
which is why an `AnswerKey` is keyed by the row's *values* instead. At match time the
matcher reads those columns out of the record step it was handed, looks up each row's
true entity, and emits edges between whatever IDs are actually present. That makes it
independent of how identity is assigned, and lets it survive any future change to ID
minting.
"""

from collections.abc import Iterable
from itertools import combinations
from typing import ClassVar

import polars as pl
from pydantic import BaseModel, ConfigDict

from matchlab.core.hash import HASH_FUNC
from matchlab.core.schemas import SCHEMA_MODEL_EDGES
from matchlab.models import add_model_class
from matchlab.models.dedupers.base import Deduper
from matchlab.models.linkers.base import Linker
from matchlab.testkit.entities import TrueEntity
from matchlab.testkit.sources import GeneratedSource

# Registered answer keys, by content hash. Only ever added to. A testkit process is
# short-lived and the keys are small.
_ANSWERS: dict[str, "AnswerKey"] = {}


class AnswerKey(BaseModel):
    """The sheet a perfect matcher looks answers up in: row values → true entity.

    Three things happen to one of these, in order:

    * **build**: `AnswerKey.from_sources()` derives it from generated data. Pass a
      subset of the true entities to make a matcher that knows only part of the answer.
    * **store**: `.register()` puts it in a process-local registry and returns a
      content-addressed ID. A matcher's settings carry that ID rather than the table
      itself, because settings are JSON-serialised into a step's fingerprint and a
      lookup table is not. Hashing the content keeps the fingerprint honest. A different
      answer produces a different ID, and so a different model artifact.
    * **use**: `.dedupe_edges()` / `.link_edges()` are what `PerfectDeduper` and
      `PerfectLinker` call at match time, against whatever record step they were
      handed.

    `groups` maps a tuple of column values to a true-entity ID. `columns` names the
    columns to read, in the same order they appear in the record step the model is
    given (source-qualified). Linkers carry a second set for the right-hand record
    step. Both sides map into the same entity-ID space, which is what lets them be
    joined.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    left_columns: tuple[str, ...]
    left_groups: dict[tuple, int]
    right_columns: tuple[str, ...] | None = None
    right_groups: dict[tuple, int] | None = None
    score: float = 1.0

    @classmethod
    def from_sources(
        cls,
        left: GeneratedSource,
        true_entities: Iterable[TrueEntity],
        right: GeneratedSource | None = None,
        score: float = 1.0,
    ) -> "AnswerKey":
        """Derive the lookup from generated sources.

        Each generated row is mapped to the true entity that owns its key, then keyed by
        its feature values under the names the model will see them by, source-qualified,
        because that is how they arrive in the record step a matcher is handed.

        Args:
            left: The generated source the matcher reads as its left input.
            right: Its right input, for a linker. `None` for a deduper.
            true_entities: The planted entities to answer for. Pass a subset to build a
                matcher that knows only part of the answer.
            score: The score to emit on every edge.
        """
        entities = tuple(true_entities)

        def side(source: GeneratedSource) -> tuple[tuple[str, ...], dict[tuple, int]]:
            key_to_entity: dict[str, int] = {
                str(key): entity.id
                for entity in entities
                for key in entity.get_keys(source.source.name)
            }
            # From the testkit, not the plan node. The node would have to read the
            # warehouse to list its columns, and this runs before any collect.
            # `field_names` also covers a source set by hand, where `features` is None.
            features = source.field_names
            groups: dict[tuple, int] = {}
            for row in pl.DataFrame(source.data).iter_rows(named=True):
                entity_id = key_to_entity.get(str(row["key"]))
                if entity_id is not None:
                    groups[tuple(row[feature] for feature in features)] = entity_id
            columns = tuple(f"{source.source.name}_{feature}" for feature in features)
            return columns, groups

        left_columns, left_groups = side(left)
        right_columns, right_groups = side(right) if right is not None else (None, None)
        return cls(
            left_columns=left_columns,
            left_groups=left_groups,
            right_columns=right_columns,
            right_groups=right_groups,
            score=score,
        )

    def _assign(
        self, data: pl.DataFrame, columns: tuple[str, ...], groups: dict[tuple, int]
    ) -> dict[int, list[int]]:
        """Bucket the row IDs by the true entity their values belong to."""
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise KeyError(
                f"Ground-truth columns absent from the model input: {missing}. "
                f"Available: {sorted(data.columns)}"
            )

        buckets: dict[int, list[int]] = {}
        for row in data.select("id", *columns).iter_rows(named=True):
            entity = groups.get(tuple(row[column] for column in columns))
            if entity is not None:
                buckets.setdefault(entity, []).append(row["id"])
        return buckets

    def _edges(self, pairs: list[tuple[int, int]]) -> pl.DataFrame:
        if not pairs:
            return pl.DataFrame(schema=pl.Schema(SCHEMA_MODEL_EDGES))
        left, right = zip(*pairs, strict=True)
        return pl.DataFrame(
            {
                "left_id": list(left),
                "right_id": list(right),
                "score": [self.score] * len(pairs),
            },
            schema=pl.Schema(SCHEMA_MODEL_EDGES),
        )

    def dedupe_edges(self, data: pl.DataFrame) -> pl.DataFrame:
        """All within-entity pairs among the record step's records."""
        pairs: list[tuple[int, int]] = []
        for ids in self._assign(data, self.left_columns, self.left_groups).values():
            unique = sorted(set(ids))
            pairs.extend(combinations(unique, 2))
        return self._edges(pairs)

    def link_edges(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """All cross-side pairs whose records share a true entity."""
        left_buckets = self._assign(left, self.left_columns, self.left_groups)
        right_buckets = self._assign(
            right, self.right_columns or (), self.right_groups or {}
        )
        pairs = [
            (left_id, right_id)
            for entity, left_ids in left_buckets.items()
            for left_id in sorted(set(left_ids))
            for right_id in sorted(set(right_buckets.get(entity, [])))
        ]
        return self._edges(pairs)

    def register(self) -> str:
        """Store this key, returning a content-addressed ID for it.

        Settings are JSON-serialised into a step's fingerprint and a lookup table is
        not, so the key itself cannot live there. Hashing its content keeps the
        fingerprint honest. A different answer key produces a different ID, and
        therefore a different model artifact.
        """
        payload = repr(
            (
                self.left_columns,
                sorted(map(repr, self.left_groups.items())),
                self.right_columns,
                sorted(map(repr, (self.right_groups or {}).items())),
                self.score,
            )
        ).encode()
        key_id = HASH_FUNC(payload).hexdigest()
        _ANSWERS[key_id] = self
        return key_id


class PerfectDeduper(Deduper):
    """A perfect deduper. Emits every within-entity pair it is given."""

    version: ClassVar[int] = 1

    truth_id: str

    def dedupe(
        self, prepared_state: object, data: pl.DataFrame, *, baseline: pl.DataFrame
    ) -> pl.DataFrame:
        """Emit edges between records sharing a true entity."""
        if baseline is None:
            raise RuntimeError("Call prepare() before dedupe()")
        records = pl.concat([baseline, data]).unique()
        affected_ids = data["id"].to_list()
        return (
            _ANSWERS[self.truth_id]
            .dedupe_edges(records)
            .filter(
                pl.col("left_id").is_in(affected_ids)
                | pl.col("right_id").is_in(affected_ids)
            )
        )


class PerfectLinker(Linker):
    """A perfect linker. Emits every cross-side pair sharing a true entity."""

    version: ClassVar[int] = 1

    truth_id: str

    def link(
        self,
        prepared_state: object,
        left: pl.DataFrame | None = None,
        right: pl.DataFrame | None = None,
        *,
        baseline_left: pl.DataFrame,
        baseline_right: pl.DataFrame,
    ) -> pl.DataFrame:
        """Emit edges involving affected rows without changing the prepared inputs."""
        if left is None and right is None:
            raise ValueError("Provide at least one affected side to link()")
        answer = _ANSWERS[self.truth_id]
        edges = []
        if left is not None:
            edges.append(answer.link_edges(left, baseline_right))
        if right is not None:
            edges.append(answer.link_edges(baseline_left, right))
        if left is not None and right is not None:
            edges.append(answer.link_edges(left, right))
        return pl.concat(edges).unique(subset=["left_id", "right_id"])


add_model_class(PerfectDeduper)
add_model_class(PerfectLinker)

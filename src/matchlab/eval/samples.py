"""Client-side helpers for retrieving and preparing evaluation samples.

Everything here takes a **Resolver**: one you are holding, the label one was
published under, or a sequence of either. The sequence form is what makes two
methodologies comparable. Sampling across several resolvers unions their components,
so one round of judging covers all of them and the scores answer the same question.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import polars as pl
from fastdsu import connected_components
from pydantic import BaseModel

from matchlab.core.dataframes import qualify
from matchlab.core.exceptions import SourceTableError
from matchlab.eval.judgements import Judgement
from matchlab.eval.metrics import PrecisionRecall, precision_recall

if TYPE_CHECKING:
    from matchlab.resolvers import Resolver
    from matchlab.stores import Fingerprint, Store
else:
    Store = Any
    Fingerprint = Any
    Resolver = Any

ResolverRef: TypeAlias = "Resolver | str"
"""One resolver to read: a live `Resolver`, or the label one was published under."""

Reading: TypeAlias = "tuple[Store, Fingerprint]"
"""A located resolver: the store holding it, and its fingerprint."""


class EvaluationFieldMetadata(BaseModel):
    """Metadata describing one field shown to a reviewer during evaluation."""

    display_name: str
    source_columns: list[str]


class EvaluationItem(BaseModel):
    """One cluster shown to a reviewer, with its leaves' source data alongside it.

    `records` holds the leaf IDs and their source-qualified data columns. For example:

    | leaf | src_a_first | src_a_last | src_b_first | src_b_last |
    |------|-------------|------------|-------------|------------|
    | 1    | Thomas      | Bayes      |             |            |
    | 2    | Tommy       | B          |             |            |
    | 12   |             |            | Tom         | Bayes      |

    `fields` maps each display name to the source-qualified columns holding it, so a
    reviewer sees one column per field rather than one per source. For example:

    ```python
    EvaluationFieldMetadata(
        display_name="first", source_columns=["src_a_first", "src_b_first"]
    )
    ```
    """

    model_config = {"arbitrary_types_allowed": True}

    leaves: list[int]
    records: pl.DataFrame
    fields: list[EvaluationFieldMetadata]

    def get_unique_record_groups(self) -> list[list[int]]:
        """Group identical records by leaf ID.

        Returns:
            List of groups, where each group is a list of leaf IDs
            that have identical values across all data fields.
            Example: [[1, 3], [2], [4, 5, 6]] means records 1 & 3 are identical.
        """
        data_cols = [col for field in self.fields for col in field.source_columns]
        grouped = self.records.group_by(data_cols, maintain_order=True).agg(
            pl.col("leaf")
        )
        return [group for group in grouped["leaf"]]


def create_judgement(
    item: EvaluationItem,
    assignments: dict[int, str],
    tag: str | None = None,
) -> Judgement:
    """Build the Judgement for one reviewed item from its column assignments.

    Args:
        item: The reviewed cluster.
        assignments: Group letter chosen for each unique record group, keyed by that
            group's index in `item.get_unique_record_groups()`.
        tag: Tag to record on the judgement.

    Returns:
        A Judgement with one endorsed group per distinct letter used in
        `assignments`. A leaf whose group was never assigned is missing from
        `endorsed`, which `Judgement`'s own validation rejects, so an incomplete
        `assignments` raises rather than producing a partial judgement.
    """
    groups: dict[str, list[int]] = {}
    unique_record_groups = item.get_unique_record_groups()

    for col_idx, group in assignments.items():
        leaf_ids = unique_record_groups[col_idx]
        groups.setdefault(group, []).extend(leaf_ids)

    endorsed = [sorted(set(leaf_ids)) for leaf_ids in groups.values()]
    return Judgement(shown=item.leaves, endorsed=endorsed, tag=tag)


def create_evaluation_item(
    df: pl.DataFrame,
    source_fields: list[tuple[str, list[str]]],
    leaves: list[int],
) -> EvaluationItem:
    """Build an EvaluationItem, grouping each cluster's columns by field across sources.

    Args:
        df: The cluster's rows, with source-qualified data columns.
        source_fields: `(prefix, qualified columns)` per source. The columns come
            from the fetched data rather than from the sources, which would have to
            re-read the warehouse just to list their names.
        leaves: The leaf IDs in this cluster.
    """
    data_cols = [c for c in df.columns if c not in ["root", "leaf", "key"]]

    # The same field in two sources shares a display name, which is what lines them
    # up for review.
    field_to_columns: dict[str, list[str]] = {}

    for prefix, columns in source_fields:
        for column in columns:
            if column in data_cols:
                field = column.removeprefix(prefix)
                field_to_columns.setdefault(field, []).append(column)

    fields: list[EvaluationFieldMetadata] = []
    for field_name, source_columns in field_to_columns.items():
        fields.append(
            EvaluationFieldMetadata(
                display_name=field_name, source_columns=source_columns
            )
        )

    records = df.select(["leaf"] + data_cols)

    return EvaluationItem(leaves=leaves, records=records, fields=fields)


def _many(resolver: "ResolverRef | Sequence[ResolverRef]") -> bool:
    """Whether several resolvers were asked for.

    A `str` is itself a `Sequence`. A label names one resolver, not a pile of
    one-character ones, so it is excluded before the sequence check, not after.
    """
    return not isinstance(resolver, str) and isinstance(resolver, Sequence)


def _locate(resolver: "ResolverRef", store: "Store | None") -> tuple["Store", bytes]:
    """Turn a resolver or a label into the store and fingerprint to read.

    The two forms differ only in how the fingerprint is found. A live resolver
    already knows its own fingerprint, while a label needs a lookup in a store.
    Everything downstream wants the same pair, so this is the only place the
    distinction exists.

    Raises:
        SourceTableError: If nothing is published under that label.
    """
    from matchlab.stores.default import default_store  # noqa: PLC0415 - avoids a cycle

    if isinstance(resolver, str):
        store = store or default_store()
        fingerprint = store.find(resolver)
        if fingerprint is None:
            known = ", ".join(store.labels()) or "none"
            raise SourceTableError(
                f"No resolver is published under the label '{resolver}'. "
                f"Known labels: {known}."
            )
        return store, fingerprint

    if not resolver.is_collected:
        resolver.collect(store)
    collected_in, fingerprint = resolver._collected()
    return store or collected_in, fingerprint


def _readings(
    resolver: "ResolverRef | Sequence[ResolverRef]", store: "Store | None"
) -> list[Reading]:
    """Locate every resolver asked for, in the order given."""
    if not _many(resolver):
        return [_locate(resolver, store)]
    if not resolver:
        raise ValueError("At least one resolver must be given.")
    return [_locate(one, store) for one in resolver]


def _sources_of(readings: list[Reading]) -> dict[str, Reading]:
    """Map each source name to the store and fingerprint its rows come from.

    Raises:
        SourceTableError: If two resolvers cover the same source name with different
            artifacts. Names repeat across generations of a source, so agreeing on the
            name is not agreeing on the data, and comparing methodologies over different
            data is not a comparison.
    """
    located: dict[str, Reading] = {}
    for store, fp in readings:
        for name, source_fp in store.resolver_output_sources(fp).items():
            seen = located.setdefault(name, (store, source_fp))
            if seen[1] != source_fp:
                raise SourceTableError(
                    f"These resolvers disagree about source '{name}': one covers "
                    f"{seen[1].hex()[:8]}, another {source_fp.hex()[:8]}. They are "
                    "built over different data, so their clusters cannot be compared. "
                    "Re-collect them over the same sources."
                )
    return located


def _merged_resolver_output(readings: list[Reading]) -> pl.DataFrame:
    """Union several resolvers' output into one `(root, leaf, key, source)` table.

    Two records land in the same merged component if either resolver put them
    together. This is the right sample for a bake-off. Every cluster where the
    methodologies could disagree ends up on screen, so one judgement settles it for
    both, and neither resolver gets to pick the clusters it is scored on.

    Merged roots are minted with `root_id`, the same content-addressed function a
    resolver uses for its own roots, so two people running the same comparison arrive
    at identical root IDs for identical components. Nothing persists these merged roots.
    `store_judgement` re-mints its own root from the leaves it is given, so a merged
    root only ever lives as far as the reviewer.
    """
    resolved: pl.DataFrame = pl.concat(
        [store.read_resolver(fp) for store, fp in readings]
    )
    edges: pl.DataFrame = (
        resolved.group_by("root")
        .agg(
            pl.col("leaf").first().alias("src"),
            pl.col("leaf").implode().alias("dst"),
        )
        .select(pl.col("src"), pl.col("dst"))
        .explode("dst", empty_as_null=True)
        .sort("src", "dst")
    )
    components: pl.DataFrame = pl.from_arrow(
        connected_components(edges["src"].to_arrow(), edges["dst"].to_arrow())
    ).rename({"key": "leaf", "label": "root"})

    return (
        components.unique()
        .join(
            resolved.select("leaf", "key", "source"),
            on="leaf",
        )
        .select("root", "leaf", "key", "source")
    )


def _sample_clusters(
    resolver_output: pl.DataFrame, n: int, seed: int | None
) -> pl.DataFrame:
    """Take up to `n` whole clusters from merge-forwarded Resolver output in memory.

    The in-memory twin of `Store.sample`, used for merged Resolver output from
    several readings. No store holds that merged output, since it exists only for
    this comparison.
    """
    roots = resolver_output["root"].unique()
    if n < roots.len():
        roots = roots.sample(n=n, seed=seed, shuffle=True)
    return resolver_output.filter(pl.col("root").is_in(roots.to_list()))


def get_samples(
    n: int,
    resolver: "ResolverRef | Sequence[ResolverRef]",
    store: "Store | None" = None,
    seed: int | None = None,
) -> dict[int, EvaluationItem]:
    """Retrieve sampled clusters enriched with source data, as EvaluationItems.

    Record values come from the extract stored when each source was collected, not
    from a fresh warehouse read. This works offline, and shows the data the matching
    actually saw.

    Args:
        n: Number of clusters to sample.
        resolver: The resolver to sample from, collected first if it isn't
            already, or the label one was published under. The label form needs no
            plan, because a stored resolver's output records which source artifacts
            it covers. Pass several and the sample is drawn from their merged
            components, so one round of judging scores all of them against the same
            clusters.
        store: Where to read from. Defaults to the resolver's, else the module
            default.
        seed: Fixes which clusters come back. The same store, `n` and seed give the
            same sample, which is how two people review the same clusters.

    Returns:
        Dictionary of cluster ID to EvaluationItems describing the cluster.

    Raises:
        SourceTableError: If nothing is published under `resolver`, if a source the
            resolver's output covers isn't in the store, or if several resolvers
            disagree about a source.
        ValueError: If `resolver` is an empty sequence.
    """
    readings = _readings(resolver, store)

    if len(readings) == 1:
        store, resolver_fp = readings[0]
        samples = store.sample(resolver_fp, n, seed)
    else:
        samples = _sample_clusters(_merged_resolver_output(readings), n, seed)

    if not len(samples):
        return {}

    sources = _sources_of(readings)
    results_by_source: list[pl.DataFrame] = []
    source_fields: list[tuple[str, list[str]]] = []

    for source_step in samples["source"].unique():
        located = sources.get(source_step)
        if located is None:
            raise SourceTableError(
                f"This resolver references source '{source_step}', which is not "
                "in the store. Re-collect the plan to repopulate it."
            )
        source_store, source_fp = located

        samples_by_source = samples.filter(pl.col("source") == source_step)
        rows, qualified_key = source_store.read_source_records(
            source_fp, source_step, samples_by_source["key"]
        )
        values = [column for column in rows.columns if column != qualified_key]

        samples_and_source = samples_by_source.join(
            rows, left_on="key", right_on=qualified_key
        )
        source_fields.append((qualify(source_step), values))
        results_by_source.append(samples_and_source[["root", "leaf", "key"] + values])

    if not results_by_source:
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    results_by_root: dict[int, EvaluationItem] = {}
    for root in all_results["root"].unique():
        cluster_df = all_results.filter(pl.col("root") == root).drop("root")
        leaves = cluster_df.select("leaf").to_series().unique().to_list()
        evaluation_item = create_evaluation_item(cluster_df, source_fields, leaves)
        results_by_root[root] = evaluation_item

    return results_by_root


class EvalData:
    """Caches a store's judgements, and scores resolvers against them."""

    def __init__(self, store: Store, tag: str | None = None) -> None:
        """Load judgement and expansion data used to compute evaluation metrics.

        Args:
            store: The store holding judgements, for example the one a
                resolver was collected into.
            tag: Optional tag to filter judgements by.
        """
        self.store = store
        self.tag = tag
        self.judgements, self.expansion = store.read_eval_data(tag)

    def precision_recall(
        self, resolver: "ResolverRef | Sequence[ResolverRef]"
    ) -> PrecisionRecall | list[PrecisionRecall]:
        """Score one or more resolvers against these judgements.

        Only pairs present in every resolver's output *and* in the judgements are
        compared, so scoring several at once is the fair way to rank them. Each is
        measured over the same records, and none is flattered by clusters the others
        never saw. Scoring them one at a time gives each its own comparison set, and
        those numbers do not line up.

        Args:
            resolver: A resolver, the label one was published under, or a sequence of
                either.

        Returns:
            One `(precision, recall)` pair, or a list of them in the order given if a
            sequence was passed.
        """
        readings = _readings(resolver, self.store)
        scores = precision_recall(
            [
                store.read_resolver(fp).select("root", "leaf").unique()
                for store, fp in readings
            ],
            self.judgements,
            self.expansion,
        )
        return scores if _many(resolver) else scores[0]

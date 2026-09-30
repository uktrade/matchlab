"""The storage contract every `Store` must honour, over each backend.

These assert behaviour, not DuckDB: round-trips, `has` before store, schema rejection,
idempotence, `read_identifiers`, sampling, the evaluation round-trip, pruning, and
movable labels. A test taking the `store` fixture runs over every backend. A test
taking `durable_store` runs only over backends that survive a reopen. See
`conftest.py` for how the backends are wired in.
"""

from collections.abc import Callable

import polars as pl
import pytest

from matchlab.core.exceptions import SchemaMismatch
from matchlab.eval.judgements import Judgement
from matchlab.eval.metrics import precision_recall
from matchlab.stores import Store

from .conftest import Fingerprints

# -- existence + round-trips ----------------------------------------------------------


def test_has_before_store(store: Store, fp: Fingerprints) -> None:
    """`has` is false until the fingerprint is stored."""
    assert not store.has(fp.src)


def test_round_trip_source(
    store: Store, fp: Fingerprints, extract: pl.DataFrame, leaves: pl.DataFrame
) -> None:
    """A source's extract and leaves survive a store-and-read."""
    store.store_source(fp.src, "key", extract, leaves)

    assert store.has(fp.src)
    assert store.read_source_extract(fp.src).sort("key").equals(extract.sort("key"))

    read = store.read_source_leaves(fp.src).sort("key")
    assert read.equals(leaves.sort("key"))
    assert read.schema["leaf"] == pl.UInt64


def test_round_trip_missing_raises(store: Store, fp: Fingerprints) -> None:
    """Reading an unstored fingerprint raises, not returns empty."""
    with pytest.raises(KeyError):
        store.read_source_extract(fp.src)


def test_round_trip_model(store: Store, fp: Fingerprints, edges: pl.DataFrame) -> None:
    """A model's edges survive a store-and-read."""
    store.store_model(fp.model, edges)
    assert store.read_model(fp.model).sort("left_id").equals(edges.sort("left_id"))


def test_round_trip_resolver(
    store: Store, fp: Fingerprints, resolver_output: pl.DataFrame
) -> None:
    """A resolver's output survives a store-and-read."""
    store.store_resolver(fp.resolver, resolver_output)
    assert (
        store.read_resolver(fp.resolver)
        .sort("leaf")
        .equals(resolver_output.sort("leaf"))
    )


def test_store_idempotent(
    store: Store, fp: Fingerprints, resolver_output: pl.DataFrame
) -> None:
    """Storing the same fingerprint twice leaves one copy, not two."""
    store.store_resolver(fp.resolver, resolver_output)
    store.store_resolver(fp.resolver, resolver_output)  # no duplicate rows
    assert store.read_resolver(fp.resolver).height == resolver_output.height


# -- schema rejection -----------------------------------------------------------------


def test_store_rejects_resolver_schema(store: Store, fp: Fingerprints) -> None:
    """A resolver table missing key/source columns is refused."""
    bad = pl.DataFrame({"root": [1], "leaf": [2]})  # missing key/source
    with pytest.raises(SchemaMismatch):
        store.store_resolver(fp.resolver, bad)


def test_store_rejects_model_schema(store: Store, fp: Fingerprints) -> None:
    """A model table missing its score column is refused."""
    bad = pl.DataFrame({"left_id": [1], "right_id": [2]})  # missing score
    with pytest.raises(SchemaMismatch):
        store.store_model(fp.model, bad)


# -- introspection --------------------------------------------------------------------


def test_stats_counts_artifacts(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """Stats tally one entry per stored artifact, and the labels."""
    assert store.stats().artifacts == {}

    store.store_source(fp.src, "key", extract, leaves)
    store.store_model(fp.model, edges)
    store.store_resolver(fp.resolver, resolver_output)
    store.publish("production", fp.resolver)

    stats = store.stats()
    assert stats.artifacts == {"source": 1, "model": 1, "resolver": 1}
    assert stats.labels == 1


# -- read_identifiers -----------------------------------------------------------------


def test_read_identifiers_direct(
    store: Store, fp: Fingerprints, extract: pl.DataFrame, leaves: pl.DataFrame
) -> None:
    """Without a resolver, a record's `id` is its own leaf."""
    store.store_source(fp.src, "key", extract, leaves)

    out = store.read_identifiers(fp.src, "crn").sort("key")

    assert out.columns == ["id", "source", "key", "leaf"]
    assert out["id"].to_list() == [1, 2, 3]
    assert out["leaf"].to_list() == [1, 2, 3]
    assert out["source"].to_list() == ["crn"] * 3
    assert out.schema["id"] == pl.UInt64
    assert out.schema["source"] == pl.Utf8


@pytest.mark.parametrize(
    ("ids", "keys"),
    [
        pytest.param({2}, ["k2"], id="one_leaf"),
        pytest.param({2, 3}, ["k2", "k3"], id="two_leaves"),
        pytest.param({999}, [], id="unknown_leaf"),
        pytest.param(set(), [], id="empty"),
    ],
)
def test_read_identifiers_direct_filtered(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    ids: set[int],
    keys: list[str],
) -> None:
    """A direct reading selects source leaves by the returned ID."""
    store.store_source(fp.src, "key", extract, leaves)

    out = store.read_identifiers(fp.src, "crn", ids=ids).sort("key")

    assert out["key"].to_list() == keys
    assert out.schema == store.read_identifiers(fp.src, "crn").schema


def test_read_identifiers_through_resolver(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """Through a resolver, `id` is the root cluster and `leaf` still names the row."""
    store.store_source(fp.src, "key", extract, leaves)
    store.store_resolver(fp.resolver, resolver_output)

    out = store.read_identifiers(fp.src, "crn", fp.resolver).sort("key")

    assert out.columns == ["id", "source", "key", "leaf"]
    # k1/k2 were clustered together upstream, so they share an id but keep their leaves.
    assert out["id"].to_list() == [10, 10]
    assert out["leaf"].to_list() == [1, 2]


@pytest.mark.parametrize(
    ("ids", "keys"),
    [
        pytest.param({10}, ["k1", "k2"], id="whole_root"),
        pytest.param({10, 20}, ["k1", "k2"], id="other_source_excluded"),
        pytest.param({1}, [], id="leaf_not_root"),
        pytest.param(set(), [], id="empty"),
    ],
)
def test_read_identifiers_resolver_filtered(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    resolver_output: pl.DataFrame,
    ids: set[int],
    keys: list[str],
) -> None:
    """A resolver reading selects whole roots within the named source."""
    store.store_source(fp.src, "key", extract, leaves)
    store.store_resolver(fp.resolver, resolver_output)

    out = store.read_identifiers(fp.src, "crn", fp.resolver, ids=ids).sort("key")

    assert out["key"].to_list() == keys
    assert out.schema == store.read_identifiers(fp.src, "crn", fp.resolver).schema


def test_read_identifiers_one_source(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """The whole point. One source's rows, not every source in the resolver output.

    `resolver_output` holds crn and dh together. Reading crn must not return dh's rows.
    A plan linking every pair of sources asks for this once per pair, so returning the
    whole table and filtering afterwards is what made it quadratic.
    """
    store.store_source(fp.src, "key", extract, leaves)
    store.store_resolver(fp.resolver, resolver_output)

    assert store.read_identifiers(fp.src, "crn", fp.resolver).height == 2
    assert store.read_identifiers(fp.src, "dh", fp.resolver).height == 2
    assert store.read_identifiers(fp.src, "nope", fp.resolver).height == 0


def test_read_identifiers_missing_raises(
    store: Store, fp: Fingerprints, extract: pl.DataFrame, leaves: pl.DataFrame
) -> None:
    """A read naming an unstored source or resolver raises."""
    with pytest.raises(KeyError):
        store.read_identifiers(fp.src, "crn")

    store.store_source(fp.src, "key", extract, leaves)
    with pytest.raises(KeyError):
        store.read_identifiers(fp.src, "crn", fp.resolver)


def test_read_identifiers_empty_missing_raises(
    store: Store, fp: Fingerprints, extract: pl.DataFrame, leaves: pl.DataFrame
) -> None:
    """An empty filter does not hide a missing source or resolver."""
    with pytest.raises(KeyError):
        store.read_identifiers(fp.src, "crn", ids=set())

    store.store_source(fp.src, "key", extract, leaves)
    with pytest.raises(KeyError):
        store.read_identifiers(fp.src, "crn", fp.resolver, ids=set())


# -- sampling -------------------------------------------------------------------------


def test_sample_whole_clusters(
    store: Store, fp: Fingerprints, resolver_output: pl.DataFrame
) -> None:
    """A sample returns each cluster whole, and is stable for a seed."""
    store.store_resolver(fp.resolver, resolver_output)

    sample_a = store.sample(fp.resolver, n=1, seed=42)
    sample_b = store.sample(fp.resolver, n=1, seed=42)
    assert sample_a.equals(sample_b)

    # Sampling one root returns all of that root's rows, nothing partial.
    roots = sample_a["root"].unique().to_list()
    assert len(roots) == 1
    full = resolver_output.filter(pl.col("root") == roots[0]).sort("leaf")
    assert sample_a.sort("leaf").equals(full)


def test_sample_caps_at_available(
    store: Store, fp: Fingerprints, resolver_output: pl.DataFrame
) -> None:
    """Asking for more clusters than exist returns all of them."""
    store.store_resolver(fp.resolver, resolver_output)
    assert store.sample(fp.resolver, n=999).height == resolver_output.height


# -- evaluation round-trip ------------------------------------------------------------


def test_eval_round_trip(store: Store) -> None:
    """Store a judgement, read it back, and score a model that matches it exactly."""
    # User shown leaves 1-4, endorses {1,2} and {3,4} as two separate entities.
    store.store_judgement(
        Judgement(shown=[1, 2, 3, 4], endorsed=[[1, 2], [3, 4]]), user_name="alice"
    )

    judgements, expansion = store.read_eval_data()
    assert judgements.height == 2  # one row per endorsed group

    # A model that clusters exactly {1,2} and {3,4}.
    model_root_leaf = pl.DataFrame(
        {"root": [100, 100, 200, 200], "leaf": [1, 2, 3, 4]},
        schema={"root": pl.UInt64, "leaf": pl.UInt64},
    )

    (precision, recall) = precision_recall([model_root_leaf], judgements, expansion)[0]
    assert precision == 1.0
    assert recall == 1.0


def test_eval_tag_filter(store: Store) -> None:
    """Reading by tag returns only that tag's judgements."""
    store.store_judgement(Judgement(tag="v1", shown=[1, 2], endorsed=[[1, 2]]))
    store.store_judgement(Judgement(tag="v2", shown=[3, 4], endorsed=[[3, 4]]))

    assert store.read_eval_data(tag="v1")[0].height == 1
    assert store.read_eval_data()[0].height == 2


def test_eval_empty_schema(store: Store) -> None:
    """An empty store returns eval frames with the right columns."""
    judgements, expansion = store.read_eval_data()
    assert judgements.height == 0
    assert set(judgements.columns) == {"user_name", "endorsed", "shown"}
    assert set(expansion.columns) == {"root", "leaves"}


# -- movable labels -------------------------------------------------------------------


def test_label_is_movable(
    store: Store, fp: Fingerprints, resolver_output: pl.DataFrame
) -> None:
    """A label points at a fingerprint, and moves when told to.

    The store does not argue about overwriting. That is `Resolver.publish`'s call.
    What it guarantees is that a label resolves to exactly one fingerprint. The old
    (kind, name) column on `artifacts` did not. Several generations shared a name, and
    the lookup picked whichever row came back first.
    """
    store.store_resolver(fp.resolver, resolver_output)
    store.store_resolver(fp.resolver_b, resolver_output)

    assert store.find("entities") is None
    assert store.labels() == []

    store.publish("entities", fp.resolver)
    assert store.find("entities") == fp.resolver
    assert store.labels() == ["entities"]

    store.publish("entities", fp.resolver_b)
    assert store.find("entities") == fp.resolver_b
    assert store.labels() == ["entities"]  # moved, not duplicated


def test_label_survives_restore(
    store: Store, fp: Fingerprints, resolver_output: pl.DataFrame
) -> None:
    """Re-collecting a published plan must not quietly revoke the publication.

    Storing replaces the artifact for a fingerprint. Because a fingerprint addresses
    content, the label resolves to the same bytes it always did. Dropping it here used
    to mean a re-collect silently unpublished your resolver output.
    """
    store.store_resolver(fp.resolver, resolver_output)
    store.publish("entities", fp.resolver)

    store.store_resolver(fp.resolver, resolver_output)  # same fingerprint, again

    assert store.find("entities") == fp.resolver
    assert store.labels() == ["entities"]
    assert store.read_resolver(fp.resolver).height == resolver_output.height


# -- prune ---------------------------------------------------------------------------


def _publish_over_one_source(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """A published resolver output over one source, plus a model to throw away."""
    store.store_source(fp.src, "key", extract, leaves)
    store.store_resolver(fp.resolver, resolver_output, sources={"crn": fp.src})
    store.store_model(fp.model, edges)
    store.publish("entities", fp.resolver)


def test_prune_keeps_named(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
) -> None:
    """Prune keeps the fingerprints it was given and drops the rest."""
    store.store_source(fp.src, "key", extract, leaves)
    store.store_model(fp.model, edges)

    result = store.prune(keep=[fp.src])

    assert result.removed == 1
    assert result.kept == 1
    assert store.has(fp.src)
    assert not store.has(fp.model)
    # The kept artifact is not merely listed. It still reads back.
    assert store.read_source_extract(fp.src).height == extract.height


def test_prune_rejects_non_fingerprint(
    store: Store, fp: Fingerprints, extract: pl.DataFrame, leaves: pl.DataFrame
) -> None:
    """Prune rejects roots that are not artifact fingerprints."""
    store.store_source(fp.src, "key", extract, leaves)

    with pytest.raises(TypeError, match="expects fingerprints"):
        store.prune(keep=["entities"])

    assert store.has(fp.src)


def test_prune_keeps_label_and_sources(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """A publication survives a prune that never mentioned it, and stays *usable*.

    Keeping the label row alone is not enough. Reading a published resolver output
    without a plan goes through `resolver_output_sources` to each source's extract. A
    label kept without its sources resolves to a fingerprint whose data has gone. That
    fails with a bare `KeyError` well away from the cause.
    """
    _publish_over_one_source(store, fp, extract, leaves, edges, resolver_output)

    result = store.prune(keep=[])  # names nothing at all

    assert result.removed == 1  # the model, and only the model
    assert store.labels() == ["entities"]
    # Walk the whole label-only read path.
    assert store.find("entities") == fp.resolver
    assert store.sample(fp.resolver, n=10).height > 0
    assert store.resolver_output_sources(fp.resolver) == {"crn": fp.src}
    assert store.read_source_extract(fp.src).height == extract.height
    assert store.source_key_field(fp.src) == "key"


def test_prune_keeps_judgements(
    store: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """Human work, and the only thing in the store that cannot be recomputed."""
    _publish_over_one_source(store, fp, extract, leaves, edges, resolver_output)
    store.store_judgement(Judgement(shown=[1, 2, 3, 4], endorsed=[[1, 2], [3, 4]]))

    store.prune(keep=[])

    judgements, expansion = store.read_eval_data()
    assert judgements.height == 2
    assert expansion.height > 0


def test_prune_refuses_to_empty(
    store: Store, fp: Fingerprints, extract: pl.DataFrame, leaves: pl.DataFrame
) -> None:
    """An accidentally-empty list should not be how a store gets emptied."""
    store.store_source(fp.src, "key", extract, leaves)

    with pytest.raises(ValueError, match="would empty the store"):
        store.prune(keep=[])

    assert store.has(fp.src)


# -- durable stores: persistence across a reopen --------------------------------------


def test_persists_across_reopen(
    durable_store: Callable[[], Store],
    fp: Fingerprints,
    resolver_output: pl.DataFrame,
) -> None:
    """Artifacts, publications and judgements are all still there after a reopen.

    Artifacts are a cache and can be recomputed. Publications and judgements cannot.
    `_open_schema` drops every table when the schema version moves, so this test would
    catch a version bump taken without thinking about what else is in there.
    """
    a = durable_store()
    a.store_resolver(fp.resolver, resolver_output)
    a.publish("entities", fp.resolver)
    a.store_judgement(Judgement(shown=[1, 2], endorsed=[[1, 2]]))
    a.close()

    b = durable_store()
    try:
        assert b.has(fp.resolver)
        assert b.read_resolver(fp.resolver).height == resolver_output.height
        assert b.labels() == ["entities"]
        assert b.read_eval_data()[0].height == 1
    finally:
        b.close()

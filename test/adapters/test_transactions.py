"""Artifact isolation, mixed-store reads, lifecycle, and atomic promotion."""

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from matchlab.core.exceptions import SchemaMismatch
from matchlab.eval.judgements import Judgement
from matchlab.stores import DuckDBStore, DuckDBTransaction, Store, Transaction

from .conftest import Fingerprints


def _schemas(store: DuckDBStore) -> set[str]:
    return {
        name
        for (name,) in store.conn.execute(
            "SELECT schema_name FROM information_schema.schemata"
        ).fetchall()
    }


def test_transaction_abstract(store: DuckDBStore) -> None:
    """The generic factory returns a concrete implementation of an abstract contract."""
    with pytest.raises(TypeError, match="abstract"):
        Transaction()
    generic: Store = store
    tx = generic.transaction()
    try:
        assert isinstance(tx, Transaction)
        assert isinstance(tx, DuckDBTransaction)
    finally:
        tx.close()


@pytest.mark.parametrize(
    ("disk",),
    [pytest.param(False, id="memory"), pytest.param(True, id="disk")],
)
@pytest.mark.parametrize(
    ("ending",),
    [
        pytest.param("commit", id="commit"),
        pytest.param("rollback", id="rollback"),
        pytest.param("close", id="close"),
    ],
)
def test_transaction_schema(tmp_path: Path, disk: bool, ending: str) -> None:
    """Staging shares the parent database and removes only its private schema."""
    parent = DuckDBStore(tmp_path / "schema.duckdb" if disk else ":memory:")
    committed_fp, staged_fp = b"committed", b"staged"
    original = pl.DataFrame({"id": [1]})
    pending = pl.DataFrame({"id": [2]})
    parent.store_transform(committed_fp, original)
    before = _schemas(parent)
    tx = parent.transaction()
    schema = tx._staged._schema
    try:
        assert _schemas(parent) == before | {schema}
        assert parent.conn.execute("SELECT current_schema()").fetchall() == [("main",)]
        assert tx._staged.conn.execute("SELECT current_schema()").fetchall() == [
            (schema,)
        ]
        assert tx._staged.path == parent.path
        tx.store_transform(staged_fp, pending)
        assert not parent.has(staged_fp)
        assert (
            parent.conn.execute(
                f'SELECT * FROM "{schema}"."transform_{staged_fp.hex()}"'
            )
            .pl()
            .equals(pending)
        )
        assert tx.read_transform(committed_fp).equals(original)
        assert tx.stats().artifacts == {"transform": 1}
        if disk:
            observer = DuckDBStore(parent.path)
            try:
                assert (
                    observer.conn.execute(
                        f'SELECT * FROM "{schema}"."transform_{staged_fp.hex()}"'
                    )
                    .pl()
                    .equals(pending)
                )
                assert observer.has(committed_fp)
                assert not observer.has(staged_fp)
            finally:
                observer.close()
        getattr(tx, ending)()
        assert _schemas(parent) == before
        assert parent.read_transform(committed_fp).equals(original)
        assert parent.has(staged_fp) == (ending == "commit")
        if ending == "commit":
            assert tx.read_transform(staged_fp).equals(pending)
    finally:
        tx.close()
        parent.close()


def test_transaction_schema_failure(store: DuckDBStore) -> None:
    """Failed setup rolls back its DDL and leaves the parent namespace unchanged."""
    before = _schemas(store)
    with (
        patch(
            "matchlab.stores.duckdb._SCHEMA_DDL", "SELECT * FROM missing_setup_table"
        ),
        pytest.raises(duckdb.CatalogException, match="missing_setup_table"),
    ):
        store.transaction()
    assert _schemas(store) == before
    assert store.conn.execute("SELECT current_schema()").fetchall() == [("main",)]
    assert store.stats().artifacts == {}


def _write(
    target: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    target.store_source(fp.src, "key", extract, leaves)
    target.store_transform(fp.record_step, extract)
    target.store_model(fp.model, edges)
    target.store_resolver(fp.resolver, resolver_output, {"crn": fp.src})


def _assert_reads(
    target: Store,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    assert target.read_source_extract(fp.src).equals(extract)
    assert target.read_source_leaves(fp.src).equals(leaves)
    assert target.read_transform(fp.record_step).equals(extract)
    assert target.read_model(fp.model).equals(edges)
    assert target.read_resolver(fp.resolver).equals(resolver_output)
    assert target.source_key_field(fp.src) == "key"
    assert target.resolver_output_sources(fp.resolver) == {"crn": fp.src}
    rows, key = target.read_source_records(fp.src, "crn", pl.Series(["k1"]))
    assert key == "crn_key"
    assert rows["crn_key"].to_list() == ["k1"]


@pytest.mark.parametrize(
    ("placement",),
    [pytest.param("parent", id="fallback"), pytest.param("staged", id="own-writes")],
)
def test_transaction_reads(
    store: DuckDBStore,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
    placement: str,
) -> None:
    """All artifact payloads and metadata read identically from either overlay layer."""
    tx = store.transaction()
    try:
        _write(
            store if placement == "parent" else tx,
            fp,
            extract,
            leaves,
            edges,
            resolver_output,
        )
        _assert_reads(tx, fp, extract, leaves, edges, resolver_output)
        for artifact in (fp.src, fp.record_step, fp.model, fp.resolver):
            assert tx.has(artifact)
            assert store.has(artifact) == (placement == "parent")
    finally:
        tx.close()


@pytest.mark.parametrize(
    ("staged_source",),
    [pytest.param(False, id="parent-source"), pytest.param(True, id="staged-source")],
)
def test_transaction_identifiers(
    store: DuckDBStore,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    resolver_output: pl.DataFrame,
    staged_source: bool,
) -> None:
    """Mixed-store identifier reads filter in SQL without materialising full outputs."""
    tx = store.transaction()
    try:
        (tx if staged_source else store).store_source(fp.src, "key", extract, leaves)
        (store if staged_source else tx).store_resolver(fp.resolver, resolver_output)
        direct = tx.read_identifiers(fp.src, "crn")
        assert direct.columns == ["id", "source", "key", "leaf"]
        assert direct["id"].to_list() == [1, 2, 3]
        assert direct.schema == {
            "id": pl.UInt64,
            "source": pl.Utf8,
            "key": pl.Utf8,
            "leaf": pl.UInt64,
        }
        with patch.object(
            DuckDBStore,
            "read_resolver",
            side_effect=AssertionError("Full resolver output was materialised"),
        ):
            through = tx.read_identifiers(fp.src, "crn", fp.resolver)
            assert through["id"].to_list() == [10, 10]
            assert through["leaf"].to_list() == [1, 2]
            assert through.schema == direct.schema
            other = tx.read_identifiers(fp.src, "dh", fp.resolver)
            assert other["leaf"].to_list() == [3, 4]
            assert other.schema == direct.schema
            assert tx.read_identifiers(fp.src, "absent", fp.resolver).is_empty()
        with pytest.raises(KeyError):
            tx.read_identifiers(b"missing", "crn", fp.resolver)
        with pytest.raises(KeyError):
            tx.read_identifiers(fp.src, "crn", b"missing")
    finally:
        tx.close()


def test_transaction_rollback(
    store: DuckDBStore,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """Shadowing a parent artifact changes only the overlay, including on close."""
    _write(store, fp, extract, leaves, edges, resolver_output)
    store.publish("baseline", fp.resolver)
    store.store_judgement(Judgement(shown=[1, 2], endorsed=[[1, 2]], tag="baseline"))
    before_eval = store.read_eval_data()
    tx = store.transaction()
    try:
        tx.store_transform(fp.record_step, extract.head(1))
        tx.store_resolver(fp.resolver_b, resolver_output.head(1), {"crn": fp.src})
        assert tx.read_transform(fp.record_step).height == 1
        assert store.read_transform(fp.record_step).height == 3
        tx.close()
        tx.close()
        _assert_reads(store, fp, extract, leaves, edges, resolver_output)
        assert not store.has(fp.resolver_b)
        assert store.labels() == ["baseline"]
        assert store.find("baseline") == fp.resolver
        assert all(
            left.equals(right)
            for left, right in zip(before_eval, store.read_eval_data(), strict=True)
        )
        assert sum(store.stats().artifacts.values()) == 4
    finally:
        tx.close()


@pytest.mark.parametrize(
    ("empty",),
    [
        pytest.param(False, id="populated"),
        pytest.param(True, id="empty-edges-and-output"),
    ],
)
def test_transaction_commit(
    store: DuckDBStore,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
    empty: bool,
) -> None:
    """Promotion preserves artifact data, metadata, and empty table schemas."""
    if empty:
        edges, resolver_output = edges.head(0), resolver_output.head(0)
    tx = store.transaction()
    try:
        _write(tx, fp, extract, leaves, edges, resolver_output)
        tx.commit()
        _assert_reads(store, fp, extract, leaves, edges, resolver_output)
        _assert_reads(tx, fp, extract, leaves, edges, resolver_output)
        assert sum(store.stats().artifacts.values()) == 4
        assert tx.stats().artifacts == {}
        tx.close()
        assert store.has(fp.src)
    finally:
        tx.close()


def test_transaction_commit_failure(
    store: DuckDBStore,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
    resolver_output: pl.DataFrame,
) -> None:
    """Failed promotion rolls back replacements and newly created tables."""
    _write(store, fp, extract, leaves, edges, resolver_output)
    tx = store.transaction()
    schema = tx._staged._schema
    try:
        tx.store_source(fp.src, "key", extract.head(1), leaves.head(1))
        tx.store_transform(fp.record_step, extract.head(1))
        tx.store_transform(b"new-transform", extract)
        tx.store_model(fp.model, edges.head(1))
        tx.store_resolver(fp.resolver_b, resolver_output, {"crn": fp.src})
        original = store.store_resolver

        def fail_after_write(
            fingerprint: bytes,
            output: pl.DataFrame,
            sources: Mapping[str, bytes] | None = None,
        ) -> None:
            original(fingerprint, output, sources)
            raise ValueError("promotion failed")

        with (
            patch.object(store, "store_resolver", side_effect=fail_after_write),
            pytest.raises(ValueError, match="promotion failed"),
        ):
            tx.commit()
        _assert_reads(store, fp, extract, leaves, edges, resolver_output)
        assert not store.has(fp.resolver_b)
        assert not store.has(b"new-transform")
        assert tx.read_source_extract(fp.src).height == 1
        assert tx.has(fp.resolver_b)
        assert schema in _schemas(store)
        tx.rollback()
        assert schema not in _schemas(store)
        assert store.has(fp.src)
    finally:
        tx.close()


def test_transaction_validation(store: DuckDBStore, fp: Fingerprints) -> None:
    """The overlay preserves Store schema validation and missing-artifact failures."""
    tx = store.transaction()
    try:
        with pytest.raises(SchemaMismatch):
            tx.store_model(fp.model, pl.DataFrame({"left_id": [1]}))
        with pytest.raises(SchemaMismatch):
            tx.store_resolver(fp.resolver, pl.DataFrame({"root": [1]}))
        for read in (
            tx.read_source_extract,
            tx.read_source_leaves,
            tx.read_transform,
            tx.read_model,
            tx.read_resolver,
            tx.source_key_field,
        ):
            with pytest.raises(KeyError):
                read(fp.src)
        assert not tx.has(fp.model)
        assert not store.has(fp.model)
    finally:
        tx.close()


@pytest.mark.parametrize(
    ("ending",),
    [
        pytest.param("commit", id="committed"),
        pytest.param("rollback", id="rolled-back"),
    ],
)
def test_transaction_lifecycle(store: DuckDBStore, ending: str) -> None:
    """Empty transactions finish cleanly and terminal writes are rejected."""
    tx = store.transaction()
    getattr(tx, ending)()
    with pytest.raises(RuntimeError, match="writes are not allowed"):
        tx.store_transform(b"new", pl.DataFrame({"id": [1]}))
    with pytest.raises(RuntimeError, match="writes are not allowed"):
        tx.commit()
    with pytest.raises(RuntimeError, match="writes are not allowed"):
        tx.rollback()
    if ending == "rollback":
        with pytest.raises(RuntimeError, match="rolled back"):
            tx.has(b"new")
        with pytest.raises(RuntimeError, match="rolled back"):
            tx.stats()
    tx.close()
    assert not store.has(b"new")


@pytest.mark.parametrize(
    ("method", "args"),
    [
        pytest.param("publish", ("label", b"fp"), id="publish"),
        pytest.param("find", ("label",), id="find"),
        pytest.param("labels", (), id="labels"),
        pytest.param(
            "store_judgement", (Judgement(shown=[1], endorsed=[[1]]),), id="judgements"
        ),
        pytest.param("read_eval_data", (), id="evaluation"),
        pytest.param("sample", (b"fp", 1), id="sampling"),
        pytest.param("prune", (), id="pruning"),
        pytest.param("transaction", (), id="nested"),
    ],
)
def test_transaction_unsupported(
    store: DuckDBStore, method: str, args: tuple[object, ...]
) -> None:
    """Unsupported administration fails explicitly and never acts on the parent."""
    tx = store.transaction()
    try:
        with pytest.raises(NotImplementedError):
            getattr(tx, method)(*args)
        assert store.stats().artifacts == {}
        assert store.labels() == []
        assert all(table.is_empty() for table in store.read_eval_data())
    finally:
        tx.close()


def test_transaction_stats(store: DuckDBStore) -> None:
    """Staging stats do not checkpoint or count the parent Store."""
    store.store_transform(b"committed", pl.DataFrame({"id": [1]}))
    tx = store.transaction()
    try:
        with patch.object(store, "stats", side_effect=AssertionError("parent stats")):
            before = tx.stats()
            tx.store_transform(b"new", pl.DataFrame({"id": [2]}))
            assert tx.stats().describe(since=before) == "Transaction 1 staged artifact"
        assert isinstance(tx, Transaction)
    finally:
        tx.close()

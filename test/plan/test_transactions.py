"""Ordinary branching DAG execution against a provisional Store overlay."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

import matchlab as mb
from matchlab.stores import DuckDBStore, Store
from test.conftest import SourceFactory


def _plan(source: SourceFactory) -> tuple[mb.Resolver, mb.Source, mb.Source]:
    crn, dh = source("crn"), source("dh")
    cleaned = crn.clean({"name": "crn_company"})
    dedupe = cleaned.dedupe(mb.NaiveDeduper, model_settings={"unique_fields": ["name"]})
    link = cleaned.link(
        dh,
        mb.DeterministicLinker,
        model_settings={"comparisons": "l.name = r.dh_company"},
    )
    return dedupe.resolve(link), crn, dh


def _assert_entities(result: pl.DataFrame) -> None:
    ids = {(row["source"], row["key"]): row["root"] for row in result.to_dicts()}
    assert len(ids) == 5
    assert ids["crn", "a1"] == ids["crn", "a2"] == ids["dh", "b1"]
    assert len({ids["crn", "a1"], ids["crn", "a3"], ids["dh", "b2"]}) == 3


def _collect_cached(root: mb.Step, store: Store) -> None:
    with ExitStack() as patches:
        for step in root.lineage():
            patches.enter_context(
                patch.object(step, "_execute", side_effect=AssertionError("cache miss"))
            )
        root.collect(store=store)


def test_transaction_dag_overlay(source: SourceFactory, store: DuckDBStore) -> None:
    """A shared transform runs once, new downstream artifacts stay provisional."""
    apex, crn, _dh = _plan(source)
    crn.collect(store=store)
    committed = crn.fingerprints()
    original = crn.data()
    tx = store.transaction()
    try:
        transform = apex.parents[0].parents[0]
        with patch.object(transform, "_execute", wraps=transform._execute) as execute:
            apex.collect(store=tx)
            assert execute.call_count == 1
        _assert_entities(apex.entities())
        assert tx.stats().artifacts == {
            "source": 1,
            "transform": 1,
            "model": 2,
            "resolver": 1,
        }
        for fp in apex.fingerprints() - committed:
            assert tx.has(fp)
            assert not store.has(fp)
        _collect_cached(apex, tx)
        tx.rollback()
        assert store.stats().artifacts == {"source": 1}
        assert crn.collect(store=store).data().equals(original)
        with pytest.raises(RuntimeError, match="rolled back"):
            apex.entities()
        _assert_entities(apex.collect(store=store).entities())
    finally:
        tx.close()


def test_transaction_dag_resolver_input(
    source: SourceFactory, store: DuckDBStore
) -> None:
    """A staged resolver joins to source records held by the parent."""
    apex, crn, _dh = _plan(source)
    crn.collect(store=store)
    tx = store.transaction()
    try:
        continuation = (
            apex.clean({"name": "coalesce(crn_company, dh_company)"})
            .dedupe(mb.NaiveDeduper, model_settings={"unique_fields": ["name"]})
            .resolve()
        )
        continuation.collect(store=tx)
        _assert_entities(continuation.entities())
        assert continuation.data().height == 5
        tx.rollback()
        assert store.stats().artifacts == {"source": 1}
    finally:
        tx.close()


def test_transaction_dag_failure(source: SourceFactory, store: DuckDBStore) -> None:
    """A failed downstream step cannot leak earlier staged work into its parent."""
    apex, crn, _dh = _plan(source)
    crn.collect(store=store)
    tx = store.transaction()
    try:
        with (
            patch.object(apex, "_execute", side_effect=ValueError("resolver failed")),
            pytest.raises(ValueError, match="resolver failed"),
        ):
            apex.collect(store=tx)
        assert sum(tx.stats().artifacts.values()) == 4
        tx.rollback()
        assert store.stats().artifacts == {"source": 1}
    finally:
        tx.close()


def test_transaction_dag_commit(
    source: SourceFactory, store: DuckDBStore, caplog: pytest.LogCaptureFixture
) -> None:
    """Committed results stay readable and the parent collection uses its cache."""
    apex, crn, _dh = _plan(source)
    crn.collect(store=store)
    tx = store.transaction()
    try:
        apex.collect(store=tx, interactive=False)
        before = apex.entities()
        assert "Transaction 5 staged artifacts" in caplog.text
        tx.commit()
        assert apex.entities().equals(before)
        assert all(store.has(fp) for fp in apex.fingerprints())
        _collect_cached(apex, store)
        assert apex.entities().equals(before)
    finally:
        tx.close()


def test_transaction_dag_durable(source: SourceFactory, tmp_path: Path) -> None:
    """Promoted resolver outputs and source generations survive reopening."""
    path = tmp_path / "transactions.duckdb"
    store = DuckDBStore(path)
    tx = store.transaction()
    apex, crn, _dh = _plan(source)
    try:
        crn.collect(store=store)
        apex.collect(store=tx)
        expected = apex.entities()
        fp = apex._collected()[1]
        sources = tx.resolver_output_sources(fp)
        tx.commit()
        tx.close()
    finally:
        tx.close()
        store.close()
    reopened = DuckDBStore(path)
    try:
        assert reopened.read_resolver(fp).equals(expected)
        assert reopened.resolver_output_sources(fp) == sources
        for name, source_fp in sources.items():
            keys = expected.filter(pl.col("source") == name)["key"]
            rows, qualified_key = reopened.read_source_records(source_fp, name, keys)
            assert set(rows[qualified_key]) == set(keys)
        _assert_entities(apex.collect(store=reopened).entities())
    finally:
        reopened.close()

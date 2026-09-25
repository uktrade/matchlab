"""The incoming record is caller data aligned with a collected source."""

from collections.abc import Callable
from unittest.mock import patch

import polars as pl
import pytest
from pydantic import ValidationError

from matchlab import Source, read_dataframe
from matchlab.record import Record
from matchlab.stores import DuckDBStore


def test_record_identity(source: Callable[..., Source], store: DuckDBStore) -> None:
    """Equal content gets the collected leaf even though its transient key differs."""
    crn = source("crn").collect()
    values = {"company": "acme", "town": "london"}
    first = Record.from_input(crn, values, store)
    second = Record.from_input(crn, values, store)
    baseline = crn._read_cache(store)
    leaf = crn.identifiers(store).filter(pl.col("key") == "a1")["id"].item()

    assert first.data.schema == baseline.schema
    assert first.data["id"].item() == leaf
    assert first.ids == second.ids == {leaf}
    assert first.source is crn
    assert first.source_data["pk"].item() != second.source_data["pk"].item()
    assert first.source_data.select("company", "town").row(0) == ("acme", "london")


def test_record_casts(store: DuckDBStore) -> None:
    """A compatible input value takes the type of the collected extract."""
    source = read_dataframe(
        "numbers",
        df=pl.DataFrame({"key": ["a"], "count": pl.Series([1], dtype=pl.Int32)}),
        key_field="key",
    ).collect()

    record = Record.from_input(source, {"count": "2"}, store)

    assert (
        record.source_data.schema
        == store.read_source_extract(source._collected()[1]).schema
    )
    assert record.source_data["count"].item() == 2
    assert record.data.schema["id"] == pl.UInt64


@pytest.mark.parametrize(
    ("values", "message"),
    [
        pytest.param({"company": "acme"}, "missing fields", id="missing"),
        pytest.param(
            {"company": "acme", "town": "leeds", "extra": 1},
            "unexpected fields",
            id="extra",
        ),
        pytest.param(
            {"company": "acme", "town": "leeds", "pk": "custom"},
            "unexpected fields",
            id="source-key",
        ),
    ],
)
def test_record_rejects_fields(
    source: Callable[..., Source],
    store: DuckDBStore,
    values: dict[str, object],
    message: str,
) -> None:
    """Unknown or omitted fields cannot silently change the source evidence."""
    crn = source("crn").collect()
    with pytest.raises(ValueError, match=message):
        Record.from_input(crn, values, store)


def test_record_rejects_value(store: DuckDBStore) -> None:
    """An invalid cast names the offending field rather than yielding a null."""
    source = read_dataframe(
        "numbers", df=pl.DataFrame({"key": ["a"], "count": [1]}), key_field="key"
    ).collect()

    with pytest.raises(ValueError, match="invalid value for 'count'"):
        Record.from_input(source, {"count": "not a number"}, store)


def test_record_rejects_uncollected(
    source: Callable[..., Source], store: DuckDBStore
) -> None:
    """Building a record does not collect its source as a side effect."""
    crn = source("crn")
    with pytest.raises(RuntimeError, match="has not been collected"):
        Record.from_input(crn, {"company": "acme", "town": "leeds"}, store)


def test_record_no_origin_read(
    source: Callable[..., Source], store: DuckDBStore
) -> None:
    """The stored extract supplies the schema even when the warehouse is gone."""
    crn = source("crn").collect()

    with patch.object(
        type(crn.location), "read", side_effect=AssertionError("origin read")
    ):
        Record.from_input(crn, {"company": "acme", "town": "leeds"}, store)


def test_record_no_writes(source: Callable[..., Source], store: DuckDBStore) -> None:
    """A transient record leaves both stored source artifacts unchanged."""
    crn = source("crn").collect()
    fp = crn._collected()[1]
    extract = store.read_source_extract(fp)
    leaves = store.read_source_leaves(fp)
    artifacts = store.stats().artifacts

    Record.from_input(crn, {"company": "acme", "town": "leeds"}, store)

    assert store.read_source_extract(fp).equals(extract)
    assert store.read_source_leaves(fp).equals(leaves)
    assert store.stats().artifacts == artifacts


def test_record_immutable(source: Callable[..., Source], store: DuckDBStore) -> None:
    """Public state cannot mutate a record or its branch evidence."""
    crn = source("crn").collect()
    record = Record.from_input(crn, {"company": "acme", "town": "leeds"}, store)
    data = record.data
    source_data = record.source_data

    with pytest.raises(ValidationError):
        record.ids = frozenset()
    with pytest.raises(AttributeError):
        record.ids.add(42)
    with pytest.raises(AttributeError, match="Record is immutable"):
        record._data = None
    with pytest.raises(TypeError):
        record.edges[object()] = pl.DataFrame()

    assert data is not None
    data.drop_in_place("id")
    source_data.drop_in_place("company")
    assert "id" in record.data.columns
    assert "company" in record.source_data.columns

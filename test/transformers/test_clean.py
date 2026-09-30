"""Tests for `Clean`, keep-all derivation with DuckDB SQL.

`Clean` derives the named columns *without dropping* the rest, which is the invariant
that distinguishes it from `Select`.
"""

import polars as pl
import pytest
from pydantic import ValidationError
from sqlglot.errors import ParseError

from matchlab.transformers import Clean


def test_clean_derives_and_keeps_unreferenced() -> None:
    """A derived column is added. Every column it doesn't name survives untouched."""
    data = pl.DataFrame(
        {"id": [1, 2, 3], "crn_company": ["A", "B", "C"], "crn_town": ["x", "y", "z"]}
    )

    result = (
        Clean(cleaning={"name": "lower(crn_company)"})
        .prepare(data)
        .apply(data, baseline=data)
    )

    assert set(result.columns) == {"id", "crn_company", "crn_town", "name"}
    assert result["name"].to_list() == ["a", "b", "c"]
    assert result["crn_town"].to_list() == ["x", "y", "z"]


def test_clean_alias_replaces_in_place() -> None:
    """An expression aliased to an existing column overwrites it, not duplicates it."""
    data = pl.DataFrame({"id": [1, 2], "crn_company": ["A", "B"]})

    result = (
        Clean(cleaning={"crn_company": "lower(crn_company)"})
        .prepare(data)
        .apply(data, baseline=data)
    )

    assert result.columns.count("crn_company") == 1
    assert result["crn_company"].to_list() == ["a", "b"]


def test_clean_multi_column_expression() -> None:
    """One expression may combine several columns."""
    data = pl.DataFrame(
        {"id": [1, 2], "first": ["John", "Jane"], "last": ["Doe", "Smith"]}
    )

    result = (
        Clean(cleaning={"name": "first || ' ' || last"})
        .prepare(data)
        .apply(data, baseline=data)
    )

    assert result["name"].to_list() == ["John Doe", "Jane Smith"]


def test_clean_invalid_sql_raises() -> None:
    """Invalid cleaning SQL fails at apply time, not silently."""
    data = pl.DataFrame({"id": [1, 2], "crn_company": ["A", "B"]})
    with pytest.raises(ParseError):
        Clean(cleaning={"x": "foo bar baz"}).prepare(data).apply(data, baseline=data)


def test_clean_empty_rejected() -> None:
    """A `Clean` with nothing to derive is meaningless. 'No cleaning' is no `Clean`."""
    with pytest.raises(ValidationError):
        Clean(cleaning={})

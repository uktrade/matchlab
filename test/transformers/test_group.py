"""Tests for `Group`, one row per `id`, each column an aggregate."""

import duckdb
import polars as pl
import pytest
from pydantic import ValidationError

from matchlab.transformers import Group


def test_group_one_row_per_id() -> None:
    """Aggregates decide how each column combines, per column, not wholesale."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 2],
            "crn_company": ["acme", "acme", "beta"],
            "crn_town": ["london", "leeds", "hull"],
        }
    )

    result = (
        Group(
            aggregates={
                "name": "any_value(crn_company)",
                "towns": "list(distinct crn_town)",
            }
        )
        .prepare(data)
        .apply(data, baseline=data)
        .sort("id")
    )

    assert result["id"].to_list() == [1, 2]
    assert result["name"].to_list() == ["acme", "beta"]
    assert sorted(result["towns"][0]) == ["leeds", "london"]
    assert result["towns"][1].to_list() == ["hull"]


def test_group_non_aggregate_raises() -> None:
    """DuckDB names the offending column, a better error than we would write."""
    data = pl.DataFrame({"id": [1, 1], "crn_town": ["london", "leeds"]})

    with pytest.raises(duckdb.BinderException, match="crn_town"):
        Group(aggregates={"crn_town": "crn_town"}).prepare(data).apply(
            data, baseline=data
        )


def test_group_empty_rejected() -> None:
    """Grouping with no aggregates to say how columns combine is rejected."""
    with pytest.raises(ValidationError):
        Group(aggregates={})

"""Tests for `Explode`, the true cross product `Group` doesn't offer."""

import polars as pl
import pytest
from pydantic import ValidationError

from matchlab.transformers import Explode


def test_explode_cross_product() -> None:
    """Each id's non-id columns combine as a true cross product, not a parallel zip."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1],
            "a": ["x1", "x2", None],
            "b": [None, "y1", None],
            "c": [None, None, "z1"],
        }
    )

    result = Explode().prepare(data).apply(data, baseline=data).sort("a")

    assert result["a"].to_list() == ["x1", "x2"]
    assert result["b"].to_list() == ["y1", "y1"]
    assert result["c"].to_list() == ["z1", "z1"]


def test_explode_deduplicates_values() -> None:
    """A value repeated across an id's rows counts once, not once per repeat."""
    data = pl.DataFrame({"id": [1, 1, 1], "a": ["x1", "x1", "x2"]})

    result = Explode().prepare(data).apply(data, baseline=data).sort("a")

    assert result["a"].to_list() == ["x1", "x2"]


def test_explode_keeps_id_without_value() -> None:
    """An id with no non-null value in a column keeps its row, rather than vanishing."""
    data = pl.DataFrame({"id": [1], "a": ["x1"], "b": [None]})

    result = Explode().prepare(data).apply(data, baseline=data)

    assert result.height == 1
    assert result["b"].to_list() == [None]


def test_explode_over_max_raises() -> None:
    """A combination count past `max_combinations` raises, naming the id."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "a": ["x1", "x2", "x1", "x2"],
            "b": ["y1", "y1", "y2", "y2"],
        }
    )

    with pytest.raises(ValueError, match="id 1"):
        Explode(max_combinations=3).prepare(data).apply(data, baseline=data)


def test_explode_non_positive_max_rejected() -> None:
    """A cap of zero or fewer would reject every group, so it isn't a valid cap."""
    with pytest.raises(ValidationError):
        Explode(max_combinations=0)

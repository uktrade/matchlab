"""Tests for the testkit's perfect matching methodologies."""

import polars as pl

from matchlab.testkit.matchers import AnswerKey, PerfectLinker


def test_link_both_affected() -> None:
    """Both additions match baseline and new rows, but not old-only pairs."""
    truth_id = AnswerKey(
        left_columns=("name",),
        left_groups={("shared",): 1},
        right_columns=("name",),
        right_groups={("shared",): 1},
    ).register()
    linker = PerfectLinker(truth_id=truth_id)
    left = pl.DataFrame({"id": [1], "name": ["shared"]})
    right = pl.DataFrame({"id": [10], "name": ["shared"]})
    prepared = linker.prepare(left, right)

    new_left = pl.DataFrame({"id": [2], "name": ["shared"]})
    new_right = pl.DataFrame({"id": [20], "name": ["shared"]})
    assert set(
        prepared.link(new_left, new_right, baseline_left=left, baseline_right=right)
        .select("left_id", "right_id")
        .rows()
    ) == {
        (2, 10),
        (1, 20),
        (2, 20),
    }
    assert prepared.link(new_left, baseline_left=left, baseline_right=right)[
        "right_id"
    ].to_list() == [10]


def test_link_same_id_variant() -> None:
    """An affected variant never exposes a baseline-only pair with the same ID."""
    truth_id = AnswerKey(
        left_columns=("name",),
        left_groups={("old",): 1, ("new",): 2},
        right_columns=("name",),
        right_groups={("old",): 1, ("new",): 2},
    ).register()
    linker = PerfectLinker(truth_id=truth_id)
    baseline_left = pl.DataFrame({"id": [1], "name": ["old"]})
    baseline_right = pl.DataFrame({"id": [10], "name": ["old"]})
    prepared = linker.prepare(baseline_left, baseline_right)

    results = prepared.link(
        left=pl.DataFrame({"id": [1], "name": ["new"]}),
        right=pl.DataFrame({"id": [20], "name": ["new"]}),
        baseline_left=baseline_left,
        baseline_right=baseline_right,
    )
    assert results.select("left_id", "right_id").rows() == [(1, 20)]

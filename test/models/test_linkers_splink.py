"""Tests for SplinkLinker input selection, scoring and schema errors."""

from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from splink import SettingsCreator
from splink import blocking_rule_library as brl
from splink import comparison_library as cl
from splink.internals.linker_components.training import LinkerTraining

from matchlab.models.linkers.splinklinker import SplinkLinker, SplinkLinkerFunction


@pytest.fixture
def splink_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Supply distinct candidates and a non-uniform baseline for term frequencies."""
    left = pl.DataFrame(
        {"id": [1, 2, 3], "name": ["alice", "bob", "alice"], "city": ["A", "B", "B"]}
    )
    right = pl.DataFrame(
        {"id": [4, 5, 6], "name": ["alice", "alice", "bob"], "city": ["A", "B", "A"]}
    )
    return left, right


def make_linker(
    threshold: float | None = None,
    training_functions: list[SplinkLinkerFunction] | None = None,
) -> SplinkLinker:
    """Use real Splink blocking, comparisons and term frequency adjustments."""
    return SplinkLinker(
        linker_training_functions=training_functions or [],
        linker_settings=SettingsCreator(
            link_type="link_only",
            blocking_rules_to_generate_predictions=[brl.block_on("name")],
            comparisons=[
                cl.ExactMatch("name").configure(term_frequency_adjustments=True),
                cl.ExactMatch("city"),
            ],
        ),
        threshold=threshold,
    )


@pytest.mark.parametrize(
    ("affected_left", "affected_right", "expected_pairs"),
    [
        pytest.param(True, False, {(1, 4), (1, 5)}, id="left"),
        pytest.param(False, True, {(1, 4), (3, 4)}, id="right"),
        pytest.param(True, True, {(1, 4), (1, 5), (3, 4)}, id="both"),
    ],
)
def test_splink_affected(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
    affected_left: bool,
    affected_right: bool,
    expected_pairs: set[tuple[int, int]],
) -> None:
    """Score each supplied side against both the baseline and supplied opposite."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    result = prepared.link(
        left=left.head(1) if affected_left else None,
        right=right.head(1) if affected_right else None,
        baseline_left=left,
        baseline_right=right,
    )
    assert set(result.select("left_id", "right_id").rows()) == expected_pairs
    assert result.height == len(expected_pairs)


def test_splink_baseline(splink_inputs: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """Full inputs still score the whole collection after an affected prediction."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    prepared.link(left.head(1), right.head(1), baseline_left=left, baseline_right=right)
    with patch.object(
        prepared.state.inference,
        "predict",
        wraps=prepared.state.inference.predict,
    ) as predict:
        result = prepared.link(
            left.clone(), right.clone(), baseline_left=left, baseline_right=right
        )
    assert predict.call_count == 1
    assert set(result.select("left_id", "right_id").rows()) == {
        (1, 4),
        (1, 5),
        (2, 6),
        (3, 4),
        (3, 5),
    }
    assert result.height == 5


def test_splink_baseline_changed_rows(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """A changed row with an existing ID must contribute its new score."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    baseline = prepared.link(left, right, baseline_left=left, baseline_right=right)
    changed_right = right.with_columns(
        pl.when(pl.col("id") == 4)
        .then(pl.lit("B"))
        .otherwise(pl.col("city"))
        .alias("city")
    )

    result = prepared.link(
        left, changed_right, baseline_left=left, baseline_right=right
    )
    baseline_score = baseline.filter(pl.col("left_id") == 3, pl.col("right_id") == 4)[
        "score"
    ].item()
    changed_score = result.filter(pl.col("left_id") == 3, pl.col("right_id") == 4)[
        "score"
    ].item()
    assert changed_score > baseline_score
    assert set(result.select("left_id", "right_id").rows()) == set(
        baseline.select("left_id", "right_id").rows()
    )


def test_splink_baseline_alternating_sides(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """Alternating affected sides must not leave stale tables or term frequencies."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    baseline = prepared.link(
        left, right, baseline_left=left, baseline_right=right
    ).sort("left_id", "right_id")

    cases = [
        (left.head(1), None, baseline.filter(pl.col("left_id") == 1)),
        (None, right.tail(1), baseline.filter(pl.col("right_id") == 6)),
        (
            left.tail(1),
            right.head(2),
            baseline.filter(
                (pl.col("left_id") == 3) | pl.col("right_id").is_in([4, 5])
            ),
        ),
        (None, right.head(1), baseline.filter(pl.col("right_id") == 4)),
        (left.head(2), None, baseline.filter(pl.col("left_id").is_in([1, 2]))),
    ]
    for affected_left, affected_right, expected in cases:
        assert_frame_equal(
            prepared.link(
                left=affected_left,
                right=affected_right,
                baseline_left=left,
                baseline_right=right,
            ).sort("left_id", "right_id"),
            expected,
        )
        assert_frame_equal(
            prepared.link(left, right, baseline_left=left, baseline_right=right).sort(
                "left_id", "right_id"
            ),
            baseline,
        )


def test_splink_term_frequencies(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """A first preview uses frequencies from the prepared inputs."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    affected = prepared.link(
        left.head(1), right.head(1), baseline_left=left, baseline_right=right
    ).filter(pl.col("left_id") == 1, pl.col("right_id") == 4)
    baseline_linker = make_linker()
    baseline_linker_prepared = baseline_linker.prepare(left, right)
    baseline = baseline_linker_prepared.link(
        left, right, baseline_left=left, baseline_right=right
    ).filter(pl.col("left_id") == 1, pl.col("right_id") == 4)
    assert affected["score"].to_list() == baseline["score"].to_list()


def test_splink_preparations_are_independent(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """A second preparation cannot replace the first backend's term frequencies."""
    left, right = splink_inputs
    linker = make_linker()
    settings = linker.model_dump_json()
    original_left, original_right = left.clone(), right.clone()
    first = linker.prepare(left, right)
    expected = first.link(
        left.head(1), right.head(1), baseline_left=left, baseline_right=right
    ).sort("left_id", "right_id")
    other_left, other_right = left.head(1), right.head(1)
    second = linker.prepare(other_left, other_right)
    alternate = second.link(
        other_left,
        other_right,
        baseline_left=other_left,
        baseline_right=other_right,
    )
    first_score = expected.filter(pl.col("left_id") == 1, pl.col("right_id") == 4)[
        "score"
    ].item()
    assert alternate["score"].item() != first_score
    assert first.state is not second.state
    for _ in range(2):
        assert_frame_equal(
            first.link(
                left.head(1), right.head(1), baseline_left=left, baseline_right=right
            ).sort("left_id", "right_id"),
            expected,
        )
        assert_frame_equal(
            second.link(
                other_left,
                other_right,
                baseline_left=other_left,
                baseline_right=other_right,
            ),
            alternate,
        )
    assert_frame_equal(left, original_left)
    assert_frame_equal(right, original_right)
    assert linker.model_dump_json() == settings
    assert not linker.__pydantic_private__


def test_splink_additions_preserve_baselines(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """New rows link across all three pairings without adding baseline-only edges."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    baseline = prepared.link(
        left, right, baseline_left=left, baseline_right=right
    ).sort("left_id", "right_id")
    new_left = pl.DataFrame({"id": [7], "name": ["alice"], "city": ["A"]})
    new_right = pl.DataFrame({"id": [8], "name": ["alice"], "city": ["B"]})

    result = prepared.link(
        new_left, new_right, baseline_left=left, baseline_right=right
    )
    affected_left_to_baseline_right = {(7, 4), (7, 5)}
    baseline_left_to_affected_right = {(1, 8), (3, 8)}
    affected_to_affected = {(7, 8)}
    result_pairs = result.select("left_id", "right_id").rows()
    assert set(result_pairs) == (
        affected_left_to_baseline_right
        | baseline_left_to_affected_right
        | affected_to_affected
    )
    assert len(result_pairs) == 5  # The three predictions must not duplicate edges.
    assert (1, 4) in baseline.select("left_id", "right_id").rows()
    assert (1, 4) not in result_pairs  # No baseline-only edges on preview.
    score_by_pair = {
        (row["left_id"], row["right_id"]): row["score"]
        for row in result.iter_rows(named=True)
    }
    baseline_scores = {
        (row["left_id"], row["right_id"]): row["score"]
        for row in baseline.iter_rows(named=True)
    }
    assert score_by_pair[7, 4] == baseline_scores[1, 4]
    assert score_by_pair[1, 8] == baseline_scores[1, 5]
    assert score_by_pair[7, 8] == baseline_scores[1, 5]
    assert_frame_equal(
        prepared.link(left, right, baseline_left=left, baseline_right=right).sort(
            "left_id", "right_id"
        ),
        baseline,
    )
    assert set(
        prepared.link(new_left, baseline_left=left, baseline_right=right)
        .select("left_id", "right_id")
        .rows()
    ) == {
        (7, 4),
        (7, 5),
    }
    assert set(
        prepared.link(right=new_right, baseline_left=left, baseline_right=right)
        .select("left_id", "right_id")
        .rows()
    ) == {
        (1, 8),
        (3, 8),
    }


def test_splink_additions_blocked_pairs(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """The Cartesian comparison of new rows must still obey Splink blocking."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    new_left = pl.DataFrame(
        {"id": [7, 8], "name": ["alice", "bob"], "city": ["A", "B"]}
    )
    new_right = pl.DataFrame(
        {"id": [9, 10], "name": ["bob", "alice"], "city": ["B", "A"]}
    )

    result = prepared.link(
        new_left, new_right, baseline_left=left, baseline_right=right
    )
    assert set(result.select("left_id", "right_id").rows()) == {
        (7, 4),
        (7, 5),
        (8, 6),
        (2, 9),
        (1, 10),
        (3, 10),
        (7, 10),
        (8, 9),
    }


@pytest.mark.parametrize(
    ("threshold", "expected_pairs"),
    [
        pytest.param(0.01, {(1, 4)}, id="filtered"),
        pytest.param(1.0, set(), id="exact"),
    ],
)
def test_splink_threshold(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
    threshold: float,
    expected_pairs: set[tuple[int, int]],
) -> None:
    """Affected previews retain strong matches and discard weaker ones."""
    left, right = splink_inputs
    linker = make_linker(threshold=threshold)
    prepared = linker.prepare(left, right)
    result = prepared.link(
        left.head(1), right.head(1), baseline_left=left, baseline_right=right
    )
    assert set(result.select("left_id", "right_id").rows()) == expected_pairs


def test_splink_threshold_boundary(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """A preview includes a match whose probability equals the threshold."""
    left = splink_inputs[0].select("id", "name")
    right = splink_inputs[1].select("id", "name")
    linker = SplinkLinker(
        linker_training_functions=[],
        linker_settings=SettingsCreator(
            link_type="link_only",
            probability_two_random_records_match=0.5,
            blocking_rules_to_generate_predictions=[brl.block_on("name")],
            comparisons=[
                cl.ExactMatch("name").configure(
                    m_probabilities=[0.5, 0.5],
                    u_probabilities=[0.5, 0.5],
                )
            ],
        ),
        threshold=0.5,
    )
    prepared = linker.prepare(left, right)

    result = prepared.link(left.head(1), baseline_left=left, baseline_right=right)
    assert set(result.select("left_id", "right_id").rows()) == {(1, 4), (1, 5)}
    assert result["score"].to_list() == [0.5, 0.5]


def test_splink_training(splink_inputs: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """Training on the full baseline is not repeated for affected rows."""
    left, right = splink_inputs
    linker = make_linker(
        training_functions=[
            SplinkLinkerFunction(
                function="estimate_probability_two_random_records_match",
                arguments={
                    "deterministic_matching_rules": "l.name = r.name",
                    "recall": 0.8,
                },
            )
        ]
    )
    prepared = linker.prepare(left, right)
    with patch.object(
        LinkerTraining,
        "estimate_probability_two_random_records_match",
        side_effect=AssertionError("training ran during link"),
    ):
        assert set(
            prepared.link(
                left.head(1), right.head(1), baseline_left=left, baseline_right=right
            )
            .select("left_id", "right_id")
            .rows()
        ) == {(1, 4), (1, 5), (3, 4)}


@pytest.mark.parametrize(
    ("side",),
    [
        pytest.param("left", id="left"),
        pytest.param("right", id="right"),
        pytest.param("both", id="both"),
    ],
)
def test_splink_empty_side(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame], side: str
) -> None:
    """Empty additions do not suppress matches from the other supplied side."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    result = prepared.link(
        left=left.head(0) if side in ("left", "both") else None,
        right=right.head(0) if side in ("right", "both") else None,
        baseline_left=left,
        baseline_right=right,
    )
    assert result.schema == {
        "left_id": pl.Int64,
        "right_id": pl.Int64,
        "score": pl.Float32,
    }
    assert result.is_empty()


def test_splink_one_empty_addition(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    """An empty supplied side still scores additions on the other side."""
    left, right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(left, right)
    assert set(
        prepared.link(
            left.head(0), right.head(1), baseline_left=left, baseline_right=right
        )
        .select("left_id", "right_id")
        .rows()
    ) == {(1, 4), (3, 4)}
    assert set(
        prepared.link(
            left.head(1), right.head(0), baseline_left=left, baseline_right=right
        )
        .select("left_id", "right_id")
        .rows()
    ) == {(1, 4), (1, 5)}


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        pytest.param(None, None, "at least one affected", id="no_side"),
        pytest.param("bad_schema", None, "conformant", id="left_schema"),
        pytest.param(None, "bad_schema", "conformant", id="right_schema"),
    ],
)
def test_splink_rejects_inputs(
    splink_inputs: tuple[pl.DataFrame, pl.DataFrame],
    left: str | None,
    right: str | None,
    message: str,
) -> None:
    """Reject calls without a supplied side or matching schemas."""
    baseline_left, baseline_right = splink_inputs
    linker = make_linker()
    prepared = linker.prepare(baseline_left, baseline_right)
    replacements = {
        "bad_schema": baseline_left.drop("city"),
    }
    with pytest.raises(ValueError, match=message):
        prepared.link(
            replacements[left] if left is not None else None,
            replacements[right] if right is not None else None,
            baseline_left=baseline_left,
            baseline_right=baseline_right,
        )


def test_splink_conformancy_error() -> None:
    """Splink surfaces the concrete schema mismatch from the real prepare path."""
    linker = SplinkLinker(
        left_id="id",
        right_id="id",
        linker_training_functions=[],
        linker_settings=SettingsCreator(
            link_type="link_only",
            blocking_rules_to_generate_predictions=[brl.block_on("id")],
            comparisons=[cl.ExactMatch("id")],
        ),
        threshold=None,
    )
    left = pl.DataFrame({"id": [1], "company_name": ["left"]})
    right = pl.DataFrame({"id": [2], "postcode": ["SW1A 1AA"]})

    with pytest.raises(
        ValueError,
        match="SplinkLinker requires input data to be conformant",
    ) as exc_info:
        linker.prepare(left, right)

    message = str(exc_info.value)
    assert "Left schema: id=Int64, company_name=String" in message
    assert "Right schema: id=Int64, postcode=String" in message
    assert "Columns only on left: ['company_name']" in message
    assert "Columns only on right: ['postcode']" in message

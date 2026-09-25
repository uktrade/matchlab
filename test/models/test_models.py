"""Model-node execution and score normalisation."""

from collections.abc import Callable
from typing import ClassVar
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import PrivateAttr

from matchlab import Source
from matchlab.models import Model
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.dedupers.base import Deduper
from matchlab.models.models import add_model_class, normalise_model_scores
from matchlab.record import Record
from matchlab.stores import DuckDBStore


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        pytest.param(
            pl.DataFrame(
                [
                    {"left_id": 4, "right_id": 5, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            pl.DataFrame([{"left_id": 4, "right_id": 5, "score": 1.0}]),
            id="a-pair-scored-twice-keeps-the-highest",
        ),
        pytest.param(
            pl.DataFrame(
                [
                    {"left_id": 5, "right_id": 4, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            pl.DataFrame([{"left_id": 4, "right_id": 5, "score": 1.0}]),
            id="a-reversed-pair-is-the-same-pair",
        ),
        pytest.param(
            pl.DataFrame(
                [
                    {"left_id": 4, "right_id": 6, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            pl.DataFrame(
                [
                    {"left_id": 4, "right_id": 6, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            id="distinct-pairs-are-both-kept",
        ),
    ],
)
def test_normalise_one_edge_per_pair(
    scores: pl.DataFrame, expected: pl.DataFrame
) -> None:
    """An unordered pair collapses to one edge, keeping the top score.

    `(a, b)` and `(b, a)` count as the same pair.
    """
    assert_frame_equal(
        normalise_model_scores(scores),
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )


def test_normalise_empty_input() -> None:
    """No edges is a valid answer, so an empty table normalises to the edge schema."""
    empty = pl.DataFrame(
        {"left_id": [], "right_id": [], "score": []},
        schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float64},
    )
    result = normalise_model_scores(empty)
    assert result.height == 0
    assert set(result.columns) == {"left_id", "right_id", "score"}


def test_normalise_rejects_non_dataframe() -> None:
    """A methodology must hand back a table, not a list of dicts or None."""
    with pytest.raises(ValueError, match="Expected a polars DataFrame"):
        normalise_model_scores([{"left_id": 1, "right_id": 2, "score": 0.5}])


def test_normalise_rejects_wrong_columns() -> None:
    """Missing the score column names what was expected against what was found."""
    bad = pl.DataFrame({"left_id": [1], "right_id": [2]})  # no score
    with pytest.raises(ValueError, match="Expected.*score"):
        normalise_model_scores(bad)


def test_normalise_rejects_non_numeric_score() -> None:
    """A score column that isn't numeric can't be a probability."""
    bad = pl.DataFrame({"left_id": [1], "right_id": [2], "score": ["high"]})
    with pytest.raises(ValueError, match="numeric values in the range"):
        normalise_model_scores(bad)


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(1.5, id="above-one"),
        pytest.param(-0.5, id="below-zero"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_normalise_rejects_out_of_range_score(score: float) -> None:
    """A score outside `[0.0, 1.0]` is a misconfigured methodology, not a match."""
    bad = pl.DataFrame({"left_id": [1], "right_id": [2], "score": [score]})
    with pytest.raises(ValueError, match="Score range misconfigured"):
        normalise_model_scores(bad)


def test_add_model_class_rejects_non_subclass() -> None:
    """Only a Deduper or Linker subclass can be registered as a model class."""
    with pytest.raises(ValueError, match="not a subclass of Deduper or Linker"):
        add_model_class(str)


def test_add_model_class_registers_a_subclass() -> None:
    """A real subclass registers under its name, so a plan can name it as a string."""
    add_model_class(NaiveDeduper)  # already known, so this call is idempotent
    from matchlab.models.models import _MODEL_CLASSES  # noqa: PLC0415

    assert _MODEL_CLASSES["NaiveDeduper"] is NaiveDeduper


def _dedupe(source: Source) -> Model:
    """Build the small deduper plan used by model record tests."""
    return source.dedupe("NaiveDeduper", {"unique_fields": ["crn_company"]})


def _link(left: Source, right: Source) -> Model:
    """Build the cross-source linker plan used by model record tests."""
    return left.link(
        right,
        "DeterministicLinker",
        {"comparisons": ["l.crn_company = r.dh_company"]},
    )


def test_model_record_dedupe(source: Callable[..., Source], store: DuckDBStore) -> None:
    """The model passes affected data through and records touched edge IDs."""
    crn = source("crn")
    model = _dedupe(crn).collect()
    record = Record.from_input(crn, {"company": "acme", "town": "oxford"}, store)
    affected_id = next(iter(record.ids))
    scores = pl.DataFrame({"left_id": [affected_id], "right_id": [42], "score": [1.0]})

    with patch.object(
        type(model.model_instance), "dedupe", return_value=scores
    ) as dedupe:
        result = model._execute_record(store, left=record)

    dedupe.assert_called_once()
    assert dedupe.call_args.args[0].equals(record.data)
    assert result.edges[model].rows() == [(affected_id, 42, 1.0)]
    assert result.edges[model].schema == {
        "left_id": pl.UInt64,
        "right_id": pl.UInt64,
        "score": pl.Float32,
    }
    assert result.ids == record.ids | {42}
    assert result.data is None


@pytest.mark.parametrize(
    ("side",),
    [
        pytest.param("left", id="left-only"),
        pytest.param("right", id="right-only"),
        pytest.param("both", id="both-sides"),
    ],
)
def test_model_record_link_routes(
    source: Callable[..., Source], store: DuckDBStore, side: str
) -> None:
    """The model passes each affected-side shape to one linker call."""
    crn, dh = source("crn"), source("dh")
    if side == "both":
        model = crn.link(
            crn,
            "DeterministicLinker",
            {"comparisons": ["l.crn_company = r.crn_company"]},
        ).collect()
        record = Record.from_input(crn, {"company": "acme", "town": "oxford"}, store)
        left, right = record, record
    else:
        model = _link(crn, dh).collect()
        record_source = crn if side == "left" else dh
        record = Record.from_input(
            record_source, {"company": "acme", "town": "oxford"}, store
        )
        left = record if side == "left" else None
        right = record if side == "right" else None
    scores = pl.DataFrame({"left_id": [41], "right_id": [42], "score": [1.0]})

    with patch.object(type(model.model_instance), "link", return_value=scores) as link:
        result = model._execute_record(store, left=left, right=right)

    link.assert_called_once()
    actual_left = link.call_args.kwargs["left"]
    actual_right = link.call_args.kwargs["right"]
    assert (actual_left is None) == (left is None)
    assert (actual_right is None) == (right is None)
    if actual_left is not None:
        assert actual_left.equals(record.data)
    if actual_right is not None:
        assert actual_right.equals(record.data)
    assert result.ids == record.ids | {41, 42}
    assert result.edges[model].height == 1
    assert result.data is None


def test_model_record_empty(source: Callable[..., Source], store: DuckDBStore) -> None:
    """No scored edge still leaves the new ID available for merge-forward."""
    crn = source("crn")
    model = _dedupe(crn).collect()
    record = Record.from_input(crn, {"company": "novel", "town": "oxford"}, store)

    with patch.object(
        type(model.model_instance),
        "dedupe",
        return_value=pl.DataFrame(
            schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float32}
        ),
    ):
        result = model._execute_record(store, left=record)

    assert result.ids == record.ids
    assert result.edges[model].is_empty()
    assert result.edges[model].schema["score"] == pl.Float32


def test_model_record_branches(
    source: Callable[..., Source], store: DuckDBStore
) -> None:
    """Evidence from both inputs survives without changing either input ledger."""
    crn = source("crn")
    model = crn.link(
        crn, "DeterministicLinker", {"comparisons": ["l.crn_company = r.crn_company"]}
    ).collect()
    prior_left = _dedupe(crn)
    prior_right = _dedupe(crn)
    record = Record.from_input(crn, {"company": "novel", "town": "oxford"}, store)
    empty_edges = pl.DataFrame(
        schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float32}
    )
    left = Record(
        data=record.data,
        ids=record.ids,
        source=record.source,
        source_data=record.source_data,
        edges={prior_left: empty_edges},
    )
    right = Record(
        data=record.data,
        ids=record.ids,
        source=record.source,
        source_data=record.source_data,
        edges={prior_right: empty_edges},
    )

    with patch.object(
        type(model.model_instance),
        "link",
        return_value=empty_edges,
    ):
        result = model._execute_record(store, left=left, right=right)

    assert set(result.edges) == {prior_left, prior_right, model}
    assert set(left.edges) == {prior_left}
    assert set(right.edges) == {prior_right}
    assert result.edges[prior_left] is not left.edges[prior_left]
    assert result.data is None


def test_model_record_no_writes(
    source: Callable[..., Source], store: DuckDBStore
) -> None:
    """Scoring one row neither persists its edges nor reads a source location."""
    crn = source("crn")
    model = _dedupe(crn).collect()
    record = Record.from_input(crn, {"company": "acme", "town": "oxford"}, store)
    artifacts = store.stats().artifacts
    baseline = model.edges()

    with (
        patch.object(
            type(crn.location), "read", side_effect=AssertionError("origin read")
        ),
        patch.object(
            type(model.model_instance),
            "dedupe",
            return_value=pl.DataFrame(
                schema={
                    "left_id": pl.UInt64,
                    "right_id": pl.UInt64,
                    "score": pl.Float32,
                }
            ),
        ),
    ):
        model._execute_record(store, left=record)

    assert store.stats().artifacts == artifacts
    assert model.edges().equals(baseline)


class CountingDeduper(Deduper):
    """Expose preparation calls to check the model step's cache lifecycle."""

    version: ClassVar[int] = 1

    _calls: int = PrivateAttr(default=0)
    _fail_once: bool = PrivateAttr(default=False)

    def prepare(self, data: pl.DataFrame) -> None:
        """Count attempts, with an optional failure to test retry."""
        self._calls += 1
        if self._fail_once:
            self._fail_once = False
            raise ValueError("baseline unavailable")

    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return no edges so preparation remains the only variable."""
        return pl.DataFrame(
            schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float32}
        )


def test_model_prepare_cache(source: Callable[..., Source]) -> None:
    """A cache hit prepares on demand, then retries cleanly after a failure."""
    crn = source("crn")
    crn.dedupe(CountingDeduper).collect()
    cached = crn.dedupe(CountingDeduper).collect()

    assert cached.model_instance._calls == 0
    cached.model_instance._fail_once = True
    with pytest.raises(ValueError, match="baseline unavailable"):
        cached._ensure_prepared()
    assert not cached._prepared

    cached._ensure_prepared()
    cached._ensure_prepared()
    assert cached._prepared
    assert cached.model_instance._calls == 2


def test_model_record_unprepared(
    source: Callable[..., Source], store: DuckDBStore
) -> None:
    """A cached artifact alone cannot score without preparing its methodology."""
    crn = source("crn")
    _dedupe(crn).collect()
    cached = _dedupe(crn).collect()
    record = Record.from_input(crn, {"company": "acme", "town": "oxford"}, store)

    with pytest.raises(RuntimeError, match="not prepared"):
        cached._execute_record(store, left=record)

    cached._ensure_prepared()
    empty_edges = pl.DataFrame(
        schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float32}
    )
    with patch.object(
        type(cached.model_instance), "dedupe", return_value=empty_edges
    ) as dedupe:
        assert cached._execute_record(store, left=record).edges[cached].is_empty()
    dedupe.assert_called_once()


@pytest.mark.parametrize(
    ("kind",),
    [
        pytest.param("deduper-right", id="deduper-right"),
        pytest.param("different-source", id="different-source"),
    ],
)
def test_model_record_rejects_input(
    source: Callable[..., Source], store: DuckDBStore, kind: str
) -> None:
    """A missing or incompatible affected side cannot masquerade as valid work."""
    crn, dh = source("crn"), source("dh")
    model = (
        _dedupe(crn).collect() if kind == "deduper-right" else _link(crn, dh).collect()
    )
    left = Record.from_input(crn, {"company": "acme", "town": "oxford"}, store)
    right = (
        Record.from_input(dh, {"company": "acme", "town": "oxford"}, store)
        if kind == "different-source"
        else left
    )
    inputs = {
        "deduper-right": {"right": right},
        "different-source": {"left": left, "right": right},
    }

    with pytest.raises(ValueError):
        model._execute_record(store, **inputs[kind])

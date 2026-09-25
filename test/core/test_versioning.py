"""Tests for what a methodology contributes to its step's fingerprint."""

from typing import ClassVar

import polars as pl
import pytest

from matchlab.core.versioning import declared_version, methodology_key
from matchlab.transformers import Transformer


class _Unversioned(Transformer):
    """A transformer promising nothing about its own code."""

    def prepare(self, data: pl.DataFrame) -> None:
        """No baseline state is needed."""

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return data


def test_versioning_key_nonce_differs() -> None:
    """An unversioned methodology contributes fresh bytes every time it is asked.

    This is what re-keys the step on every collect, so it must not be stable the way
    the rest of a spec key is.
    """
    assert methodology_key(_Unversioned) != methodology_key(_Unversioned)


def test_versioning_shadowed_by_field() -> None:
    """Declaring `version` as a setting raises, rather than quietly meaning nothing.

    Pydantic lets a field shadow the inherited class attribute with only a warning, and
    it strips the field default from the class body, so the class reads back as
    unversioned. The step would then refresh on every collect however the setting was
    filled in, and nothing would say why.
    """
    with pytest.warns(UserWarning, match="shadows an attribute"):

        class _Shadowed(Transformer):
            version: int = 1  # a field, not a ClassVar

            def prepare(self, data: pl.DataFrame) -> None:
                """No baseline state is needed."""

            def apply(self, data: pl.DataFrame) -> pl.DataFrame:
                return data

    with pytest.raises(TypeError, match="declares `version` as a setting"):
        declared_version(_Shadowed)


def test_versioning_not_an_int() -> None:
    """A version that is not an int raises, rather than reading as a promise.

    A string version is the likely slip, since a version looks like a release number.
    """

    class _Semver(_Unversioned):
        version: ClassVar[str] = "2.1"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="holds str"):
        declared_version(_Semver)


def test_versioning_no_methodology() -> None:
    """A step keyed by something other than code adds nothing.

    `Source` is the case: it hashes the rows it read, so its location declares no
    version and must not be given a nonce, which would re-read the warehouse forever.
    """
    assert methodology_key(None) == b""


def test_versioning_key_tracks_version() -> None:
    """A versioned methodology contributes its version, so a bump moves the key."""

    class _Versioned(_Unversioned):
        version: ClassVar[int] = 1

    first = methodology_key(_Versioned)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(_Versioned, "version", 2)
        assert methodology_key(_Versioned) != first
    assert methodology_key(_Versioned) == first

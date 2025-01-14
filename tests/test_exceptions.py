"""Tests for hypofuse.exceptions."""

from __future__ import annotations

from hypofuse.exceptions import (
    AlignmentError,
    CalibrationError,
    DuplicateIdError,
    FusionError,
    HypofuseError,
    LanguageModelError,
    SchemaError,
)


def test_hypofuse_error_is_base() -> None:
    for cls in (
        SchemaError,
        DuplicateIdError,
        AlignmentError,
        FusionError,
        LanguageModelError,
        CalibrationError,
    ):
        assert issubclass(cls, HypofuseError)


def test_schema_error_carries_message() -> None:
    err = SchemaError("missing field id")
    assert str(err) == "missing field id"


def test_duplicate_id_error_distinct() -> None:
    err = DuplicateIdError("dup id utt_1")
    assert isinstance(err, HypofuseError)
    assert "dup" in str(err)


def test_subclass_caught_as_base() -> None:
    try:
        raise AlignmentError("boom")
    except HypofuseError as exc:
        assert isinstance(exc, AlignmentError)

"""Tests for from_manifest_row / to_manifest_row roundtrip."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import SchemaError
from hypofuse.timings import (
    TimingTrack,
    TokenTiming,
    from_manifest_row,
    to_manifest_row,
)


def _t(token: str, start: float, end: float) -> TokenTiming:
    return TokenTiming(token=token, start_s=start, end_s=end)


def test_roundtrip_exact_equality() -> None:
    original = TimingTrack(tokens=(_t("hello", 0.0, 0.5), _t("world", 0.6, 1.2)))
    row = to_manifest_row(original)
    restored = from_manifest_row(row)
    assert restored.tokens == original.tokens


def test_roundtrip_empty_track() -> None:
    original = TimingTrack()
    row = to_manifest_row(original)
    restored = from_manifest_row(row)
    assert restored.tokens == ()


def test_to_manifest_row_shape() -> None:
    track = TimingTrack(tokens=(_t("a", 0.0, 1.0),))
    row = to_manifest_row(track)
    assert row == {"tokens": ["a"], "timings": [{"start_s": 0.0, "end_s": 1.0}]}


def test_from_manifest_row_missing_tokens() -> None:
    with pytest.raises(SchemaError, match="tokens"):
        from_manifest_row({"timings": []})


def test_from_manifest_row_missing_timings() -> None:
    with pytest.raises(SchemaError, match="timings"):
        from_manifest_row({"tokens": []})


def test_from_manifest_row_length_mismatch() -> None:
    with pytest.raises(SchemaError, match="same length"):
        from_manifest_row({"tokens": ["a"], "timings": []})


def test_from_manifest_row_bad_token_type() -> None:
    with pytest.raises(SchemaError, match="string"):
        from_manifest_row({"tokens": [123], "timings": [{"start_s": 0, "end_s": 1}]})


def test_from_manifest_row_bad_timing_shape() -> None:
    with pytest.raises(SchemaError, match="start_s"):
        from_manifest_row({"tokens": ["a"], "timings": [{"start_s": 0}]})


def test_from_manifest_row_not_a_dict() -> None:
    with pytest.raises(SchemaError, match="dict"):
        from_manifest_row("not a dict")


def test_from_manifest_row_rejects_invalid_timings() -> None:
    # end < start should be caught by TokenTiming.validate via track.validate
    with pytest.raises(ValueError, match="end_s must not precede"):
        from_manifest_row({"tokens": ["a"], "timings": [{"start_s": 2.0, "end_s": 1.0}]})

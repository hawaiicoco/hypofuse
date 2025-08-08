"""Tests for runmeta.stamp."""

from __future__ import annotations

from datetime import datetime

from hypofuse.runmeta import capture, stamp


def test_stamp_adds_iso_timestamp() -> None:
    meta = capture(seed=1)
    when = datetime(2026, 9, 26, 12, 30, 45)
    stamped = stamp(meta, when)
    assert stamped.stamped_at == "2026-09-26T12:30:45"


def test_stamp_preserves_other_fields() -> None:
    meta = capture(seed=1, label="preserve")
    when = datetime(2026, 1, 1, 0, 0, 0)
    stamped = stamp(meta, when)
    assert stamped.seed == meta.seed
    assert stamped.created_from == meta.created_from
    assert stamped.stamped_at is not None


def test_stamp_returns_new_instance() -> None:
    meta = capture(seed=1)
    when = datetime(2026, 1, 1)
    stamped = stamp(meta, when)
    assert stamped is not meta
    assert meta.stamped_at is None


def test_no_timestamp_by_default() -> None:
    meta = capture(seed=0)
    assert meta.stamped_at is None


def test_two_captures_identical_without_timestamp() -> None:
    meta1 = capture(seed=42, label="test")
    meta2 = capture(seed=42, label="test")
    assert meta1 == meta2

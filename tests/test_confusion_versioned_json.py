"""Versioned JSON serialization tests."""

from __future__ import annotations

import json

import pytest

from hypofuse.confusion import (
    build_confusion_network,
    confusion_from_json,
    confusion_to_json,
)
from hypofuse.multi_align import progressive_align


def test_json_has_schema_and_version() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid)
    payload = confusion_to_json(net)
    obj = json.loads(payload)
    assert obj["schema"] == "hypofuse.confusion"
    assert obj["schema_version"] == 1
    assert "slots" in obj


def test_byte_stable_roundtrip() -> None:
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid)
    payload = confusion_to_json(net)
    parsed = confusion_from_json(payload)
    assert confusion_to_json(parsed) == payload


def test_golden_json_text() -> None:
    # Hand-derived: slot 0 has "a" at posterior 1.0 (both hyps agree).
    # Slot 1 has "b" at 0.5 and "c" at 0.5 (one hyp each).
    grid = progressive_align([("a", "b"), ("a", "c")])
    net = build_confusion_network(grid)
    payload = confusion_to_json(net)
    expected = (
        '{"schema": "hypofuse.confusion", "schema_version": 1, '
        '"slots": [{"arcs": [{"posterior": 1.0, "token": "a"}], '
        '"pivot": "a"}, {"arcs": [{"posterior": 0.5, "token": "b"}, '
        '{"posterior": 0.5, "token": "c"}], "pivot": "b"}]}'
    )
    assert payload == expected


def test_reject_missing_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        confusion_from_json('{"schema_version": 1, "slots": []}')


def test_reject_missing_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        confusion_from_json('{"schema": "hypofuse.confusion", "slots": []}')


def test_reject_missing_slots() -> None:
    with pytest.raises(ValueError, match="slots"):
        confusion_from_json('{"schema": "hypofuse.confusion", "schema_version": 1}')


def test_reject_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unknown schema"):
        confusion_from_json('{"schema": "other.thing", "schema_version": 1, "slots": []}')


def test_reject_wrong_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        confusion_from_json('{"schema": "hypofuse.confusion", "schema_version": 99, "slots": []}')


def test_roundtrip_preserves_structure() -> None:
    grid = progressive_align([("x", "y", "z"), ("x", "w")])
    net = build_confusion_network(grid)
    payload = confusion_to_json(net)
    parsed = confusion_from_json(payload)
    assert parsed.to_dict() == net.to_dict()

"""Tests for runmeta serialization helpers."""

from __future__ import annotations

from hypofuse.runmeta import capture, from_dict, from_json, to_dict, to_json


def test_to_dict_and_from_dict_roundtrip() -> None:
    meta = capture(seed=99, label="roundtrip")
    data = to_dict(meta)
    restored = from_dict(data)
    assert restored == meta


def test_to_json_and_from_json_roundtrip() -> None:
    meta = capture(seed=7, label="json")
    text = to_json(meta)
    restored = from_json(text)
    assert restored == meta


def test_to_json_is_stable() -> None:
    meta = capture(seed=7, label="json")
    text1 = to_json(meta)
    text2 = to_json(meta)
    assert text1 == text2


def test_to_dict_contains_required_keys() -> None:
    meta = capture(seed=1, label="keys")
    data = to_dict(meta)
    required_keys = {
        "config_hash",
        "created_from",
        "hypofuse_version",
        "numpy_version",
        "platform",
        "python_version",
        "seed",
        "torch_version",
    }
    assert required_keys.issubset(set(data))

"""Pair normalization and empty-reference policy."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import HypofuseError
from hypofuse.normalize import NormalizationConfig, normalize, normalize_pair


def test_pair_returns_two_normalized_strings() -> None:
    ref, hyp = normalize_pair("Hello, World!", "hello world")
    assert ref == "hello world"
    assert hyp == "hello world"


def test_empty_reference_default_policy_returns_pair() -> None:
    ref, hyp = normalize_pair("", "hi")
    assert ref == ""
    assert hyp == "hi"


def test_empty_reference_error_policy_raises() -> None:
    cfg = NormalizationConfig(empty_reference_policy="error")
    with pytest.raises(HypofuseError):
        normalize_pair("", "hi", cfg)


def test_empty_reference_pass_policy_returns_pair() -> None:
    cfg = NormalizationConfig(empty_reference_policy="pass")
    ref, hyp = normalize_pair("", "hi", cfg)
    assert ref == ""
    assert hyp == "hi"


def test_normalization_is_deterministic() -> None:
    text = "The QUICK brown  FOX 123."
    a = normalize(text)
    b = normalize(text)
    c = normalize(text)
    assert a == b == c

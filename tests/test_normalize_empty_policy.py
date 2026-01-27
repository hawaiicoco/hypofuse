"""Empty-reference policy completeness: all 6 combinations + sanity."""

from __future__ import annotations

import pytest

from hypofuse.exceptions import HypofuseError
from hypofuse.normalize import NormalizationConfig, normalize_pair


def test_skip_empty_reference() -> None:
    cfg = NormalizationConfig(empty_reference_policy="skip")
    ref, hyp = normalize_pair("", "hi", cfg)
    assert ref == ""
    assert hyp == "hi"


def test_skip_whitespace_reference() -> None:
    cfg = NormalizationConfig(empty_reference_policy="skip")
    ref, hyp = normalize_pair("   ", "hi", cfg)
    assert ref == ""
    assert hyp == "hi"


def test_error_empty_reference_raises() -> None:
    cfg = NormalizationConfig(empty_reference_policy="error")
    with pytest.raises(HypofuseError):
        normalize_pair("", "hi", cfg)


def test_error_whitespace_reference_raises() -> None:
    cfg = NormalizationConfig(empty_reference_policy="error")
    with pytest.raises(HypofuseError):
        normalize_pair("   ", "hi", cfg)


def test_pass_empty_reference() -> None:
    cfg = NormalizationConfig(empty_reference_policy="pass")
    ref, hyp = normalize_pair("", "hi", cfg)
    assert ref == ""
    assert hyp == "hi"


def test_pass_whitespace_reference() -> None:
    cfg = NormalizationConfig(empty_reference_policy="pass")
    ref, hyp = normalize_pair("   ", "hi", cfg)
    assert ref == ""
    assert hyp == "hi"


def test_sanity_nonempty_reference_error_policy() -> None:
    cfg = NormalizationConfig(empty_reference_policy="error")
    ref, hyp = normalize_pair("Hello!", "hello", cfg)
    assert ref == "hello"
    assert hyp == "hello"


def test_invalid_policy_raises() -> None:
    with pytest.raises(ValueError, match=r"empty_reference_policy"):
        NormalizationConfig(empty_reference_policy="warn")

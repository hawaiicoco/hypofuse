"""Tests for the UnicodeForm type alias and runtime validation."""

from __future__ import annotations

import pytest

from hypofuse.normalize import NormalizationConfig, UnicodeForm, normalize


def test_valid_unicode_form_nfc() -> None:
    """NFC is accepted and composes characters."""
    cfg = NormalizationConfig(unicode_form="NFC")
    assert cfg.unicode_form == "NFC"
    # e + combining acute -> U+00E9 under NFC
    result = normalize("cafe\u0301", cfg)
    assert "\u00e9" in result


def test_valid_unicode_form_nfd() -> None:
    """NFD is accepted and decomposes characters."""
    cfg = NormalizationConfig(unicode_form="NFD")
    assert cfg.unicode_form == "NFD"
    # U+00E9 decomposes to e + combining acute under NFD
    result = normalize("\u00e9", cfg)
    assert len(result) == 2


def test_valid_unicode_form_nfkc() -> None:
    """NFKC is accepted."""
    cfg = NormalizationConfig(unicode_form="NFKC")
    assert cfg.unicode_form == "NFKC"


def test_valid_unicode_form_nfkd() -> None:
    """NFKD is accepted."""
    cfg = NormalizationConfig(unicode_form="NFKD")
    assert cfg.unicode_form == "NFKD"


def test_invalid_unicode_form_raises() -> None:
    """An invalid unicode_form string raises ValueError at runtime."""
    with pytest.raises(ValueError, match="unicode_form"):
        NormalizationConfig(unicode_form="INVALID")  # type: ignore[arg-type]


def test_unicode_form_alias_exported() -> None:
    """The UnicodeForm alias is importable from the normalize module."""
    assert UnicodeForm is not None

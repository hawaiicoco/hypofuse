"""ITN integration with normalize pipeline."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, normalize


def test_itn_disabled_by_default() -> None:
    # ITN is off by default, so number words are not converted
    assert normalize("twenty one") == "twenty one"


def test_itn_enabled_converts_english() -> None:
    cfg = NormalizationConfig(itn=True)
    assert normalize("twenty one", cfg) == "21"


def test_itn_enabled_converts_chinese() -> None:
    # 十五 -> 15
    cfg = NormalizationConfig(itn=True, language_hint="zh")
    assert normalize("十五", cfg) == "15"


def test_itn_with_other_normalization() -> None:
    # Case folding happens before ITN
    cfg = NormalizationConfig(itn=True, case_fold=True)
    assert normalize("Twenty One", cfg) == "21"


def test_itn_preserves_non_number_text() -> None:
    cfg = NormalizationConfig(itn=True)
    assert normalize("hello world", cfg) == "hello world"


def test_itn_empty_string() -> None:
    cfg = NormalizationConfig(itn=True)
    assert normalize("", cfg) == ""

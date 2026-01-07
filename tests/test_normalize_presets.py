"""Preset recipes and unknown-name error tests."""

from __future__ import annotations

import pytest

from hypofuse.normalize import (
    PRESETS,
    NormalizationConfig,
    normalize,
    preset,
)


def test_presets_dict_has_four_entries() -> None:
    assert len(PRESETS) == 4


def test_score_en_exact_fields() -> None:
    cfg = preset("score_en")
    expected = NormalizationConfig(tokenization="word")
    assert cfg == expected


def test_score_zh_exact_fields() -> None:
    cfg = preset("score_zh")
    expected = NormalizationConfig(tokenization="char")
    assert cfg == expected


def test_raw_exact_fields() -> None:
    cfg = preset("raw")
    expected = NormalizationConfig(
        case_fold=False,
        case_mode="none",
        strip_punctuation=False,
        collapse_whitespace=False,
        fullwidth_to_halfwidth=False,
        remove_control=False,
    )
    assert cfg == expected


def test_display_exact_fields() -> None:
    cfg = preset("display")
    expected = NormalizationConfig(
        strip_punctuation=False,
        case_fold=False,
        case_mode="none",
    )
    assert cfg == expected


def test_unknown_preset_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"unknown preset"):
        preset("banana")


def test_unknown_preset_lists_valid_names() -> None:
    with pytest.raises(ValueError, match=r"display"):
        preset("nope")


def test_raw_preset_is_identity_on_mixed_string() -> None:
    text = "Hello, World! 123 \u4e00\u4e01"
    assert normalize(text, preset("raw")) == text

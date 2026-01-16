"""Digit normalization: spoken, hash, keep, digits modes."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, normalize


def test_spoken_english_simple() -> None:
    cfg = NormalizationConfig(digits_to="spoken")
    assert normalize("42", cfg) == "forty two"


def test_spoken_english_in_sentence() -> None:
    cfg = NormalizationConfig(digits_to="spoken")
    result = normalize("test 42 ok", cfg)
    assert result == "test forty two ok"


def test_spoken_chinese_via_language_hint() -> None:
    cfg = NormalizationConfig(digits_to="spoken", language_hint="zh")
    result = normalize("42", cfg)
    assert result == "\u56db\u5341\u4e8c"


def test_spoken_decimal_with_punct_off() -> None:
    cfg = NormalizationConfig(digits_to="spoken", strip_punctuation=False)
    result = normalize("3.14", cfg)
    assert result == "three point one four"


def test_spoken_decimal_punct_on_strips_dot() -> None:
    cfg = NormalizationConfig(digits_to="spoken")
    result = normalize("3.14", cfg)
    assert result == "three hundred fourteen"


def test_hash_unchanged() -> None:
    cfg = NormalizationConfig(digits_to="hash")
    assert normalize("abc 123 def", cfg) == "abc ### def"


def test_keep_unchanged() -> None:
    cfg = NormalizationConfig(digits_to="keep")
    assert normalize("abc 123 def", cfg) == "abc 123 def"


def test_digits_mode_converts_words() -> None:
    cfg = NormalizationConfig(digits_to="digits")
    assert normalize("twenty one", cfg) == "21"


def test_digits_mode_chinese() -> None:
    cfg = NormalizationConfig(digits_to="digits", language_hint="zh")
    assert normalize("\u5341\u4e94", cfg) == "15"


def test_roundtrip_spoken_then_digits() -> None:
    spoken_cfg = NormalizationConfig(digits_to="spoken")
    digits_cfg = NormalizationConfig(digits_to="digits")
    original = "42"
    spoken = normalize(original, spoken_cfg)
    restored = normalize(spoken, digits_cfg)
    assert restored == original


def test_roundtrip_chinese() -> None:
    spoken_cfg = NormalizationConfig(
        digits_to="spoken",
        language_hint="zh",
    )
    digits_cfg = NormalizationConfig(
        digits_to="digits",
        language_hint="zh",
    )
    original = "15"
    spoken = normalize(original, spoken_cfg)
    restored = normalize(spoken, digits_cfg)
    assert restored == original


def test_invalid_digits_to_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match=r"digits_to"):
        NormalizationConfig(digits_to="roman")


def test_spoken_zero() -> None:
    cfg = NormalizationConfig(digits_to="spoken")
    assert normalize("0", cfg) == "zero"

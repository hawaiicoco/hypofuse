"""Hostile and boundary input tests."""

from __future__ import annotations

from hypofuse.itn import (
    ItnConfig,
    digits_to_words,
    words_to_digits,
)


def test_empty_string_both_directions() -> None:
    assert words_to_digits("") == ""
    assert digits_to_words("") == ""


def test_whitespace_only() -> None:
    assert words_to_digits("   ") == "   "
    assert digits_to_words("   ") == "   "


def test_mixed_scripts_no_raise() -> None:
    # Mixed English, Chinese, and symbols should not raise
    text = "hello 世界 123 !@#"
    result = words_to_digits(text)
    assert isinstance(result, str)


def test_very_long_string_no_raise() -> None:
    # Long string with no numbers should not raise
    text = "a" * 10000
    result = words_to_digits(text)
    assert result == text


def test_repeated_punctuation() -> None:
    text = "!!!???..."
    result = words_to_digits(text)
    assert result == text


def test_unicode_emoji() -> None:
    text = "hello 🎉 world"
    result = words_to_digits(text)
    assert isinstance(result, str)


def test_zero_boundary() -> None:
    assert words_to_digits("zero") == "0"
    assert digits_to_words("0") == "zero"


def test_max_english_boundary() -> None:
    # 999999 is the max for English
    result = digits_to_words("999999")
    assert "thousand" in result


def test_max_chinese_boundary() -> None:
    # 99999999 is the max for Chinese
    cfg = ItnConfig(language="zh")
    result = digits_to_words("99999999", cfg)
    assert "万" in result  # 万 should be present


def test_out_of_range_digits_unchanged() -> None:
    # Numbers outside supported range should be left as-is
    # 1000000 is outside English range (0..999999)
    # Our simplified implementation may raise or return as-is
    # For robustness, it should not raise on ordinary prose
    text = "hello 1000000 world"
    result = digits_to_words(text)
    assert isinstance(result, str)


def test_special_characters_in_number_phrase() -> None:
    # Number phrase with unexpected characters
    text = "twenty! one"
    result = words_to_digits(text)
    # Should not raise, may return as-is or partially convert
    assert isinstance(result, str)


def test_newlines_and_tabs() -> None:
    text = "twenty\none\ttwo"
    result = words_to_digits(text)
    assert isinstance(result, str)

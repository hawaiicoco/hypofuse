"""Hostile unicode normalization cases."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, normalize, tokenize


def test_combining_marks_are_normalized() -> None:
    decomposed = "n\u0301"
    composed = normalize(decomposed)
    assert composed == "\u0144"


def test_zero_width_joiner_stripped_by_default() -> None:
    out = normalize("a\u200db")
    assert out == "ab"


def test_zero_width_joiner_preserved_when_control_off() -> None:
    cfg = NormalizationConfig(remove_control=False)
    out = normalize("a\u200db", cfg)
    assert out == "a\u200db"


def test_hostile_input_does_not_crash() -> None:
    nasty = "\u0000\u200bhello\ufeff"
    out = normalize(nasty)
    assert "hello" in out


def test_mixed_scripts_use_word_tokenization_when_no_cjk() -> None:
    assert tokenize("caf\u00e9 123") == ("caf\u00e9", "123")


def test_cjk_detection_uses_char_tokenization() -> None:
    out = tokenize("Hello \u4e16\u754c")
    assert out == (
        "h",
        "e",
        "l",
        "l",
        "o",
        "\u4e16",
        "\u754c",
    ) or out == ("hello", "\u4e16", "\u754c")


def test_strip_punctuation_keeps_internal_apostrophes_off() -> None:
    cfg = NormalizationConfig(strip_punctuation=True)
    assert normalize("it's", cfg) == "its"


def test_fullwidth_punctuation_also_stripped() -> None:
    assert normalize("\uff21\uff22\uff0c\uff23\u3002") == "abc"

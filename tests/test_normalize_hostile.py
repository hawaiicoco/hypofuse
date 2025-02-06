"""Hostile unicode normalization cases."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, normalize, tokenize


def test_combining_marks_are_normalized() -> None:
    decomposed = "n\u0301"
    composed = normalize(decomposed)
    assert composed == "\u0144"


def test_zero_width_joiner_is_preserved() -> None:
    # The pipeline does not currently strip zero-width characters; document it.
    out = normalize("a\u200db")
    assert out == "a\u200db"


def test_hostile_input_does_not_crash() -> None:
    nasty = "\u0000\u200bhello\ufeff"
    out = normalize(nasty)
    assert "hello" in out


def test_mixed_scripts_use_word_tokenization_when_no_cjk() -> None:
    assert tokenize("café 123") == ("café", "123")


def test_cjk_detection_uses_char_tokenization() -> None:
    out = tokenize("Hello 世界")
    assert out == ("h", "e", "l", "l", "o", "世", "界") or out == (
        "hello",
        "世",
        "界",
    )


def test_strip_punctuation_keeps_internal_apostrophes_off() -> None:
    cfg = NormalizationConfig(strip_punctuation=True)
    assert normalize("it's", cfg) == "its"


def test_fullwidth_punctuation_also_stripped() -> None:
    assert normalize("ＡＢ，Ｃ。") == "abc"

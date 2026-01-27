"""CJK whitespace treatment under char and word tokenization."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, tokenize


def test_char_mode_drops_spaces_between_cjk() -> None:
    cfg = NormalizationConfig(tokenization="char")
    result = tokenize(
        "\u4f60 \u597d \u4e16 \u754c",
        cfg,
    )
    assert result == (
        "\u4f60",
        "\u597d",
        "\u4e16",
        "\u754c",
    )


def test_word_mode_splits_cjk_on_spaces() -> None:
    cfg = NormalizationConfig(tokenization="word")
    result = tokenize(
        "\u4f60\u597d \u4e16\u754c",
        cfg,
    )
    assert result == (
        "\u4f60\u597d",
        "\u4e16\u754c",
    )


def test_mixed_text_char_mode() -> None:
    cfg = NormalizationConfig(tokenization="char")
    result = tokenize(
        "\u4e2d\u6587 mixed text",
        cfg,
    )
    assert result == (
        "\u4e2d",
        "\u6587",
        "m",
        "i",
        "x",
        "e",
        "d",
        "t",
        "e",
        "x",
        "t",
    )


def test_mixed_text_word_mode() -> None:
    cfg = NormalizationConfig(tokenization="word")
    result = tokenize(
        "\u4e2d\u6587 mixed text",
        cfg,
    )
    assert result == ("\u4e2d\u6587", "mixed", "text")


def test_cjk_no_spaces_char_mode() -> None:
    cfg = NormalizationConfig(tokenization="char")
    result = tokenize("\u4f60\u597d\u4e16\u754c", cfg)
    assert result == (
        "\u4f60",
        "\u597d",
        "\u4e16",
        "\u754c",
    )


def test_auto_picks_char_for_cjk() -> None:
    result = tokenize("\u4f60\u597d")
    assert result == ("\u4f60", "\u597d")


def test_auto_picks_word_for_english() -> None:
    result = tokenize("hello world")
    assert result == ("hello", "world")

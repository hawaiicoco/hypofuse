"""char_bigram tokenization tests for CJK and non-CJK text."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, tokenize


def test_bigram_four_cjk_chars() -> None:
    cfg = NormalizationConfig(tokenization="char_bigram")
    result = tokenize(
        "\u4f60\u597d\u4e16\u754c",
        cfg,
    )
    assert result == (
        "\u4f60\u597d",
        "\u597d\u4e16",
        "\u4e16\u754c",
    )


def test_bigram_single_cjk_char() -> None:
    cfg = NormalizationConfig(tokenization="char_bigram")
    result = tokenize("\u4f60", cfg)
    assert result == ("\u4f60",)


def test_bigram_two_cjk_chars() -> None:
    cfg = NormalizationConfig(tokenization="char_bigram")
    result = tokenize("\u4f60\u597d", cfg)
    assert result == ("\u4f60\u597d",)


def test_bigram_non_cjk_falls_back_to_word() -> None:
    cfg = NormalizationConfig(tokenization="char_bigram")
    result = tokenize("hello world", cfg)
    assert result == ("hello", "world")


def test_bigram_empty_returns_empty() -> None:
    cfg = NormalizationConfig(tokenization="char_bigram")
    result = tokenize("", cfg)
    assert result == ()


def test_bigram_invalid_tokenization_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match=r"tokenization"):
        NormalizationConfig(tokenization="trigram")


def test_bigram_cjk_with_spaces() -> None:
    cfg = NormalizationConfig(tokenization="char_bigram")
    result = tokenize(
        "\u4f60 \u597d \u4e16 \u754c",
        cfg,
    )
    assert result == (
        "\u4f60\u597d",
        "\u597d\u4e16",
        "\u4e16\u754c",
    )

"""Golden table test: frozen (input, config, text, tokens) rows."""

from __future__ import annotations

import pytest

from hypofuse.normalize import NormalizationConfig, normalize, tokenize


def _check_row(
    text: str,
    config: NormalizationConfig,
    expected_text: str,
    expected_tokens: tuple[str, ...],
) -> None:
    result_text = normalize(text, config)
    result_tokens = tokenize(text, config)
    assert result_text == expected_text
    assert result_tokens == expected_tokens


@pytest.mark.parametrize(
    ("text", "config", "expected_text", "expected_tokens"),
    [
        # English with default config
        (
            "Hello, World!",
            NormalizationConfig(),
            "hello world",
            ("hello", "world"),
        ),
        # Chinese with default config
        (
            "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
            NormalizationConfig(),
            "\u4f60\u597d\u4e16\u754c",
            ("\u4f60", "\u597d", "\u4e16", "\u754c"),
        ),
        # Digits hash mode
        (
            "abc 123 def",
            NormalizationConfig(digits_to="hash"),
            "abc ### def",
            ("abc", "###", "def"),
        ),
        # Digits keep mode
        (
            "test 42 ok",
            NormalizationConfig(digits_to="keep"),
            "test 42 ok",
            ("test", "42", "ok"),
        ),
        # Keep apostrophe
        (
            "it's a test",
            NormalizationConfig(keep_punctuation=("'",)),
            "it's a test",
            ("it's", "a", "test"),
        ),
        # Spoken digit mode English
        (
            "42",
            NormalizationConfig(digits_to="spoken"),
            "forty two",
            ("forty", "two"),
        ),
    ],
    ids=[
        "en_default",
        "zh_default",
        "hash_digits",
        "keep_digits",
        "keep_apostrophe",
        "spoken_en",
    ],
)
def test_golden_table(
    text: str,
    config: NormalizationConfig,
    expected_text: str,
    expected_tokens: tuple[str, ...],
) -> None:
    _check_row(text, config, expected_text, expected_tokens)

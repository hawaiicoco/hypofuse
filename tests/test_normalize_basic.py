"""Basic normalization tests."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, normalize, tokenize


def test_empty_string_returns_empty() -> None:
    assert normalize("") == ""


def test_whitespace_string_returns_empty() -> None:
    assert normalize("   \t  ") == ""


def test_fullwidth_digits_unified() -> None:
    assert normalize("\uff11\uff12\uff13") == "123"


def test_fullwidth_letters_unified() -> None:
    assert normalize("\uff21\uff22\uff23") == "abc"


def test_punctuation_stripped() -> None:
    assert normalize("hello, world!") == "hello world"


def test_case_folding_lowercases() -> None:
    assert normalize("Hello WORLD") == "hello world"


def test_digits_kept_by_default() -> None:
    assert normalize("abc 123 def") == "abc 123 def"


def test_digits_spoken() -> None:
    cfg = NormalizationConfig(digits_to="spoken")
    assert normalize("abc 123 def", cfg) == "abc 000 def"


def test_digits_hash() -> None:
    cfg = NormalizationConfig(digits_to="hash")
    assert normalize("abc 123 def", cfg) == "abc ### def"


def test_whitespace_collapsed() -> None:
    assert normalize("hello\n\tworld  ") == "hello world"


def test_chinese_characters_preserved() -> None:
    out = normalize("你好\uff0c世界\uff01")
    assert out.startswith("你好世界")


def test_tokenize_auto_uses_words_for_english() -> None:
    assert tokenize("Hello World") == ("hello", "world")


def test_tokenize_auto_uses_chars_for_chinese() -> None:
    assert tokenize("你好世界") == ("你", "好", "世", "界")


def test_tokenize_word_mode_splits_chinese_per_char() -> None:
    cfg = NormalizationConfig(tokenization="char")
    assert tokenize("hi 你好", cfg) == ("h", "i", "你", "好")


def test_config_with_overrides_returns_new_instance() -> None:
    cfg = NormalizationConfig()
    assert cfg.case_fold is True
    cfg2 = cfg.with_overrides(case_fold=False)
    assert cfg2.case_fold is False
    assert cfg.case_fold is True

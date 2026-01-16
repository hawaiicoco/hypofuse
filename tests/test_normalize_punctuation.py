"""keep_punctuation exemption character tests."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, normalize


def test_default_strips_all_punctuation() -> None:
    assert normalize("hello, world!") == "hello world"


def test_apostrophe_kept_when_configured() -> None:
    cfg = NormalizationConfig(keep_punctuation=("'",))
    assert normalize("it's a test", cfg) == "it's a test"


def test_hyphen_kept_when_configured() -> None:
    cfg = NormalizationConfig(keep_punctuation=("-",))
    assert normalize("well-known fact", cfg) == "well-known fact"


def test_multiple_kept_characters() -> None:
    cfg = NormalizationConfig(keep_punctuation=("'", "-"))
    assert normalize("it's well-known!", cfg) == "it's well-known"


def test_kept_punctuation_with_adjacent_spaces() -> None:
    cfg = NormalizationConfig(keep_punctuation=("-",))
    result = normalize("a - b", cfg)
    assert result == "a - b"


def test_not_kept_punctuation_still_stripped() -> None:
    cfg = NormalizationConfig(keep_punctuation=("-",))
    result = normalize("hello, world!", cfg)
    assert result == "hello world"


def test_empty_keep_punctuation_default_unchanged() -> None:
    cfg = NormalizationConfig(keep_punctuation=())
    assert normalize("it's", cfg) == "its"

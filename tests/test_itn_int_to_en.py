"""Integer to English words inverse conversion."""

from __future__ import annotations

import pytest

from hypofuse.itn import int_to_en_words


def test_zero() -> None:
    assert int_to_en_words(0) == "zero"


def test_single_digit() -> None:
    assert int_to_en_words(5) == "five"


def test_teen() -> None:
    assert int_to_en_words(13) == "thirteen"


def test_twenty_one() -> None:
    assert int_to_en_words(21) == "twenty one"


def test_one_hundred() -> None:
    assert int_to_en_words(100) == "one hundred"


def test_one_hundred_five() -> None:
    assert int_to_en_words(105) == "one hundred five"


def test_one_thousand() -> None:
    assert int_to_en_words(1000) == "one thousand"


def test_max_supported() -> None:
    assert int_to_en_words(999999) == ("nine hundred ninety nine thousand nine hundred ninety nine")


def test_negative_raises() -> None:
    with pytest.raises(ValueError, match="out of supported range"):
        int_to_en_words(-1)


def test_too_large_raises() -> None:
    with pytest.raises(ValueError, match="out of supported range"):
        int_to_en_words(1_000_000)


def test_bool_raises() -> None:
    with pytest.raises(TypeError):
        int_to_en_words(True)  # type: ignore[arg-type]

"""Chinese cardinal number words to integer, zero through ninety-nine."""

from __future__ import annotations

import pytest

from hypofuse.itn import zh_words_to_int


def test_zero() -> None:
    assert zh_words_to_int("零") == 0


def test_single_digit() -> None:
    assert zh_words_to_int("五") == 5  # 五 = 5


def test_ten() -> None:
    assert zh_words_to_int("十") == 10  # 十 = 10


def test_fifteen() -> None:
    # 10 + 5 = 15
    assert zh_words_to_int("十五") == 15  # 十五


def test_twenty() -> None:
    # 2 * 10 = 20
    assert zh_words_to_int("二十") == 20  # 二十


def test_ninety_nine() -> None:
    # 9 * 10 + 9 = 99
    assert zh_words_to_int("九十九") == 99  # 九十九


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        zh_words_to_int("")


def test_unknown_char_raises() -> None:
    with pytest.raises(ValueError, match="unknown Chinese"):
        zh_words_to_int("abc")

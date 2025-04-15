"""English cardinal number words to integer, zero through ninety-nine."""

from __future__ import annotations

import pytest

from hypofuse.itn import en_words_to_int


def test_zero() -> None:
    assert en_words_to_int("zero") == 0


def test_single_digit() -> None:
    assert en_words_to_int("five") == 5


def test_teen() -> None:
    # 10 + 3 = 13
    assert en_words_to_int("thirteen") == 13


def test_nineteen() -> None:
    assert en_words_to_int("nineteen") == 19


def test_round_ten() -> None:
    assert en_words_to_int("twenty") == 20


def test_two_word_number() -> None:
    # 20 + 1 = 21
    assert en_words_to_int("twenty one") == 21


def test_hyphenated() -> None:
    # 40 + 2 = 42
    assert en_words_to_int("forty-two") == 42


def test_ninety_nine() -> None:
    # 90 + 9 = 99
    assert en_words_to_int("ninety nine") == 99


def test_case_insensitive() -> None:
    assert en_words_to_int("Twenty One") == 21


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        en_words_to_int("")


def test_unknown_word_raises() -> None:
    with pytest.raises(ValueError, match="unknown number word"):
        en_words_to_int("banana")

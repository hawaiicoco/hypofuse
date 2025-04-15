"""English number words with hundreds, thousands and millions."""

from __future__ import annotations

import pytest

from hypofuse.itn import en_words_to_int


def test_one_hundred() -> None:
    assert en_words_to_int("one hundred") == 100


def test_hundred_and_five() -> None:
    # 1 * 100 + 5 = 105
    assert en_words_to_int("one hundred and five") == 105


def test_three_hundred_twenty_five() -> None:
    # 3 * 100 + 20 + 5 = 325
    assert en_words_to_int("three hundred twenty-five") == 325


def test_one_thousand() -> None:
    assert en_words_to_int("one thousand") == 1000


def test_one_thousand_one() -> None:
    # 1000 + 1 = 1001
    assert en_words_to_int("one thousand one") == 1001


def test_large_number() -> None:
    # 325 * 1000 + 100 = 325100
    assert en_words_to_int("three hundred twenty five thousand one hundred") == 325100


def test_max_supported() -> None:
    # 999 * 1000 + 999 = 999999
    assert en_words_to_int("nine hundred ninety nine thousand nine hundred ninety nine") == 999999


def test_million() -> None:
    assert en_words_to_int("one million") == 1_000_000


def test_hundred_alone_raises() -> None:
    with pytest.raises(ValueError, match="no numeric value"):
        en_words_to_int("hundred")

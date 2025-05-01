"""Decimal and negative number phrase conversions."""

from __future__ import annotations

from hypofuse.itn import en_phrase_to_digits, zh_phrase_to_digits


def test_en_decimal() -> None:
    # 3.14
    assert en_phrase_to_digits("three point one four") == "3.14"


def test_en_decimal_zero() -> None:
    assert en_phrase_to_digits("zero point five") == "0.5"


def test_en_minus() -> None:
    assert en_phrase_to_digits("minus five") == "-5"


def test_en_negative() -> None:
    assert en_phrase_to_digits("negative five") == "-5"


def test_en_negative_decimal() -> None:
    assert en_phrase_to_digits("negative three point one four") == "-3.14"


def test_en_plain_integer() -> None:
    assert en_phrase_to_digits("twenty one") == "21"


def test_zh_decimal() -> None:
    # 三点一四 -> 3.14
    assert zh_phrase_to_digits("三点一四") == "3.14"


def test_zh_decimal_zero() -> None:
    # 零点五 -> 0.5
    assert zh_phrase_to_digits("零点五") == "0.5"


def test_zh_negative() -> None:
    # 负五 -> -5
    assert zh_phrase_to_digits("负五") == "-5"


def test_zh_negative_decimal() -> None:
    # 负三点一四 -> -3.14
    assert zh_phrase_to_digits("负三点一四") == "-3.14"


def test_zh_plain_integer() -> None:
    # 十五 -> 15
    assert zh_phrase_to_digits("十五") == "15"

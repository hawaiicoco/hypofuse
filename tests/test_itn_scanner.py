"""words_to_digits and digits_to_words scanner tests."""

from __future__ import annotations

from hypofuse.itn import ItnConfig, digits_to_words, words_to_digits


def test_words_to_digits_english_simple() -> None:
    assert words_to_digits("twenty one") == "21"


def test_words_to_digits_english_decimal() -> None:
    assert words_to_digits("three point one four") == "3.14"


def test_words_to_digits_english_negative() -> None:
    assert words_to_digits("minus five") == "-5"


def test_words_to_digits_chinese_simple() -> None:
    # 十五 -> 15
    cfg = ItnConfig(language="zh")
    assert words_to_digits("十五", cfg) == "15"


def test_words_to_digits_chinese_decimal() -> None:
    # 三点一四 -> 3.14
    cfg = ItnConfig(language="zh")
    assert words_to_digits("三点一四", cfg) == "3.14"


def test_words_to_digits_empty() -> None:
    assert words_to_digits("") == ""


def test_words_to_digits_no_numbers() -> None:
    assert words_to_digits("hello world") == "hello world"


def test_words_to_digits_idempotent() -> None:
    text = "twenty one"
    once = words_to_digits(text)
    twice = words_to_digits(once)
    assert once == twice


def test_digits_to_words_english_simple() -> None:
    assert digits_to_words("21") == "twenty one"


def test_digits_to_words_english_decimal() -> None:
    assert digits_to_words("3.14") == "three point one four"


def test_digits_to_words_english_negative() -> None:
    assert digits_to_words("-5") == "negative five"


def test_digits_to_words_chinese_simple() -> None:
    # 15 -> 十五
    cfg = ItnConfig(language="zh")
    assert digits_to_words("15", cfg) == "十五"


def test_digits_to_words_chinese_decimal() -> None:
    # 3.14 -> 三点一四
    cfg = ItnConfig(language="zh")
    assert digits_to_words("3.14", cfg) == "三点一四"


def test_digits_to_words_empty() -> None:
    assert digits_to_words("") == ""


def test_digits_to_words_no_digits() -> None:
    assert digits_to_words("hello world") == "hello world"

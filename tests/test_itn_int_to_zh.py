"""Integer to Chinese characters inverse conversion."""

from __future__ import annotations

import pytest

from hypofuse.itn import int_to_zh_words


def test_zero() -> None:
    assert int_to_zh_words(0) == "零"  # 零


def test_single_digit() -> None:
    assert int_to_zh_words(5) == "五"  # 五


def test_fifteen() -> None:
    assert int_to_zh_words(15) == "十五"  # 十五


def test_one_hundred() -> None:
    assert int_to_zh_words(100) == "一百"  # 一百


def test_three_hundred_twenty_five() -> None:
    assert int_to_zh_words(325) == "三百二十五"  # 三百二十五


def test_one_thousand_one() -> None:
    assert int_to_zh_words(1001) == "一千零一"  # 一千零一


def test_ten_thousand() -> None:
    assert int_to_zh_words(10000) == "一万"  # 一万


def test_ten_thousand_one() -> None:
    assert int_to_zh_words(10001) == "一万零一"  # 一万零一


def test_max_supported() -> None:
    expected = (
        "九千九百九十九"  # 九千九百九十九
        "万"  # 万
        "九千九百九十九"  # 九千九百九十九
    )
    assert int_to_zh_words(99_999_999) == expected


def test_negative_raises() -> None:
    with pytest.raises(ValueError, match="out of supported range"):
        int_to_zh_words(-1)


def test_too_large_raises() -> None:
    with pytest.raises(ValueError, match="out of supported range"):
        int_to_zh_words(100_000_000)

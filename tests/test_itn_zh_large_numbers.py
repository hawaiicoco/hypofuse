"""Chinese number words with hundreds, thousands and wan."""

from __future__ import annotations

from hypofuse.itn import zh_words_to_int


def test_one_hundred() -> None:
    # 1 * 100 = 100
    assert zh_words_to_int("一百") == 100  # 一百


def test_three_hundred_twenty_five() -> None:
    # 3*100 + 2*10 + 5 = 325
    assert zh_words_to_int("三百二十五") == 325  # 三百二十五


def test_one_thousand_one() -> None:
    # 1*1000 + 0 + 1 = 1001
    assert zh_words_to_int("一千零一") == 1001  # 一千零一


def test_ten_thousand() -> None:
    assert zh_words_to_int("一万") == 10000  # 一万


def test_wan_alone() -> None:
    # 万 alone = 10000
    assert zh_words_to_int("万") == 10000  # 万


def test_large_wan_number() -> None:
    # 1234 * 10000 + 5678 = 12345678
    text = (
        "一千二百三十四"  # 一千二百三十四
        "万"  # 万
        "五千六百七十八"  # 五千六百七十八
    )
    assert zh_words_to_int(text) == 12345678


def test_max_supported() -> None:
    # 9999 * 10000 + 9999 = 99999999
    text = (
        "九千九百九十九"  # 九千九百九十九
        "万"  # 万
        "九千九百九十九"  # 九千九百九十九
    )
    assert zh_words_to_int(text) == 99_999_999

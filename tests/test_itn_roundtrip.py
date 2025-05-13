"""Roundtrip stability tests."""

from __future__ import annotations

from hypofuse.itn import ItnConfig, itn_roundtrip_stable


def test_english_simple_roundtrip() -> None:
    assert itn_roundtrip_stable("twenty one")


def test_english_decimal_roundtrip() -> None:
    assert itn_roundtrip_stable("three point one four")


def test_english_negative_roundtrip() -> None:
    assert itn_roundtrip_stable("minus five")


def test_chinese_simple_roundtrip() -> None:
    cfg = ItnConfig(language="zh")
    assert itn_roundtrip_stable("十五", cfg)  # 十五


def test_chinese_decimal_roundtrip() -> None:
    cfg = ItnConfig(language="zh")
    assert itn_roundtrip_stable("三点一四", cfg)  # 三点一四


def test_empty_roundtrip() -> None:
    assert itn_roundtrip_stable("")


def test_no_numbers_roundtrip() -> None:
    assert itn_roundtrip_stable("hello world")


def test_already_digits_roundtrip() -> None:
    assert itn_roundtrip_stable("42")


def test_large_number_roundtrip() -> None:
    assert itn_roundtrip_stable("nine hundred ninety nine")

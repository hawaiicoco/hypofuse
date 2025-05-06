"""Percent and unit table conversions."""

from __future__ import annotations

from hypofuse.itn import en_lookup_unit, zh_lookup_unit


def test_en_kilograms() -> None:
    assert en_lookup_unit("kilograms") == "kg"


def test_en_kilogram_singular() -> None:
    assert en_lookup_unit("kilogram") == "kg"


def test_en_meters() -> None:
    assert en_lookup_unit("meters") == "m"


def test_en_metres_british() -> None:
    assert en_lookup_unit("metres") == "m"


def test_en_seconds() -> None:
    assert en_lookup_unit("seconds") == "s"


def test_en_hours() -> None:
    assert en_lookup_unit("hours") == "h"


def test_en_unknown_unit() -> None:
    assert en_lookup_unit("bananas") is None


def test_en_case_insensitive() -> None:
    assert en_lookup_unit("Kilograms") == "kg"


def test_zh_kilogram() -> None:
    # 公斤 -> kg
    assert zh_lookup_unit("公斤") == "kg"


def test_zh_meter() -> None:
    # 米 -> m
    assert zh_lookup_unit("米") == "m"


def test_zh_second() -> None:
    # 秒 -> s
    assert zh_lookup_unit("秒") == "s"


def test_zh_unknown_unit() -> None:
    assert zh_lookup_unit("一") is None  # 一 is not a unit

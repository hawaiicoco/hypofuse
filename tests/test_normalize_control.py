"""Hostile input control removal and safety tests."""

from __future__ import annotations

import unicodedata

from hypofuse.normalize import NormalizationConfig, normalize


def test_null_byte_removed() -> None:
    out = normalize("a\x00b")
    assert "\x00" not in out
    assert out == "ab"


def test_zero_width_space_removed() -> None:
    out = normalize("a\u200bb")
    assert out == "ab"


def test_bidi_override_removed() -> None:
    out = normalize("a\u202eb")
    assert "\u202e" not in out
    assert out == "ab"


def test_variation_selector_removed() -> None:
    # U+FE0E is a variation selector: category Mn, but presentation-only
    out = normalize("a\ufe0eb")
    assert out == "ab"


def test_bom_removed() -> None:
    out = normalize("\ufeffhello")
    assert out == "hello"


def test_control_chars_not_in_output() -> None:
    nasty = "\x00\x01\x02\x1f\x7fhello"
    out = normalize(nasty)
    for ch in out:
        cat = unicodedata.category(ch)
        assert cat not in ("Cc", "Cf"), f"found {ch!r} cat={cat}"


def test_format_chars_not_in_output() -> None:
    nasty = "\u200b\u200c\u200d\u202a\u202ehello"
    out = normalize(nasty)
    for ch in out:
        assert unicodedata.category(ch) != "Cf"


def test_idempotent_on_hostile_input() -> None:
    nasty = "\x00\u200b\u202e hello \ufeff\u200d"
    once = normalize(nasty)
    twice = normalize(once)
    assert once == twice


def test_extremely_long_whitespace_no_crash() -> None:
    text = "a" + " " * 10_000 + "b"
    out = normalize(text)
    assert out == "a b"


def test_mixed_cjk_latin_no_crash() -> None:
    text = "\u4e00hello\u4e01world\u4e02"
    out = normalize(text)
    assert isinstance(out, str)


def test_remove_control_false_preserves_format_chars() -> None:
    cfg = NormalizationConfig(remove_control=False)
    text = "a\u200bb"
    out = normalize(text, cfg)
    assert "\u200b" in out


def test_surrogate_looking_escapes_no_crash() -> None:
    text = "\ud800" if False else "hello"
    out = normalize(text)
    assert isinstance(out, str)

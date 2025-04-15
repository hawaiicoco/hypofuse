"""ItnConfig defaults and overrides."""

from __future__ import annotations

import pytest

from hypofuse.itn import ItnConfig


def test_default_language_is_english() -> None:
    assert ItnConfig().language == "en"


def test_all_flags_default_to_true() -> None:
    cfg = ItnConfig()
    assert cfg.numbers is True
    assert cfg.decimals is True
    assert cfg.percent is True
    assert cfg.units is True
    assert cfg.negative is True


def test_config_is_frozen() -> None:
    cfg = ItnConfig()
    with pytest.raises(AttributeError):
        cfg.language = "zh"  # type: ignore[misc]


def test_with_overrides_returns_new_instance() -> None:
    cfg = ItnConfig()
    cfg2 = cfg.with_overrides(language="zh")
    assert cfg2.language == "zh"
    assert cfg.language == "en"


def test_with_overrides_preserves_unset_fields() -> None:
    cfg = ItnConfig()
    cfg2 = cfg.with_overrides(numbers=False)
    assert cfg2.numbers is False
    assert cfg2.decimals is True
    assert cfg2.language == "en"

"""ItnConfig validation rules."""

from __future__ import annotations

import pytest

from hypofuse.itn import ItnConfig


def test_validate_default_passes() -> None:
    ItnConfig().validate()


def test_validate_chinese_passes() -> None:
    ItnConfig(language="zh").validate()


def test_validate_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unknown language"):
        ItnConfig(language="fr").validate()


def test_validate_rejects_all_flags_off() -> None:
    cfg = ItnConfig(numbers=False, decimals=False, percent=False, units=False, negative=False)
    with pytest.raises(ValueError, match="at least one"):
        cfg.validate()


def test_validate_accepts_one_flag_on() -> None:
    cfg = ItnConfig(numbers=True, decimals=False, percent=False, units=False, negative=False)
    cfg.validate()

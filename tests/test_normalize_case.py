"""Case mode precision tests: fold vs lower vs none."""

from __future__ import annotations

import pytest

from hypofuse.normalize import NormalizationConfig, normalize


def test_fold_converts_sharp_s() -> None:
    # German sharp s: casefold produces "ss", lower preserves it
    cfg = NormalizationConfig(case_fold=True, case_mode="fold")
    result = normalize("Stra\u00dfe", cfg)
    assert result == "strasse"


def test_lower_preserves_sharp_s() -> None:
    cfg = NormalizationConfig(case_fold=True, case_mode="lower")
    result = normalize("Stra\u00dfe", cfg)
    assert result == "stra\u00dfe"


def test_fold_lower_differ() -> None:
    fold = normalize(
        "Stra\u00dfe",
        NormalizationConfig(case_fold=True, case_mode="fold"),
    )
    low = normalize(
        "Stra\u00dfe",
        NormalizationConfig(case_fold=True, case_mode="lower"),
    )
    assert fold != low


def test_none_preserves_case() -> None:
    cfg = NormalizationConfig(case_fold=True, case_mode="none")
    result = normalize("Hello WORLD", cfg)
    assert result == "Hello WORLD"


def test_case_fold_false_skips_regardless_of_mode() -> None:
    cfg = NormalizationConfig(case_fold=False, case_mode="fold")
    result = normalize("Hello WORLD", cfg)
    assert result == "Hello WORLD"


def test_invalid_case_mode_raises() -> None:
    with pytest.raises(ValueError, match=r"case_mode"):
        NormalizationConfig(case_mode="upper")


def test_default_case_mode_is_fold() -> None:
    cfg = NormalizationConfig()
    assert cfg.case_mode == "fold"

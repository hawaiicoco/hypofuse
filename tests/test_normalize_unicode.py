"""Unicode normalization form selection tests."""

from __future__ import annotations

import unicodedata

import pytest

from hypofuse.normalize import NormalizationConfig, normalize


def test_nfc_combining_marks_compose() -> None:
    # "e" + combining acute (U+0301) -> U+00E9 under NFC
    decomposed = "e\u0301"
    result = normalize(decomposed, NormalizationConfig(unicode_form="NFC"))
    assert result == "\u00e9"
    assert len(result) == 1


def test_nfkc_folds_ligature() -> None:
    # U+FB01 (fi ligature) folds to "fi" under NFKC but not NFC
    ligature = "\ufb01"
    nfc_result = normalize(
        ligature,
        NormalizationConfig(
            unicode_form="NFC",
            strip_punctuation=False,
            case_fold=False,
            case_mode="none",
        ),
    )
    nfkc_result = normalize(
        ligature,
        NormalizationConfig(
            unicode_form="NFKC",
            strip_punctuation=False,
            case_fold=False,
            case_mode="none",
        ),
    )
    assert nfc_result == "\ufb01"
    assert nfkc_result == "fi"


def test_nfkc_folds_fullwidth_digits() -> None:
    # Full-width digit U+FF11 is folded to "1" by NFKC
    # (also folded by our fullwidth_to_halfwidth step under NFC)
    fw = "\uff11"
    nfkc_no_fw = normalize(
        fw,
        NormalizationConfig(
            unicode_form="NFKC",
            fullwidth_to_halfwidth=False,
            strip_punctuation=False,
            case_fold=False,
            case_mode="none",
        ),
    )
    assert nfkc_no_fw == "1"


def test_invalid_unicode_form_raises() -> None:
    with pytest.raises(ValueError, match=r"unicode_form"):
        NormalizationConfig(unicode_form="NFC_X")


def test_nfd_decomposes() -> None:
    # U+00E9 under NFD becomes "e" + combining acute
    composed = "\u00e9"
    result = normalize(
        composed,
        NormalizationConfig(
            unicode_form="NFD",
            strip_punctuation=False,
            case_fold=False,
            case_mode="none",
        ),
    )
    assert unicodedata.is_normalized("NFD", result)


def test_default_is_nfc() -> None:
    cfg = NormalizationConfig()
    assert cfg.unicode_form == "NFC"

"""normalization_report helper tests."""

from __future__ import annotations

from hypofuse.normalize import NormalizationConfig, normalization_report


def test_report_exact_keys() -> None:
    report = normalization_report("hello", "world")
    expected_keys = {
        "reference",
        "hypothesis",
        "reference_tokens",
        "hypothesis_tokens",
        "tokenization_mode",
        "config_hash",
    }
    assert set(report.keys()) == expected_keys


def test_report_values_basic() -> None:
    report = normalization_report("Hello, World!", "hello world")
    assert report["reference"] == "hello world"
    assert report["hypothesis"] == "hello world"
    assert report["reference_tokens"] == ("hello", "world")
    assert report["hypothesis_tokens"] == ("hello", "world")
    assert report["tokenization_mode"] == "word"


def test_report_hash_changes_with_config() -> None:
    r1 = normalization_report("hello", "world", NormalizationConfig())
    r2 = normalization_report(
        "hello",
        "world",
        NormalizationConfig(case_fold=False),
    )
    assert r1["config_hash"] != r2["config_hash"]


def test_report_hash_deterministic() -> None:
    r1 = normalization_report("hello", "world")
    r2 = normalization_report("hello", "world")
    assert r1["config_hash"] == r2["config_hash"]


def test_report_tokenization_mode_auto_resolves() -> None:
    report = normalization_report(
        "\u4f60\u597d",
        "\u4f60\u597d",
    )
    assert report["tokenization_mode"] == "char"


def test_report_tokenization_mode_explicit() -> None:
    cfg = NormalizationConfig(tokenization="word")
    report = normalization_report("hello", "hello", cfg)
    assert report["tokenization_mode"] == "word"


def test_report_config_hash_is_string() -> None:
    report = normalization_report("a", "b")
    assert isinstance(report["config_hash"], str)
    assert len(report["config_hash"]) == 64

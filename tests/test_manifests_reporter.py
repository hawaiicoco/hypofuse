"""Tests for ReportRecord."""

from __future__ import annotations

from hypofuse.manifests.reporter import ReportRecord


def test_report_record_defaults() -> None:
    r = ReportRecord(
        report_id="rep_1",
        schema="hypofuse.report",
        generated_at="2026-09-26T00:00:00Z",
        hypofuse_version="0.1.0",
        config_hash="abc",
        seed=42,
    )
    assert r.artifact_paths == ()
    assert r.metrics == ()


def test_report_record_metric_pairs() -> None:
    r = ReportRecord(
        report_id="rep_1",
        schema="hypofuse.report",
        generated_at="2026-09-26T00:00:00Z",
        hypofuse_version="0.1.0",
        config_hash="abc",
        seed=42,
        metrics=(("wer", 0.1), ("cer", 0.05)),
    )
    assert dict(r.metrics) == {"wer": 0.1, "cer": 0.05}


def test_report_record_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    import pytest

    r = ReportRecord(
        report_id="rep_1",
        schema="hypofuse.report",
        generated_at="2026-09-26T00:00:00Z",
        hypofuse_version="0.1.0",
        config_hash="abc",
        seed=42,
    )
    with pytest.raises(FrozenInstanceError):
        r.seed = 0  # type: ignore[misc]

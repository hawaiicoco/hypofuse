"""Tests for the SCHEMA_VERSIONS registry and to_row stamping."""

from __future__ import annotations

import hypofuse.manifests as m
from hypofuse.convert import (
    fusion_run_to_row,
    nbest_to_row,
    reference_to_row,
    report_to_row,
    system_to_row,
)
from hypofuse.manifests.fusion_run import FusionRun
from hypofuse.manifests.nbest import NBestList
from hypofuse.manifests.reference import ReferenceTranscript
from hypofuse.manifests.reporter import ReportRecord
from hypofuse.manifests.system import SystemMetadata


def test_schema_versions_covers_all_schemas() -> None:
    assert set(m.SCHEMA_VERSIONS.keys()) == m.KNOWN_SCHEMAS


def test_schema_versions_all_positive() -> None:
    for name, ver in m.SCHEMA_VERSIONS.items():
        assert isinstance(ver, int)
        assert ver >= 1, f"{name} has version {ver}"


def test_nbest_to_row_stamps_version() -> None:
    nb = NBestList(utterance_id="u1", system="a", language="en")
    row = nbest_to_row(nb)
    assert row["schema_version"] == m.SCHEMA_VERSIONS[m.SCHEMA_NBEST]


def test_reference_to_row_stamps_version() -> None:
    ref = ReferenceTranscript(utterance_id="u1", text="hi")
    row = reference_to_row(ref)
    assert row["schema_version"] == m.SCHEMA_VERSIONS[m.SCHEMA_REFERENCE]


def test_system_to_row_stamps_version() -> None:
    sys_md = SystemMetadata(system_id="s1", language="en")
    row = system_to_row(sys_md)
    assert row["schema_version"] == m.SCHEMA_VERSIONS[m.SCHEMA_SYSTEM]


def test_fusion_to_row_stamps_version() -> None:
    fr = FusionRun(
        utterance_id="u1",
        systems=("a",),
        tokens=("hi",),
        confidences=(0.9,),
        policy="majority",
    )
    row = fusion_run_to_row(fr)
    assert row["schema_version"] == m.SCHEMA_VERSIONS[m.SCHEMA_FUSION_RUN]


def test_report_to_row_stamps_version() -> None:
    rep = ReportRecord(
        report_id="r1",
        schema="hypofuse.report",
        generated_at="2026-01-01T00:00:00Z",
        hypofuse_version="0.1.0",
        config_hash="abc",
        seed=0,
    )
    row = report_to_row(rep)
    assert row["schema_version"] == m.SCHEMA_VERSIONS[m.SCHEMA_REPORT]


def test_validate_record_accepts_missing_version() -> None:
    """Missing schema_version is treated as version 1."""
    from hypofuse.manifests.validate import validate_record

    row = {
        "schema": "hypofuse.nbest",
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
    }
    assert validate_record(row) == "hypofuse.nbest"


def test_validate_record_accepts_known_version() -> None:
    from hypofuse.manifests.validate import validate_record

    row = {
        "schema": "hypofuse.nbest",
        "schema_version": 1,
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
    }
    assert validate_record(row) == "hypofuse.nbest"


def test_validate_record_rejects_unknown_version() -> None:
    import pytest

    from hypofuse.exceptions import SchemaError
    from hypofuse.manifests.validate import validate_record

    row = {
        "schema": "hypofuse.nbest",
        "schema_version": 999,
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
    }
    with pytest.raises(SchemaError, match="unknown schema_version"):
        validate_record(row)


def test_validate_record_rejects_bool_version() -> None:
    import pytest

    from hypofuse.exceptions import SchemaError
    from hypofuse.manifests.validate import validate_record

    row = {
        "schema": "hypofuse.nbest",
        "schema_version": True,
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
    }
    with pytest.raises(SchemaError, match="schema_version must be an integer"):
        validate_record(row)


def test_golden_stamped_nbest_row() -> None:
    """The exact stamped row for nbest is stable."""
    nb = NBestList(utterance_id="u1", system="a", language="en")
    row = nbest_to_row(nb)
    assert row["schema"] == "hypofuse.nbest"
    assert row["schema_version"] == m.SCHEMA_VERSIONS[m.SCHEMA_NBEST]
    assert isinstance(row["schema_version"], int)

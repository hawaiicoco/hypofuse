"""Verify the package surface."""

from __future__ import annotations

import hypofuse.manifests as m


def test_known_schemas_present() -> None:
    assert "hypofuse.nbest" in m.KNOWN_SCHEMAS
    assert "hypofuse.reference" in m.KNOWN_SCHEMAS
    assert "hypofuse.system" in m.KNOWN_SCHEMAS
    assert "hypofuse.fusion_run" in m.KNOWN_SCHEMAS
    assert "hypofuse.report" in m.KNOWN_SCHEMAS


def test_schema_version_is_int() -> None:
    assert isinstance(m.SCHEMA_VERSION, int)
    assert m.SCHEMA_VERSION >= 1


def test_dataclasses_importable() -> None:
    assert m.NBestList is not None
    assert m.ReferenceTranscript is not None
    assert m.SystemMetadata is not None
    assert m.FusionRun is not None
    assert m.FusionArc is not None
    assert m.ReportRecord is not None

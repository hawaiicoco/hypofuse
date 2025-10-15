"""Tests for the convert_row schema dispatcher."""

from __future__ import annotations

import pytest

from hypofuse.convert import convert_row
from hypofuse.exceptions import SchemaError
from hypofuse.manifests import KNOWN_SCHEMAS
from hypofuse.manifests.fusion_run import FusionRun
from hypofuse.manifests.nbest import NBestList
from hypofuse.manifests.reference import ReferenceTranscript
from hypofuse.manifests.reporter import ReportRecord
from hypofuse.manifests.system import SystemMetadata

_SAMPLES: dict[str, dict] = {
    "hypofuse.nbest": {
        "schema": "hypofuse.nbest",
        "utterance_id": "u1",
        "system": "a",
        "language": "en",
    },
    "hypofuse.reference": {
        "schema": "hypofuse.reference",
        "utterance_id": "u1",
        "text": "hi",
    },
    "hypofuse.system": {
        "schema": "hypofuse.system",
        "system_id": "s1",
        "language": "en",
    },
    "hypofuse.fusion_run": {
        "schema": "hypofuse.fusion_run",
        "utterance_id": "u1",
        "systems": ["a"],
        "tokens": ["hi"],
        "confidences": [0.9],
        "policy": "majority",
    },
    "hypofuse.report": {
        "schema": "hypofuse.report",
        "report_id": "r1",
        "generated_at": "2026-01-01T00:00:00Z",
        "hypofuse_version": "0.1.0",
        "config_hash": "abc",
        "seed": 0,
    },
}

_EXPECTED_TYPES: dict[str, type] = {
    "hypofuse.nbest": NBestList,
    "hypofuse.reference": ReferenceTranscript,
    "hypofuse.system": SystemMetadata,
    "hypofuse.fusion_run": FusionRun,
    "hypofuse.report": ReportRecord,
}


def test_convert_row_all_schemas() -> None:
    """Every known schema dispatches to the correct dataclass type."""
    for schema_name in KNOWN_SCHEMAS:
        assert schema_name in _SAMPLES, f"missing sample for {schema_name}"
        result = convert_row(_SAMPLES[schema_name])
        assert isinstance(result, _EXPECTED_TYPES[schema_name])


def test_convert_row_unknown_schema() -> None:
    with pytest.raises(SchemaError, match="unknown schema"):
        convert_row({"schema": "fake.schema", "id": "x"})


def test_convert_row_missing_schema() -> None:
    with pytest.raises(SchemaError, match="missing"):
        convert_row({"utterance_id": "u1"})


def test_convert_row_lists_supported() -> None:
    """Error message for unknown schema lists the supported names."""
    with pytest.raises(SchemaError, match=r"hypofuse.nbest"):
        convert_row({"schema": "nope"})

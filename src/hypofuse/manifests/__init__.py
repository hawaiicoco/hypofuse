"""Manifest types and JSON schema versioning.

Manifest records are encoded as JSON Lines. Every record carries a
schema name and a major version integer so downstream tooling can
detect breaking changes and round-trip safely.
"""

from __future__ import annotations

from hypofuse.manifests.fusion_run import FusionArc, FusionRun
from hypofuse.manifests.nbest import NBestHypothesis, NBestList
from hypofuse.manifests.reference import ReferenceTranscript
from hypofuse.manifests.reporter import ReportRecord
from hypofuse.manifests.system import SystemMetadata

SCHEMA_VERSION = 1

SCHEMA_NBEST = "hypofuse.nbest"
SCHEMA_REFERENCE = "hypofuse.reference"
SCHEMA_SYSTEM = "hypofuse.system"
SCHEMA_FUSION_RUN = "hypofuse.fusion_run"
SCHEMA_REPORT = "hypofuse.report"

KNOWN_SCHEMAS = frozenset(
    {
        SCHEMA_NBEST,
        SCHEMA_REFERENCE,
        SCHEMA_SYSTEM,
        SCHEMA_FUSION_RUN,
        SCHEMA_REPORT,
    }
)

SCHEMA_VERSIONS = {
    SCHEMA_NBEST: 1,
    SCHEMA_REFERENCE: 1,
    SCHEMA_SYSTEM: 1,
    SCHEMA_FUSION_RUN: 1,
    SCHEMA_REPORT: 1,
}

__all__ = [
    "KNOWN_SCHEMAS",
    "SCHEMA_FUSION_RUN",
    "SCHEMA_NBEST",
    "SCHEMA_REFERENCE",
    "SCHEMA_REPORT",
    "SCHEMA_SYSTEM",
    "SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "FusionArc",
    "FusionRun",
    "NBestHypothesis",
    "NBestList",
    "ReferenceTranscript",
    "ReportRecord",
    "SystemMetadata",
]

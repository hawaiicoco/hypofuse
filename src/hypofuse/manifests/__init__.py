"""Manifest types and JSON schema versioning.

Manifest records are encoded as JSON Lines. Every record carries a
schema name and a major version integer so downstream tooling can
detect breaking changes and round-trip safely.
"""

from __future__ import annotations

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

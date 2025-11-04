"""Strict field/type validation for manifest records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from hypofuse.exceptions import DuplicateIdError, SchemaError

from . import (
    KNOWN_SCHEMAS,
    SCHEMA_FUSION_RUN,
    SCHEMA_NBEST,
    SCHEMA_REFERENCE,
    SCHEMA_SYSTEM,
    SCHEMA_VERSIONS,
)

_STRING_FIELDS = {
    SCHEMA_NBEST: frozenset({"utterance_id", "system", "language"}),
    SCHEMA_REFERENCE: frozenset({"utterance_id"}),
    SCHEMA_SYSTEM: frozenset({"system_id", "language"}),
    SCHEMA_FUSION_RUN: frozenset({"utterance_id", "policy"}),
}


def _require_str(row: Mapping[str, Any], name: str) -> str:
    if name not in row:
        raise SchemaError(f"missing required string field: {name}")
    value = row[name]
    if not isinstance(value, str):
        raise SchemaError(f"field {name} must be a string, got {type(value).__name__}")
    if not value:
        raise SchemaError(f"field {name} must be non-empty")
    return value


def _require_num(row: Mapping[str, Any], name: str) -> float:
    if name not in row:
        raise SchemaError(f"missing required numeric field: {name}")
    value = row[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"field {name} must be numeric, got {type(value).__name__}")
    return float(value)


def validate_record(row: Mapping[str, Any]) -> str:
    """Validate a manifest record and return its schema name.

    Raises SchemaError on missing or wrong-type fields.
    """
    if "schema" not in row:
        raise SchemaError("record is missing schema field")
    schema = row["schema"]
    if not isinstance(schema, str):
        raise SchemaError("schema field must be a string")
    if schema not in KNOWN_SCHEMAS:
        raise SchemaError(f"unknown schema: {schema!r}")
    for field_name in _STRING_FIELDS.get(schema, ()):
        _require_str(row, field_name)
    if schema == SCHEMA_NBEST:
        for field_name in ("acoustic_log10", "lm_log10"):
            value = row.get(field_name)
            if field_name in row and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise SchemaError(f"field {field_name} must be numeric")
    # schema_version: missing is treated as 1, unknown is rejected
    version = row.get("schema_version")
    if version is not None:
        if not isinstance(version, int) or isinstance(version, bool):
            raise SchemaError("schema_version must be an integer")
        max_version = SCHEMA_VERSIONS.get(schema, 1)
        if version < 1 or version > max_version:
            raise SchemaError(
                f"unknown schema_version {version} for {schema!r} (max known: {max_version})"
            )
    return schema


def reject_duplicate_ids(rows: Iterable[Mapping[str, Any]], id_field: str = "utterance_id") -> None:
    seen: set[tuple[str, str]] = set()
    for idx, row in enumerate(rows):
        value = row.get(id_field)
        if value is None:
            continue
        schema = row.get("schema", "")
        key = (schema, value)
        if key in seen:
            raise DuplicateIdError(
                f"duplicate {id_field}={value!r} in schema {schema!r} at row {idx}"
            )
        seen.add(key)

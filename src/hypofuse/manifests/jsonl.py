"""JSON Lines read/write helpers for manifest files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hypofuse.exceptions import SchemaError
from hypofuse.util import json_default
from hypofuse.util import read_jsonl as _read
from hypofuse.util import write_jsonl as _write

from .validate import reject_duplicate_ids, validate_record


def read_manifest(path: Path | str) -> list[dict[str, Any]]:
    """Read every JSONL row from a manifest file.

    Each row is validated against its declared schema. Duplicate utterance
    ids raise ``DuplicateIdError``. Errors carry the offending path.
    """
    target = Path(path)
    rows = _read(target)
    for idx, row in enumerate(rows):
        try:
            validate_record(row)
        except SchemaError as exc:
            raise SchemaError(f"invalid record at {target}:{idx + 1}: {exc}") from exc
    reject_duplicate_ids(rows)
    return rows


def write_manifest(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write manifest rows as JSON Lines, validating as we go."""
    target = Path(path)
    rendered: list[dict[str, Any]] = []
    for row in rows:
        validate_record(dict(row))
        rendered.append(
            json.loads(
                json.dumps(dict(row), sort_keys=True, ensure_ascii=False, default=json_default)
            )
        )
    _write(target, rendered)

"""JSON Lines read/write helpers for manifest files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hypofuse.exceptions import SchemaError
from hypofuse.util import json_default
from hypofuse.util import read_jsonl as _read

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
    """Write manifest rows as JSON Lines, validating as we go.

    Uses atomic write (temp file + rename) so a failure mid-write
    never leaves a partial file at the target path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                validate_record(dict(row))
                line = json.dumps(
                    dict(row),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=json_default,
                )
                # round-trip to catch serialization issues
                json.loads(line)
                fh.write(line)
                fh.write("\n")
        tmp.rename(target)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise

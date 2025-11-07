"""Audio-path safety checks for manifests.

Policy
------
Audio paths stored in manifests are **relative** paths under the
manifest file's directory.  The following forms are rejected:

* Absolute paths (leading ``/`` or ``~``)
* Parent-directory traversal (``..`` segments)
* Windows-style drive letters (``C:``, ``D:\\`` etc.)
* NUL bytes, double slashes, and unsafe characters in segments
* Empty segment names

Backslash separators are normalised to forward slashes on POSIX.
Empty strings are allowed (audio is optional).
"""

from __future__ import annotations

import re
from pathlib import Path

from hypofuse.exceptions import SchemaError

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._\-]+$")
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def normalize_audio_path(raw: str) -> str:
    """Return an audio path with forward slashes and no traversal segments.

    Empty strings are allowed (audio may be optional); the function rejects
    absolute paths, parent-directory escapes, NUL bytes and double slashes.
    """
    if raw == "":
        return raw
    if "\0" in raw:
        raise SchemaError("audio_path contains a NUL byte")
    if raw.startswith(("/", "~")):
        raise SchemaError("audio_path must be relative")
    if _DRIVE_LETTER.match(raw):
        raise SchemaError("audio_path contains a drive letter")
    cleaned = raw.replace("\\", "/")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    parts = cleaned.split("/")
    if any(p in {"", ".", ".."} for p in parts[:-1]) or parts[-1] in {".", ".."}:
        raise SchemaError("audio_path contains empty or parent-directory segments")
    for segment in parts:
        if segment and not _SAFE_SEGMENT.match(segment):
            raise SchemaError(f"audio_path segment {segment!r} has unsafe characters")
    return cleaned


def safe_audio_join(base: Path | str, raw: str) -> Path:
    """Join base + normalized audio path, refusing escapes."""
    base_p = Path(base).resolve()
    rel = normalize_audio_path(raw)
    candidate = (base_p / rel).resolve() if rel else base_p
    if base_p != candidate and base_p not in candidate.parents:
        raise SchemaError(f"audio_path {raw!r} escapes base {base_p}")
    return candidate

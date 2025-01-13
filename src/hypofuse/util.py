"""Shared helpers: hashing, seeding, deterministic serialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any


def stable_hash(obj: Any) -> str:
    """Return a deterministic SHA-256 hex digest for a JSON-serializable object.

    Uses sorted keys, no whitespace, UTF-8 encoding. Two equal inputs always
    produce the same digest; two different inputs produce different digests in
    practice (collisions still require 2^128 work).
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seeded(seed: int | None) -> random.Random:
    """Return a Random instance seeded with `seed`.

    If `seed` is None, the returned RNG is *still* a deterministic instance
    with no shared global state, so downstream code can rely on it.
    """
    rng = random.Random()
    if seed is not None:
        rng.seed(seed, version=2)
    return rng


def relative_within(base: Path | str, target: Path | str) -> str:
    """Return `target` as a path relative to `base`, refusing to escape `base`.

    Raises ValueError if `target` resolves outside `base`.
    """
    base_p = Path(base).resolve()
    target_p = Path(target).resolve()
    try:
        rel = target_p.relative_to(base_p)
    except ValueError as exc:
        raise ValueError(f"path {target_p} is outside base {base_p}") from exc
    return str(rel)


def safe_join(base: Path | str, *parts: str) -> Path:
    """Join parts under base, refusing any traversal sequence."""
    base_p = Path(base).resolve()
    candidate = base_p.joinpath(*parts).resolve()
    if base_p != candidate and base_p not in candidate.parents:
        raise ValueError(f"path {candidate} escapes base {base_p}")
    return candidate


def json_default(value: Any) -> Any:
    """Default JSON encoder for numpy / dataclass values."""
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise TypeError(f"non-finite float is not JSON-safe: {value!r}")
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    """Write JSON Lines with deterministic UTF-8 + LF endings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default))
            fh.write("\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read JSON Lines; raise ValueError on parse failure with line number."""
    target = Path(path)
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {target}:{lineno}: {exc.msg}") from exc
    return rows


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

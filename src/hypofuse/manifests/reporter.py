"""Reporter / report manifest record."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportRecord:
    """Result of an analysis run that emits structured outputs.

    `artifact_paths` are file paths relative to the report's root directory
    so the record can move freely between machines. Run metadata is stored
    inline so reports can be re-rendered later.
    """

    report_id: str
    schema: str
    generated_at: str
    hypofuse_version: str
    config_hash: str
    seed: int
    artifact_paths: tuple[str, ...] = ()
    metrics: tuple[tuple[str, float], ...] = ()

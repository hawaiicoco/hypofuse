"""System / recognizer metadata records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemMetadata:
    """Static metadata describing a recognizer or system variant."""

    system_id: str
    language: str
    vocabulary_size: int = 0
    description: str = ""
    acoustic_model: str = ""
    language_model: str = ""
    decoder: str = ""
    version: str = ""
